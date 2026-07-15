// ---------------------------------------------------------------------------
//  main.cpp  --  nanoGPT C++ driver: train / sample a character-level GPT,
//  plus a self-contained gradient check.
//
//  Subcommands:
//    nanogpt train [input.txt] [options]
//        Train a char-level GPT on a text file, print loss, sample, save ckpt.
//    nanogpt sample [ckpt.bin] [options]
//        Load a trained char model and generate text.
//
//  Options (train): --steps N --lr F --batch N --block N --layers N --embd N
//                   --heads N --out FILE --eval-every N --seed N
//                   --init scratch|resume|finetune --ckpt FILE
//                   --warmup N --min-lr F --decay-iters N --no-lr-decay --grad-clip F
//                   --grad-accum N  (micro-batches per optimiser step)
//        resume:   continue training a saved model (params + AdamW state + step)
//        finetune: keep the saved weights, fresh optimiser/step, train on new data
//        lr:       cosine decay with linear warmup (nanoGPT get_lr); grad clipped
//                  by global L2 norm to --grad-clip (0 disables)
//  Options (sample): --tokens N --temp F --topk N --prompt STR --seed N
// ---------------------------------------------------------------------------
#include "gpt.h"
#include "tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>

using namespace gpt;

// ---- checkpoint I/O (params always stored as float32 for portability) -----
//  v1 = magic + ver + hdr[6] + vocab + np + params
//  v2 = v1 layout, then: adam_t(int32) iter(int32) m_mem(f32*np) v_mem(f32*np)
//       so training can resume with AdamW momentum + step count intact.
static const char CKPT_MAGIC[4] = {'G', 'P', 'T', 'c'};

static bool save_checkpoint(const std::string& path, const GPT& m, const CharTokenizer& tok,
                            int32_t iter = 0) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t ver = 2, hdr[6] = {m.config.block_size, m.config.vocab_size, m.config.n_layer,
                               m.config.n_head, m.config.n_embd, m.config.bias ? 1 : 0};
    f.write(CKPT_MAGIC, 4);
    f.write((char*)&ver, 4);
    f.write((char*)hdr, sizeof(hdr));
    int32_t vn = (int32_t)tok.itos.size();
    f.write((char*)&vn, 4);
    if (vn) f.write(tok.itos.data(), vn);
    int64_t np = (int64_t)m.num_params;
    f.write((char*)&np, 8);
    // write params as float32
    std::vector<float> buf(m.num_params);
    for (size_t i = 0; i < m.num_params; i++) buf[i] = (float)m.params_mem[i];
    f.write((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
    // v2: AdamW state + step. If the optimiser never ran, write zeros.
    int32_t adam_t = m.adam_t;
    f.write((char*)&adam_t, 4);
    f.write((char*)&iter, 4);
    if (m.m_mem.size() == m.num_params && m.v_mem.size() == m.num_params) {
        for (size_t i = 0; i < m.num_params; i++) buf[i] = (float)m.m_mem[i];
        f.write((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
        for (size_t i = 0; i < m.num_params; i++) buf[i] = (float)m.v_mem[i];
        f.write((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
    } else {
        std::vector<float> zero(m.num_params, 0.0f);
        f.write((char*)zero.data(), (std::streamsize)(zero.size() * sizeof(float)));
        f.write((char*)zero.data(), (std::streamsize)(zero.size() * sizeof(float)));
    }
    return (bool)f;
}

// Loads params (all versions). If out_iter != nullptr and the file is v2, also
// restores AdamW momentum into the model and returns the saved step via *out_iter.
static bool load_checkpoint(const std::string& path, GPT& m, CharTokenizer& tok,
                            int32_t* out_iter = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    char magic[4]; int32_t ver, hdr[6];
    f.read(magic, 4); f.read((char*)&ver, 4); f.read((char*)hdr, sizeof(hdr));
    if (std::memcmp(magic, CKPT_MAGIC, 4) != 0) { std::fprintf(stderr, "bad ckpt magic\n"); return false; }
    GPTConfig c;
    c.block_size = hdr[0]; c.vocab_size = hdr[1]; c.n_layer = hdr[2];
    c.n_head = hdr[3]; c.n_embd = hdr[4]; c.bias = hdr[5] != 0;
    int32_t vn; f.read((char*)&vn, 4);
    std::vector<char> chars(vn); if (vn) f.read(chars.data(), vn);
    tok.set_vocab(chars);
    int64_t np; f.read((char*)&np, 8);
    m.build(c);
    if ((int64_t)m.num_params != np) { std::fprintf(stderr, "param count mismatch\n"); return false; }
    std::vector<float> buf(m.num_params);
    f.read((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
    for (size_t i = 0; i < m.num_params; i++) m.params_mem[i] = (real)buf[i];
    if (out_iter) *out_iter = 0;
    if (ver >= 2) {
        int32_t adam_t = 0, iter = 0;
        f.read((char*)&adam_t, 4);
        f.read((char*)&iter, 4);
        m.m_mem.assign(m.num_params, (real)0);
        m.v_mem.assign(m.num_params, (real)0);
        f.read((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
        for (size_t i = 0; i < m.num_params; i++) m.m_mem[i] = (real)buf[i];
        f.read((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
        for (size_t i = 0; i < m.num_params; i++) m.v_mem[i] = (real)buf[i];
        m.adam_t = adam_t;
        if (out_iter) *out_iter = iter;
    }
    return (bool)f;
}

// ---- text generation -------------------------------------------------------
// sample one token id from raw logits with temperature + top-k (in-place scratch).
static int sample_token(std::vector<real>& l, real temperature, int top_k, int V,
                        std::mt19937& rng) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (auto& x : l) x /= (temperature > 0 ? temperature : (real)1);
    if (top_k > 0 && top_k < V) {
        std::vector<real> s(l); std::nth_element(s.begin(), s.end() - top_k, s.end());
        real thresh = s[s.size() - top_k];
        for (auto& x : l) if (x < thresh) x = (real)-1e30;
    }
    real mx = *std::max_element(l.begin(), l.end());
    double sum = 0; for (auto& x : l) { x = (real)std::exp((double)(x - mx)); sum += x; }
    double r = uni(rng) * sum, acc = 0; int next = V - 1;
    for (int i = 0; i < V; i++) { acc += l[i]; if (acc >= r) { next = i; break; } }
    return next;
}

// Autoregressive generation. Uses the KV cache while the sequence fits in the
// context window (fast: one O(1) step per token); falls back to the recompute-
// with-sliding path once it exceeds block_size (learned pos-emb can't slide).
static std::vector<int> generate(GPT& m, std::vector<int> idx, int max_new,
                                 real temperature, int top_k, std::mt19937& rng) {
    int V = m.config.vocab_size, bs = m.config.block_size;
    int produced = 0;
    if ((int)idx.size() <= bs) {
        KVCache kv; kv.init(m.config);
        std::vector<real> logits;
        // prime the cache with the prompt (last bs tokens fit by construction)
        for (size_t i = 0; i < idx.size() && kv.pos < bs; i++) m.forward_one(idx[i], kv, logits);
        while (produced < max_new && kv.pos < bs) {
            int next = sample_token(logits, temperature, top_k, V, rng);
            idx.push_back(next); produced++;
            if (kv.pos < bs && produced < max_new) m.forward_one(next, kv, logits);
        }
    }
    // fallback for anything beyond the context window
    for (; produced < max_new; produced++) {
        int t = (int)idx.size();
        int start = std::max(0, t - bs);
        int tc = t - start;
        std::vector<int> cond(idx.begin() + start, idx.end());
        m.forward(cond.data(), nullptr, 1, tc);
        const real* logit = m.acts.logits + (size_t)(tc - 1) * V;   // last position
        std::vector<real> l(logit, logit + V);
        idx.push_back(sample_token(l, temperature, top_k, V, rng));
    }
    return idx;
}

// ---- estimate loss on a split ---------------------------------------------
static real estimate_loss(GPT& m, const std::vector<int>& data, int B, int T,
                          int iters, std::mt19937& rng) {
    std::uniform_int_distribution<int> pick(0, (int)data.size() - T - 1);
    std::vector<int> inp(B * T), tgt(B * T);
    double sum = 0;
    for (int it = 0; it < iters; it++) {
        for (int b = 0; b < B; b++) {
            int off = pick(rng);
            for (int t = 0; t < T; t++) { inp[b * T + t] = data[off + t]; tgt[b * T + t] = data[off + t + 1]; }
        }
        m.forward(inp.data(), tgt.data(), B, T);
        sum += m.mean_loss;
    }
    return (real)(sum / iters);
}

static int arg_i(int argc, char** argv, const char* name, int def) {
    for (int i = 1; i < argc - 1; i++) if (!std::strcmp(argv[i], name)) return std::atoi(argv[i + 1]);
    return def;
}
static double arg_f(int argc, char** argv, const char* name, double def) {
    for (int i = 1; i < argc - 1; i++) if (!std::strcmp(argv[i], name)) return std::atof(argv[i + 1]);
    return def;
}
static std::string arg_s(int argc, char** argv, const char* name, const std::string& def) {
    for (int i = 1; i < argc - 1; i++) if (!std::strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static bool arg_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; i++) if (!std::strcmp(argv[i], name)) return true;
    return false;
}

// Learning-rate schedule: linear warmup then cosine decay to min_lr, matching
// nanoGPT's get_lr() in train.py. `it` is the (global) step index.
static double get_lr(int it, double lr, double min_lr, int warmup, int decay_iters) {
    const double PI = 3.14159265358979323846;
    if (it < warmup) return lr * (it + 1) / (double)(warmup + 1);   // 1) warmup
    if (it >= decay_iters) return min_lr;                            // 2) past decay
    double ratio = (double)(it - warmup) / (double)(decay_iters - warmup);
    double coeff = 0.5 * (1.0 + std::cos(PI * ratio));               // 3) cosine, 1->0
    return min_lr + coeff * (lr - min_lr);
}

// ---- train -----------------------------------------------------------------
static int cmd_train(int argc, char** argv) {
    std::string input = (argc > 2 && argv[2][0] != '-') ? argv[2] : "input.txt";
    int steps = arg_i(argc, argv, "--steps", 2000);
    int B = arg_i(argc, argv, "--batch", 32);
    int T = arg_i(argc, argv, "--block", 64);
    int n_layer = arg_i(argc, argv, "--layers", 4);
    int n_embd = arg_i(argc, argv, "--embd", 128);
    int n_head = arg_i(argc, argv, "--heads", 4);
    double lr = arg_f(argc, argv, "--lr", 1e-3);
    int eval_every = arg_i(argc, argv, "--eval-every", 250);
    uint32_t seed = (uint32_t)arg_i(argc, argv, "--seed", 1337);
    std::string out = arg_s(argc, argv, "--out", "ckpt.bin");
    // LR schedule (cosine warmup) + gradient clipping, as in nanoGPT's train.py.
    double min_lr = arg_f(argc, argv, "--min-lr", -1);       // <0 => lr/10
    int warmup = arg_i(argc, argv, "--warmup", -1);          // <0 => max(1, steps/10)
    int decay_iters = arg_i(argc, argv, "--decay-iters", -1); // <0 => final step
    bool lr_decay = !arg_flag(argc, argv, "--no-lr-decay");   // on by default
    double grad_clip = arg_f(argc, argv, "--grad-clip", 1.0); // 0 disables
    int accum = arg_i(argc, argv, "--grad-accum", 1);         // micro-batches per step
    if (accum < 1) accum = 1;
    // scratch (default) | resume (continue: params+optimiser+step) |
    // finetune (params only, fresh optimiser + step counter, possibly new data)
    std::string init = arg_s(argc, argv, "--init", "scratch");
    std::string ckpt = arg_s(argc, argv, "--ckpt", "");
    bool is_resume = (init == "resume");
    bool is_finetune = (init == "finetune");
    if ((is_resume || is_finetune) && ckpt.empty()) {
        std::fprintf(stderr, "--init %s requires --ckpt FILE\n", init.c_str()); return 2;
    }
    if (init != "scratch" && !is_resume && !is_finetune) {
        std::fprintf(stderr, "--init must be scratch, resume or finetune\n"); return 2;
    }

    std::ifstream f(input, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s (run: get the tiny shakespeare input.txt)\n", input.c_str()); return 1; }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::printf("dataset: %zu characters\n", text.size());

    GPT m;
    CharTokenizer tok;
    int32_t start_iter = 0;
    if (is_resume || is_finetune) {
        // reuse the checkpoint's vocab (baked in); encode() drops unseen chars.
        if (!load_checkpoint(ckpt, m, tok, &start_iter)) return 1;
        std::printf("%s from %s: vocab %d, model %.2fM params, saved step %d\n",
                    init.c_str(), ckpt.c_str(), tok.vocab_size(), m.num_params / 1e6, start_iter);
        // block_size may be overridden per-run only if it fits the trained model.
        if (T > m.config.block_size) {
            std::printf("  (requested --block %d > model block %d; clamping to %d)\n",
                        T, m.config.block_size, m.config.block_size);
            T = m.config.block_size;
        }
        if (is_finetune) { m.m_mem.clear(); m.v_mem.clear(); m.adam_t = 0; start_iter = 0; }
    } else {
        tok.build_from_text(text);
        GPTConfig c;
        c.block_size = T; c.vocab_size = tok.vocab_size();
        c.n_layer = n_layer; c.n_head = n_head; c.n_embd = n_embd; c.bias = false;
        m.build(c); m.init_random(seed);
        std::printf("scratch model: %d layers, %d embd, %d heads, block %d  (%.2fM params)\n",
                    n_layer, n_embd, n_head, T, m.num_params / 1e6);
    }
    std::printf("vocab size: %d\n", tok.vocab_size());
    std::vector<int> data = tok.encode(text);
    if ((int)data.size() < T + 2) { std::fprintf(stderr, "dataset too small for block %d\n", T); return 1; }
    size_t n = data.size();
    std::vector<int> train(data.begin(), data.begin() + (size_t)(n * 0.9));
    std::vector<int> val(data.begin() + (size_t)(n * 0.9), data.end());

    // resolve schedule defaults now that start_iter and steps are known.
    if (min_lr < 0) min_lr = lr / 10.0;
    if (warmup < 0) warmup = std::max(1, steps / 10);
    if (decay_iters < 0) decay_iters = start_iter + steps;  // decay reaches min_lr at the end
    if (lr_decay)
        std::printf("lr: cosine %.2e -> %.2e (warmup %d, decay to step %d), grad_clip %.2g\n",
                    lr, min_lr, warmup, decay_iters, grad_clip);
    else
        std::printf("lr: constant %.2e, grad_clip %.2g\n", lr, grad_clip);
    if (accum > 1)
        std::printf("grad accumulation: %d micro-batches -> effective batch %d\n", accum, B * accum);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, (int)train.size() - T - 1);
    std::vector<int> inp(B * T), tgt(B * T);

    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step <= steps; step++) {
        if (step % eval_every == 0 || step == steps) {
            std::mt19937 er(4242);
            real tl = estimate_loss(m, train, B, T, 20, er);
            real vl = estimate_loss(m, val, B, T, 20, er);
            auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();
            std::printf("step %5d | train loss %.4f | val loss %.4f | %.1fs\n",
                        start_iter + step, tl, vl, secs);
            std::fflush(stdout);
        }
        if (step == steps) break;

        // gradient accumulation: sum grads over `accum` micro-batches, then one
        // optimiser step on the average (simulates batch B*accum, as in nanoGPT).
        m.zero_grad();
        for (int micro = 0; micro < accum; micro++) {
            for (int b = 0; b < B; b++) {
                int off = pick(rng);
                for (int t = 0; t < T; t++) { inp[b * T + t] = train[off + t]; tgt[b * T + t] = train[off + t + 1]; }
            }
            m.forward(inp.data(), tgt.data(), B, T);
            m.zero_grad_acts();     // keep param-grad accumulator, reset scratch
            m.backward();
        }
        if (accum > 1) m.scale_grads((real)(1.0 / accum));
        // gradient clipping by global L2 norm (nanoGPT clip_grad_norm_)
        if (grad_clip > 0) {
            double sq = 0;
            for (size_t i = 0; i < m.num_params; i++) { double g = m.grads_mem[i]; sq += g * g; }
            double norm = std::sqrt(sq);
            if (norm > grad_clip) {
                real scale = (real)(grad_clip / (norm + 1e-6));
                for (size_t i = 0; i < m.num_params; i++) m.grads_mem[i] *= scale;
            }
        }
        double cur_lr = lr_decay ? get_lr(start_iter + step, lr, min_lr, warmup, decay_iters) : lr;
        m.adamw((real)cur_lr, (real)0.9, (real)0.99, (real)1e-8, (real)0.1);
    }

    if (save_checkpoint(out, m, tok, start_iter + steps))
        std::printf("saved checkpoint to %s (step %d)\n", out.c_str(), start_iter + steps);
    else std::fprintf(stderr, "failed to save checkpoint\n");

    // a quick sample
    std::mt19937 grng(seed + 1);
    std::vector<int> ctx = {tok.stoi.count('\n') ? tok.stoi.at('\n') : 0};
    auto gen = generate(m, ctx, 300, (real)0.8, 40, grng);
    std::printf("\n--- sample ---\n%s\n", tok.decode(gen).c_str());
    return 0;
}

// ---- sample ----------------------------------------------------------------
static int cmd_sample(int argc, char** argv) {
    std::string ckpt = (argc > 2 && argv[2][0] != '-') ? argv[2] : "ckpt.bin";
    int tokens = arg_i(argc, argv, "--tokens", 500);
    double temp = arg_f(argc, argv, "--temp", 0.8);
    int topk = arg_i(argc, argv, "--topk", 40);
    uint32_t seed = (uint32_t)arg_i(argc, argv, "--seed", 1337);
    std::string prompt = arg_s(argc, argv, "--prompt", "");

    GPT m; CharTokenizer tok;
    if (!load_checkpoint(ckpt, m, tok)) return 1;
    std::printf("loaded %s: %d layers, %d embd, vocab %d\n",
                ckpt.c_str(), m.config.n_layer, m.config.n_embd, m.config.vocab_size);

    std::vector<int> ctx;
    if (!prompt.empty()) ctx = tok.encode(prompt);
    if (ctx.empty()) ctx.push_back(tok.stoi.count('\n') ? tok.stoi.at('\n') : 0);

    std::mt19937 rng(seed);
    auto gen = generate(m, ctx, tokens, (real)temp, topk, rng);
    std::printf("%s\n", tok.decode(gen).c_str());
    return 0;
}

// Load a checkpoint, run one forward pass on a token-id file, dump logits.
// Used to numerically compare against PyTorch (nanoGPT). See verify_nanogpt_compat.py.
//   nanogpt verify ckpt.bin idx.bin out_logits.bin
static int cmd_verify(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s verify ckpt.bin idx.bin out_logits.bin\n", argv[0]); return 2; }
    GPT m; CharTokenizer tok;
    if (!load_checkpoint(argv[2], m, tok)) return 1;
    std::ifstream fi(argv[3], std::ios::binary);
    if (!fi) { std::fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
    int32_t T = 0; fi.read((char*)&T, 4);
    std::vector<int32_t> ids(T); fi.read((char*)ids.data(), (std::streamsize)T * 4);
    std::vector<int> idx(ids.begin(), ids.end());
    m.forward(idx.data(), nullptr, 1, T);
    int V = m.config.vocab_size;
    std::vector<float> out((size_t)T * V);
    for (size_t i = 0; i < out.size(); i++) out[i] = (float)m.acts.logits[i];
    std::ofstream fo(argv[4], std::ios::binary);
    fo.write((char*)out.data(), (std::streamsize)(out.size() * sizeof(float)));
    std::printf("wrote %d x %d logits to %s\n", T, V, argv[4]);
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "train")     return cmd_train(argc, argv);
    if (mode == "sample")    return cmd_sample(argc, argv);
    if (mode == "verify")    return cmd_verify(argc, argv);
    std::fprintf(stderr,
        "usage:\n"
        "  %s train [input.txt] [--steps N --lr F --batch N --block N --layers N --embd N --heads N --out FILE\n"
        "                        --init scratch|resume|finetune --ckpt FILE\n"
        "                        --warmup N --min-lr F --decay-iters N --no-lr-decay --grad-clip F\n"
        "                        --grad-accum N]\n"
        "  %s sample [ckpt.bin] [--tokens N --temp F --topk N --prompt STR]\n"
        "\n(gradient check: build and run the nanogpt_gradcheck target)\n",
        argv[0], argv[0]);
    return 2;
}
