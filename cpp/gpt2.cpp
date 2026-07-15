// ---------------------------------------------------------------------------
//  gpt2.cpp  --  load a real GPT-2 checkpoint (exported by export_gpt2.py) and
//  generate text OR fine-tune it, using the shared forward+backward pass in
//  gpt.h and the byte-level BPE tokenizer in bpe.h.
//
//  Usage:
//    gpt2 [model_gpt2.bin] [encoder.json] [vocab.bpe] [gen options]
//        Generate text (default). options: --prompt STR --tokens N --temp F
//                                          --topk N --seed N
//    gpt2 finetune model.bin data.txt [encoder.json] [vocab.bpe] [ft options]
//        Fine-tune the weights on a plain-text file (BPE-tokenised) and write a
//        new GPT-2 .bin that `gpt2` can generate from.
//        options: --steps N --lr F --batch N --block N --out FILE --eval-every N
//                 --warmup N --min-lr F --decay-iters N --no-lr-decay --grad-clip F
//                 --grad-accum N --init finetune|resume --seed N
// ---------------------------------------------------------------------------
#include "gpt.h"
#include "bpe.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <chrono>

using namespace gpt;

static const char G2_MAGIC[4] = {'G', 'P', 'T', '2'};

// Loads a GPT-2 .bin. If out_iter != nullptr and the file carries optimiser
// state (ver >= 2, written by save_gpt2), restores AdamW momentum + step too.
static bool load_gpt2(const std::string& path, GPT& m, int32_t* out_iter = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    char magic[4]; int32_t ver, hdr[7];
    f.read(magic, 4); f.read((char*)&ver, 4); f.read((char*)hdr, sizeof(hdr));
    if (std::memcmp(magic, G2_MAGIC, 4) != 0) { std::fprintf(stderr, "bad GPT2 magic\n"); return false; }
    GPTConfig c;
    c.block_size = hdr[0]; c.vocab_size = hdr[1]; c.n_layer = hdr[2];
    c.n_head = hdr[3]; c.n_embd = hdr[4]; c.bias = hdr[5] != 0; c.gelu_tanh = hdr[6] != 0;
    int64_t np; f.read((char*)&np, 8);
    m.build(c);
    if ((int64_t)m.num_params != np) {
        std::fprintf(stderr, "param mismatch: file %lld vs model %zu\n", (long long)np, m.num_params);
        return false;
    }
    std::vector<float> buf(m.num_params);
    f.read((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
    if (!f) { std::fprintf(stderr, "short read\n"); return false; }
    for (size_t i = 0; i < m.num_params; i++) m.params_mem[i] = (real)buf[i];
    std::printf("loaded GPT-2: %d layers, %d embd, %d heads, vocab %d, block %d, gelu_tanh=%d (%.1fM params)\n",
                c.n_layer, c.n_embd, c.n_head, c.vocab_size, c.block_size, c.gelu_tanh, m.num_params / 1e6);
    if (out_iter) *out_iter = 0;
    if (ver >= 2 && out_iter) {   // only fine-tune needs the optimiser state
        int32_t adam_t = 0, iter = 0;
        f.read((char*)&adam_t, 4);
        f.read((char*)&iter, 4);
        m.m_mem.assign(m.num_params, (real)0);
        m.v_mem.assign(m.num_params, (real)0);
        f.read((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
        for (size_t i = 0; i < m.num_params; i++) m.m_mem[i] = (real)buf[i];
        f.read((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
        for (size_t i = 0; i < m.num_params; i++) m.v_mem[i] = (real)buf[i];
        if (f) { m.adam_t = adam_t; if (out_iter) *out_iter = iter; }
    }
    return true;
}

// Saves a GPT-2 .bin (ver 2). The params block is byte-identical to the export
// format, so a plain `gpt2` generate still loads it (it ignores the trailing
// optimiser state); fine-tune --init resume can read that state back.
static bool save_gpt2(const std::string& path, const GPT& m, int32_t iter) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s for write\n", path.c_str()); return false; }
    int32_t ver = 2, hdr[7] = {m.config.block_size, m.config.vocab_size, m.config.n_layer,
                               m.config.n_head, m.config.n_embd, m.config.bias ? 1 : 0,
                               m.config.gelu_tanh ? 1 : 0};
    f.write(G2_MAGIC, 4);
    f.write((char*)&ver, 4);
    f.write((char*)hdr, sizeof(hdr));
    int64_t np = (int64_t)m.num_params;
    f.write((char*)&np, 8);
    std::vector<float> buf(m.num_params);
    for (size_t i = 0; i < m.num_params; i++) buf[i] = (float)m.params_mem[i];
    f.write((char*)buf.data(), (std::streamsize)(buf.size() * sizeof(float)));
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

static int arg_i(int argc, char** argv, const char* n, int d) {
    for (int i = 1; i < argc - 1; i++) if (!std::strcmp(argv[i], n)) return std::atoi(argv[i + 1]);
    return d;
}
static double arg_f(int argc, char** argv, const char* n, double d) {
    for (int i = 1; i < argc - 1; i++) if (!std::strcmp(argv[i], n)) return std::atof(argv[i + 1]);
    return d;
}
static std::string arg_s(int argc, char** argv, const char* n, const std::string& d) {
    for (int i = 1; i < argc - 1; i++) if (!std::strcmp(argv[i], n)) return argv[i + 1];
    return d;
}
static bool arg_flag(int argc, char** argv, const char* n) {
    for (int i = 1; i < argc; i++) if (!std::strcmp(argv[i], n)) return true;
    return false;
}

// cosine decay with linear warmup, matching nanoGPT's get_lr() (see main.cpp).
static double get_lr(int it, double lr, double min_lr, int warmup, int decay_iters) {
    const double PI = 3.14159265358979323846;
    if (it < warmup) return lr * (it + 1) / (double)(warmup + 1);
    if (it >= decay_iters) return min_lr;
    double ratio = (double)(it - warmup) / (double)(decay_iters - warmup);
    double coeff = 0.5 * (1.0 + std::cos(PI * ratio));
    return min_lr + coeff * (lr - min_lr);
}

// average cross-entropy over a few random windows of `data` (no grad).
static real eval_loss(GPT& m, const std::vector<int>& data, int B, int T,
                      int iters, std::mt19937& rng) {
    if ((int)data.size() < T + 1) return 0;
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

// sample one token id from raw logits with temperature + top-k (in-place scratch).
static int sample_token(std::vector<real>& l, double temp, int topk, int V, std::mt19937& rng) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (auto& x : l) x /= (temp > 0 ? (real)temp : (real)1);
    if (topk > 0 && topk < V) {
        std::vector<real> s(l); std::nth_element(s.begin(), s.end() - topk, s.end());
        real th = s[s.size() - topk];
        for (auto& x : l) if (x < th) x = (real)-1e30;
    }
    real mx = *std::max_element(l.begin(), l.end());
    double sum = 0; for (auto& x : l) { x = (real)std::exp((double)(x - mx)); sum += x; }
    double r = uni(rng) * sum, acc = 0; int next = V - 1;
    for (int i = 0; i < V; i++) { acc += l[i]; if (acc >= r) { next = i; break; } }
    return next;
}

// ---- generate (default subcommand) -----------------------------------------
static int cmd_generate(int argc, char** argv) {
    std::string model = (argc > 1 && argv[1][0] != '-') ? argv[1] : "model_gpt2.bin";
    std::string enc   = (argc > 2 && argv[2][0] != '-') ? argv[2] : "encoder.json";
    std::string bpe   = (argc > 3 && argv[3][0] != '-') ? argv[3] : "vocab.bpe";
    std::string prompt = arg_s(argc, argv, "--prompt", "Hello, I'm a language model,");
    int tokens = arg_i(argc, argv, "--tokens", 100);
    double temp = arg_f(argc, argv, "--temp", 0.8);
    int topk = arg_i(argc, argv, "--topk", 200);
    uint32_t seed = (uint32_t)arg_i(argc, argv, "--seed", 1337);

    GPT m;
    if (!load_gpt2(model, m)) return 1;
    BPETokenizer tok;
    if (!tok.load(enc, bpe)) { std::fprintf(stderr, "failed to load BPE files (%s, %s)\n", enc.c_str(), bpe.c_str()); return 1; }
    std::printf("BPE vocab: %zu tokens, %zu merges\n", tok.encoder.size(), tok.bpe_ranks.size());

    std::vector<int> idx = tok.encode(prompt);
    if (idx.empty()) idx.push_back(tok.encoder.count("\n") ? tok.encoder.at("\n") : 0);
    std::printf("prompt (%zu tokens): %s\n---\n", idx.size(), prompt.c_str());

    int V = m.config.vocab_size, bs = m.config.block_size;
    std::mt19937 rng(seed);
    std::fputs(prompt.c_str(), stdout);
    auto emit = [&](int next) {
        idx.push_back(next);
        std::string piece = tok.decode({next});
        std::fputs(piece.c_str(), stdout); std::fflush(stdout);
    };
    int produced = 0;
    // KV-cache fast path while the sequence fits the context window
    if ((int)idx.size() <= bs) {
        KVCache kv; kv.init(m.config);
        std::vector<real> logits;
        for (size_t i = 0; i < idx.size() && kv.pos < bs; i++) m.forward_one(idx[i], kv, logits);
        while (produced < tokens && kv.pos < bs) {
            int next = sample_token(logits, temp, topk, V, rng);
            emit(next); produced++;
            if (kv.pos < bs && produced < tokens) m.forward_one(next, kv, logits);
        }
    }
    // fallback: recompute-with-sliding beyond the context window
    for (; produced < tokens; produced++) {
        int t = (int)idx.size();
        int start = std::max(0, t - bs);
        int tc = t - start;
        std::vector<int> cond(idx.begin() + start, idx.end());
        m.forward(cond.data(), nullptr, 1, tc);
        const real* logit = m.acts.logits + (size_t)(tc - 1) * V;
        std::vector<real> l(logit, logit + V);
        emit(sample_token(l, temp, topk, V, rng));
    }
    std::printf("\n");
    return 0;
}

// ---- finetune --------------------------------------------------------------
//   gpt2 finetune model.bin data.txt [encoder.json] [vocab.bpe] [options]
static int cmd_finetune(int argc, char** argv) {
    // positionals after the "finetune" token
    std::string model = (argc > 2 && argv[2][0] != '-') ? argv[2] : "model_gpt2.bin";
    std::string data  = (argc > 3 && argv[3][0] != '-') ? argv[3] : "input.txt";
    std::string enc   = (argc > 4 && argv[4][0] != '-') ? argv[4] : "encoder.json";
    std::string bpe   = (argc > 5 && argv[5][0] != '-') ? argv[5] : "vocab.bpe";
    int steps = arg_i(argc, argv, "--steps", 200);
    int B = arg_i(argc, argv, "--batch", 1);
    int T = arg_i(argc, argv, "--block", 256);
    double lr = arg_f(argc, argv, "--lr", 3e-5);
    int eval_every = arg_i(argc, argv, "--eval-every", 20);
    uint32_t seed = (uint32_t)arg_i(argc, argv, "--seed", 1337);
    std::string out = arg_s(argc, argv, "--out", "model_gpt2_ft.bin");
    double min_lr = arg_f(argc, argv, "--min-lr", -1);          // <0 => lr/10
    int warmup = arg_i(argc, argv, "--warmup", -1);             // <0 => max(1, steps/10)
    int decay_iters = arg_i(argc, argv, "--decay-iters", -1);   // <0 => final step
    bool lr_decay = !arg_flag(argc, argv, "--no-lr-decay");
    double grad_clip = arg_f(argc, argv, "--grad-clip", 1.0);
    int accum = arg_i(argc, argv, "--grad-accum", 1);
    if (accum < 1) accum = 1;
    std::string init = arg_s(argc, argv, "--init", "finetune"); // finetune | resume

    GPT m;
    int32_t start_iter = 0;
    if (!load_gpt2(model, m, &start_iter)) return 1;
    if (init == "finetune") { m.m_mem.clear(); m.v_mem.clear(); m.adam_t = 0; start_iter = 0; }
    else if (init != "resume") { std::fprintf(stderr, "--init must be finetune or resume\n"); return 2; }

    if (T > m.config.block_size) {
        std::printf("clamping --block %d to model block_size %d\n", T, m.config.block_size);
        T = m.config.block_size;
    }

    BPETokenizer tok;
    if (!tok.load(enc, bpe)) { std::fprintf(stderr, "failed to load BPE files (%s, %s)\n", enc.c_str(), bpe.c_str()); return 1; }

    std::ifstream df(data, std::ios::binary);
    if (!df) { std::fprintf(stderr, "cannot open data file %s\n", data.c_str()); return 1; }
    std::string text((std::istreambuf_iterator<char>(df)), std::istreambuf_iterator<char>());
    std::printf("data: %zu chars, BPE-encoding...\n", text.size());
    std::vector<int> ids = tok.encode(text);
    std::printf("       -> %zu tokens\n", ids.size());
    if ((int)ids.size() < T + 2) { std::fprintf(stderr, "data too small for block %d (need > %d tokens)\n", T, T + 1); return 1; }
    size_t n = ids.size();
    std::vector<int> train(ids.begin(), ids.begin() + (size_t)(n * 0.9));
    std::vector<int> val(ids.begin() + (size_t)(n * 0.9), ids.end());

    if (min_lr < 0) min_lr = lr / 10.0;
    if (warmup < 0) warmup = std::max(1, steps / 10);
    if (decay_iters < 0) decay_iters = start_iter + steps;
    std::printf("finetune: %s, %d steps, batch %d, block %d, from step %d\n",
                init.c_str(), steps, B, T, start_iter);
    if (accum > 1) std::printf("grad accumulation: %d micro-batches -> effective batch %d\n", accum, B * accum);
    if (lr_decay)
        std::printf("lr: cosine %.2e -> %.2e (warmup %d, decay to %d), grad_clip %.2g\n",
                    lr, min_lr, warmup, decay_iters, grad_clip);
    else
        std::printf("lr: constant %.2e, grad_clip %.2g\n", lr, grad_clip);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, (int)train.size() - T - 1);
    std::vector<int> inp(B * T), tgt(B * T);
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step <= steps; step++) {
        if (step % eval_every == 0 || step == steps) {
            std::mt19937 er(4242);
            real vl = eval_loss(m, val, B, T, 4, er);
            auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();
            std::printf("step %5d | val loss %.4f | %.1fs\n", start_iter + step, vl, secs);
            std::fflush(stdout);
        }
        if (step == steps) break;

        m.zero_grad();
        for (int micro = 0; micro < accum; micro++) {
            for (int b = 0; b < B; b++) {
                int off = pick(rng);
                for (int t = 0; t < T; t++) { inp[b * T + t] = train[off + t]; tgt[b * T + t] = train[off + t + 1]; }
            }
            m.forward(inp.data(), tgt.data(), B, T);
            m.zero_grad_acts();
            m.backward();
        }
        if (accum > 1) m.scale_grads((real)(1.0 / accum));
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
        m.adamw((real)cur_lr, (real)0.9, (real)0.95, (real)1e-8, (real)0.1);
    }

    if (save_gpt2(out, m, start_iter + steps))
        std::printf("saved fine-tuned model to %s (step %d)\n", out.c_str(), start_iter + steps);
    else { std::fprintf(stderr, "failed to save\n"); return 1; }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "finetune")) return cmd_finetune(argc, argv);
    return cmd_generate(argc, argv);
}
