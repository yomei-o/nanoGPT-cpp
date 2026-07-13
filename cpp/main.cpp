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
static const char CKPT_MAGIC[4] = {'G', 'P', 'T', 'c'};

static bool save_checkpoint(const std::string& path, const GPT& m, const CharTokenizer& tok) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t ver = 1, hdr[6] = {m.config.block_size, m.config.vocab_size, m.config.n_layer,
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
    return (bool)f;
}

static bool load_checkpoint(const std::string& path, GPT& m, CharTokenizer& tok) {
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
    return (bool)f;
}

// ---- text generation -------------------------------------------------------
static std::vector<int> generate(GPT& m, std::vector<int> idx, int max_new,
                                 real temperature, int top_k, std::mt19937& rng) {
    int V = m.config.vocab_size, bs = m.config.block_size;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (int step = 0; step < max_new; step++) {
        int t = (int)idx.size();
        int start = std::max(0, t - bs);
        int tc = t - start;
        std::vector<int> cond(idx.begin() + start, idx.end());
        m.forward(cond.data(), nullptr, 1, tc);
        const real* logit = m.acts.logits + (size_t)(tc - 1) * V;   // last position
        std::vector<real> l(logit, logit + V);
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
        idx.push_back(next);
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

    std::ifstream f(input, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s (run: get the tiny shakespeare input.txt)\n", input.c_str()); return 1; }
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::printf("dataset: %zu characters\n", text.size());

    CharTokenizer tok; tok.build_from_text(text);
    std::printf("vocab size: %d\n", tok.vocab_size());
    std::vector<int> data = tok.encode(text);
    size_t n = data.size();
    std::vector<int> train(data.begin(), data.begin() + (size_t)(n * 0.9));
    std::vector<int> val(data.begin() + (size_t)(n * 0.9), data.end());

    GPTConfig c;
    c.block_size = T; c.vocab_size = tok.vocab_size();
    c.n_layer = n_layer; c.n_head = n_head; c.n_embd = n_embd; c.bias = false;
    GPT m; m.build(c); m.init_random(seed);
    std::printf("model: %d layers, %d embd, %d heads, block %d  (%.2fM params)\n",
                n_layer, n_embd, n_head, T, m.num_params / 1e6);

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
            std::printf("step %5d | train loss %.4f | val loss %.4f | %.1fs\n", step, tl, vl, secs);
            std::fflush(stdout);
        }
        if (step == steps) break;

        for (int b = 0; b < B; b++) {
            int off = pick(rng);
            for (int t = 0; t < T; t++) { inp[b * T + t] = train[off + t]; tgt[b * T + t] = train[off + t + 1]; }
        }
        m.forward(inp.data(), tgt.data(), B, T);
        m.zero_grad();
        m.backward();
        m.adamw((real)lr, (real)0.9, (real)0.99, (real)1e-8, (real)0.1);
    }

    if (save_checkpoint(out, m, tok)) std::printf("saved checkpoint to %s\n", out.c_str());
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

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "train")     return cmd_train(argc, argv);
    if (mode == "sample")    return cmd_sample(argc, argv);
    std::fprintf(stderr,
        "usage:\n"
        "  %s train [input.txt] [--steps N --lr F --batch N --block N --layers N --embd N --heads N --out FILE]\n"
        "  %s sample [ckpt.bin] [--tokens N --temp F --topk N --prompt STR]\n"
        "\n(gradient check: build and run the nanogpt_gradcheck target)\n",
        argv[0], argv[0]);
    return 2;
}
