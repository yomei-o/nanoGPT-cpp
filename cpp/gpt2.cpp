// ---------------------------------------------------------------------------
//  gpt2.cpp  --  load a real GPT-2 checkpoint (exported by export_gpt2.py) and
//  generate text, using the shared forward pass in gpt.h and the byte-level
//  BPE tokenizer in bpe.h.
//
//  Usage:
//    gpt2 [model_gpt2.bin] [encoder.json] [vocab.bpe] [options]
//    options: --prompt STR --tokens N --temp F --topk N --seed N
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

using namespace gpt;

static const char G2_MAGIC[4] = {'G', 'P', 'T', '2'};

static bool load_gpt2(const std::string& path, GPT& m) {
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
    return true;
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

int main(int argc, char** argv) {
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
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::fputs(prompt.c_str(), stdout);
    for (int step = 0; step < tokens; step++) {
        int t = (int)idx.size();
        int start = std::max(0, t - bs);
        int tc = t - start;
        std::vector<int> cond(idx.begin() + start, idx.end());
        m.forward(cond.data(), nullptr, 1, tc);
        const real* logit = m.acts.logits + (size_t)(tc - 1) * V;
        std::vector<real> l(logit, logit + V);
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
        idx.push_back(next);
        // stream the newly produced token
        std::string piece = tok.decode({next});
        std::fputs(piece.c_str(), stdout); std::fflush(stdout);
    }
    std::printf("\n");
    return 0;
}
