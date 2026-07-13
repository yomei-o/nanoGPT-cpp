// Gradient check for gpt.h. Build with -DGPT_USE_DOUBLE for crisp results.
#include "gpt.h"
#include <cstdio>
#include <random>
#include <algorithm>
using namespace gpt;

int main() {
    GPTConfig c;
    c.block_size = 8; c.vocab_size = 11; c.n_layer = 2; c.n_head = 2; c.n_embd = 16; c.bias = true;
    int B = 2, T = 6;

    GPT model; model.build(c); model.init_random(42);

    std::mt19937 rng(123);
    std::vector<int> inp(B * T), tgt(B * T);
    for (auto& v : inp) v = std::uniform_int_distribution<int>(0, c.vocab_size - 1)(rng);
    for (auto& v : tgt) v = std::uniform_int_distribution<int>(0, c.vocab_size - 1)(rng);

    model.forward(inp.data(), tgt.data(), B, T);
    model.zero_grad();
    model.backward();

    // check a spread of parameters across every tensor
    struct Grp { const char* name; real* p; real* g; size_t n; };
    Grp groups[] = {
        {"wte", model.params.wte, model.grads.wte, model.param_sizes[0]},
        {"wpe", model.params.wpe, model.grads.wpe, model.param_sizes[1]},
        {"ln1w", model.params.ln1w, model.grads.ln1w, model.param_sizes[2]},
        {"ln1b", model.params.ln1b, model.grads.ln1b, model.param_sizes[3]},
        {"qkvw", model.params.qkvw, model.grads.qkvw, model.param_sizes[4]},
        {"qkvb", model.params.qkvb, model.grads.qkvb, model.param_sizes[5]},
        {"attprojw", model.params.attprojw, model.grads.attprojw, model.param_sizes[6]},
        {"attprojb", model.params.attprojb, model.grads.attprojb, model.param_sizes[7]},
        {"ln2w", model.params.ln2w, model.grads.ln2w, model.param_sizes[8]},
        {"fcw", model.params.fcw, model.grads.fcw, model.param_sizes[10]},
        {"fcb", model.params.fcb, model.grads.fcb, model.param_sizes[11]},
        {"fcprojw", model.params.fcprojw, model.grads.fcprojw, model.param_sizes[12]},
        {"lnfw", model.params.lnfw, model.grads.lnfw, model.param_sizes[14]},
        {"lnfb", model.params.lnfb, model.grads.lnfb, model.param_sizes[15]},
    };

    const real eps = (real)1e-5;
    real max_rel = 0;
    std::printf("gradient check (double precision, eps=1e-5):\n");
    for (auto& gr : groups) {
        real grp_max = 0;
        size_t stride = std::max((size_t)1, gr.n / 13);
        for (size_t i = 0; i < gr.n; i += stride) {
            real orig = gr.p[i];
            gr.p[i] = orig + eps; model.forward(inp.data(), tgt.data(), B, T); real lp = model.mean_loss;
            gr.p[i] = orig - eps; model.forward(inp.data(), tgt.data(), B, T); real lm = model.mean_loss;
            gr.p[i] = orig;
            real num = (lp - lm) / (2 * eps);
            real ana = gr.g[i];
            real denom = std::max((real)1e-9, std::fabs(num) + std::fabs(ana));
            real rel = std::fabs(num - ana) / denom;
            grp_max = std::max(grp_max, rel);
        }
        max_rel = std::max(max_rel, grp_max);
        std::printf("  %-9s max rel err = %.3e\n", gr.name, grp_max);
    }
    std::printf("overall max rel err = %.3e  ->  %s\n", max_rel, max_rel < 1e-4 ? "PASS" : "FAIL");
    return max_rel < 1e-4 ? 0 : 1;
}
