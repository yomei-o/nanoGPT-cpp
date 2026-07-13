// ---------------------------------------------------------------------------
//  gpt.h  --  a dependency-free C++ port of nanoGPT's GPT model (model.py).
//
//  Implements the full GPT-2 architecture forward AND backward by hand:
//  token+position embeddings, LayerNorm, causal multi-head self-attention,
//  MLP with GELU, residual connections, a tied lm_head, and cross-entropy loss.
//  Plus an AdamW optimiser. No PyTorch, no Eigen, no BLAS -- only the C++
//  standard library. OpenMP is used if available (see CMake) but is optional.
//
//  A transformer is "just matmuls plus a few elementwise ops", so no transformer
//  library is required; this mirrors Karpathy's own llm.c / llama2.c.
//
//  Compute type `real` is float by default; define GPT_USE_DOUBLE for a
//  double-precision build (used by the gradient-check target for crisp checks).
// ---------------------------------------------------------------------------
#ifndef NANOGPT_GPT_H
#define NANOGPT_GPT_H

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gpt {

#ifdef GPT_USE_DOUBLE
using real = double;
#else
using real = float;
#endif

struct GPTConfig {
    int block_size = 1024;   // max sequence length
    int vocab_size = 50257;
    int n_layer = 12;
    int n_head = 12;
    int n_embd = 768;
    bool bias = true;        // bias in Linears / LayerNorms (GPT-2 uses true)
    bool gelu_tanh = false;  // false: exact erf GELU (nanoGPT default);
                             // true: tanh approx (gelu_new, what GPT-2 was trained with)
};

// ===========================================================================
//  Parameter tensors (flat buffer + offsets), matching nanoGPT / GPT-2.
//  nn.Linear weight layout is [out_features, in_features]; y = x @ W^T + b.
//  lm_head shares wte (weight tying), so it is not stored separately.
// ===========================================================================
enum { NUM_PARAM_TENSORS = 16 };
struct ParameterTensors {
    real* wte;      // (V, C)
    real* wpe;      // (maxT, C)
    real* ln1w;     // (L, C)
    real* ln1b;     // (L, C)
    real* qkvw;     // (L, 3C, C)
    real* qkvb;     // (L, 3C)
    real* attprojw; // (L, C, C)
    real* attprojb; // (L, C)
    real* ln2w;     // (L, C)
    real* ln2b;     // (L, C)
    real* fcw;      // (L, 4C, C)
    real* fcb;      // (L, 4C)
    real* fcprojw;  // (L, C, 4C)
    real* fcprojb;  // (L, C)
    real* lnfw;     // (C)
    real* lnfb;     // (C)
};

inline void fill_param_sizes(size_t* s, const GPTConfig& c) {
    size_t V = c.vocab_size, C = c.n_embd, L = c.n_layer, maxT = c.block_size;
    s[0]  = V * C;            // wte
    s[1]  = maxT * C;         // wpe
    s[2]  = L * C;            // ln1w
    s[3]  = L * C;            // ln1b
    s[4]  = L * (3 * C) * C;  // qkvw
    s[5]  = L * (3 * C);      // qkvb
    s[6]  = L * C * C;        // attprojw
    s[7]  = L * C;            // attprojb
    s[8]  = L * C;            // ln2w
    s[9]  = L * C;            // ln2b
    s[10] = L * (4 * C) * C;  // fcw
    s[11] = L * (4 * C);      // fcb
    s[12] = L * C * (4 * C);  // fcprojw
    s[13] = L * C;            // fcprojb
    s[14] = C;                // lnfw
    s[15] = C;                // lnfb
}

// ===========================================================================
//  Activation tensors (also a flat buffer + offsets).
// ===========================================================================
enum { NUM_ACT_TENSORS = 21 };
struct ActivationTensors {
    real* encoded;   // (B, T, C)          token+pos embedding
    real* ln1;       // (L, B, T, C)       layernorm1 output
    real* ln1_mean;  // (L, B, T)
    real* ln1_rstd;  // (L, B, T)
    real* qkv;       // (L, B, T, 3C)
    real* atty;      // (L, B, T, C)       attention output (pre proj)
    real* preatt;    // (L, B, NH, T, T)
    real* att;       // (L, B, NH, T, T)   softmaxed
    real* attproj;   // (L, B, T, C)
    real* residual2; // (L, B, T, C)       after attn residual
    real* ln2;       // (L, B, T, C)
    real* ln2_mean;  // (L, B, T)
    real* ln2_rstd;  // (L, B, T)
    real* fch;       // (L, B, T, 4C)
    real* fch_gelu;  // (L, B, T, 4C)
    real* fcproj;    // (L, B, T, C)
    real* residual3; // (L, B, T, C)       after mlp residual
    real* lnf;       // (B, T, C)
    real* lnf_mean;  // (B, T)
    real* lnf_rstd;  // (B, T)
    real* logits;    // (B, T, V)
};

inline void fill_act_sizes(size_t* s, const GPTConfig& c, int B, int T) {
    size_t C = c.n_embd, L = c.n_layer, NH = c.n_head, V = c.vocab_size;
    s[0]  = (size_t)B * T * C;             // encoded
    s[1]  = L * B * T * C;                 // ln1
    s[2]  = L * B * T;                     // ln1_mean
    s[3]  = L * B * T;                     // ln1_rstd
    s[4]  = L * B * T * 3 * C;             // qkv
    s[5]  = L * B * T * C;                 // atty
    s[6]  = L * B * NH * T * T;            // preatt
    s[7]  = L * B * NH * T * T;            // att
    s[8]  = L * B * T * C;                 // attproj
    s[9]  = L * B * T * C;                 // residual2
    s[10] = L * B * T * C;                 // ln2
    s[11] = L * B * T;                     // ln2_mean
    s[12] = L * B * T;                     // ln2_rstd
    s[13] = L * B * T * 4 * C;             // fch
    s[14] = L * B * T * 4 * C;             // fch_gelu
    s[15] = L * B * T * C;                 // fcproj
    s[16] = L * B * T * C;                 // residual3
    s[17] = (size_t)B * T * C;             // lnf
    s[18] = (size_t)B * T;                 // lnf_mean
    s[19] = (size_t)B * T;                 // lnf_rstd
    s[20] = (size_t)B * T * V;             // logits
}

// ===========================================================================
//  Individual op forward/backward passes
// ===========================================================================

// token + position embedding
inline void encoder_forward(real* out, const int* inp, const real* wte,
                            const real* wpe, int B, int T, int C) {
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            real* o = out + (b * T + t) * C;
            const real* tok = wte + (size_t)inp[b * T + t] * C;
            const real* pos = wpe + (size_t)t * C;
            for (int i = 0; i < C; i++) o[i] = tok[i] + pos[i];
        }
}
inline void encoder_backward(real* dwte, real* dwpe, const real* dout,
                             const int* inp, int B, int T, int C) {
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) {
            const real* d = dout + (b * T + t) * C;
            real* dtok = dwte + (size_t)inp[b * T + t] * C;
            real* dpos = dwpe + (size_t)t * C;
            for (int i = 0; i < C; i++) { dtok[i] += d[i]; dpos[i] += d[i]; }
        }
}

// layernorm over the last dim (C), with optional weight/bias
inline void layernorm_forward(real* out, real* mean, real* rstd, const real* inp,
                              const real* w, const real* b, int B, int T, int C) {
    const real eps = (real)1e-5;
    for (int bt = 0; bt < B * T; bt++) {
        const real* x = inp + (size_t)bt * C;
        real m = 0; for (int i = 0; i < C; i++) m += x[i]; m /= C;
        real v = 0; for (int i = 0; i < C; i++) { real d = x[i] - m; v += d * d; } v /= C;
        real s = (real)1.0 / std::sqrt(v + eps);
        real* o = out + (size_t)bt * C;
        for (int i = 0; i < C; i++) {
            real nrm = (x[i] - m) * s;
            o[i] = nrm * w[i] + (b ? b[i] : (real)0);
        }
        mean[bt] = m; rstd[bt] = s;
    }
}
inline void layernorm_backward(real* dinp, real* dw, real* db, const real* dout,
                               const real* inp, const real* w, const real* mean,
                               const real* rstd, int B, int T, int C) {
    for (int bt = 0; bt < B * T; bt++) {
        const real* x = inp + (size_t)bt * C;
        const real* d = dout + (size_t)bt * C;
        real* di = dinp + (size_t)bt * C;
        real m = mean[bt], s = rstd[bt];
        // two reduction terms
        real dnorm_mean = 0, dnorm_norm_mean = 0;
        for (int i = 0; i < C; i++) {
            real nrm = (x[i] - m) * s;
            real dnorm = w[i] * d[i];
            dnorm_mean += dnorm;
            dnorm_norm_mean += dnorm * nrm;
        }
        dnorm_mean /= C; dnorm_norm_mean /= C;
        for (int i = 0; i < C; i++) {
            real nrm = (x[i] - m) * s;
            real dnorm = w[i] * d[i];
            di[i] += (dnorm - dnorm_mean - nrm * dnorm_norm_mean) * s;
            dw[i] += nrm * d[i];
            if (db) db[i] += d[i];
        }
    }
}

// matmul: out[b,t,o] = sum_c inp[b,t,c]*w[o,c] + bias[o]
inline void matmul_forward(real* out, const real* inp, const real* w, const real* bias,
                           int B, int T, int C, int OC) {
    int BT = B * T;
    #pragma omp parallel for
    for (int bt = 0; bt < BT; bt++) {
        const real* x = inp + (size_t)bt * C;
        real* o = out + (size_t)bt * OC;
        for (int oc = 0; oc < OC; oc++) {
            const real* wr = w + (size_t)oc * C;
            real acc = bias ? bias[oc] : (real)0;
            for (int c = 0; c < C; c++) acc += x[c] * wr[c];
            o[oc] = acc;
        }
    }
}
inline void matmul_backward(real* dinp, real* dw, real* dbias, const real* dout,
                            const real* inp, const real* w, int B, int T, int C, int OC) {
    int BT = B * T;
    // dinp (each row bt writes its own dinp region -> race-free)
    #pragma omp parallel for
    for (int bt = 0; bt < BT; bt++) {
        const real* d = dout + (size_t)bt * OC;
        real* di = dinp + (size_t)bt * C;
        for (int oc = 0; oc < OC; oc++) {
            const real* wr = w + (size_t)oc * C;
            real dd = d[oc];
            for (int c = 0; c < C; c++) di[c] += wr[c] * dd;
        }
    }
    // dw, dbias
    #pragma omp parallel for
    for (int oc = 0; oc < OC; oc++) {
        real* dwr = dw + (size_t)oc * C;
        real dbacc = 0;
        for (int b = 0; b < B; b++)
            for (int t = 0; t < T; t++) {
                const real* x = inp + (size_t)(b * T + t) * C;
                real dd = dout[(size_t)(b * T + t) * OC + oc];
                dbacc += dd;
                for (int c = 0; c < C; c++) dwr[c] += x[c] * dd;
            }
        if (dbias) dbias[oc] += dbacc;
    }
}

// causal multi-head self-attention.
// inp is qkv (B,T,3C). out is (B,T,C). preatt/att are (B,NH,T,T).
inline void attention_forward(real* out, real* preatt, real* att, const real* inp,
                              int B, int T, int C, int NH) {
    int hs = C / NH;
    real scale = (real)(1.0 / std::sqrt((double)hs));
    int BNHT = B * NH * T;
    #pragma omp parallel for
    for (int idx = 0; idx < BNHT; idx++) {
        int b = idx / (NH * T), h = (idx / T) % NH, t = idx % T;
        {
                const real* q = inp + (size_t)(b * T + t) * 3 * C + h * hs;
                real* pa = preatt + (((size_t)b * NH + h) * T + t) * T;
                real* a  = att    + (((size_t)b * NH + h) * T + t) * T;
                // scores against keys 0..t (causal)
                real maxv = -1e30f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const real* k = inp + (size_t)(b * T + t2) * 3 * C + C + h * hs;
                    real dot = 0; for (int i = 0; i < hs; i++) dot += q[i] * k[i];
                    dot *= scale;
                    pa[t2] = dot;
                    if (dot > maxv) maxv = dot;
                }
                real sum = 0;
                for (int t2 = 0; t2 <= t; t2++) { real e = std::exp(pa[t2] - maxv); a[t2] = e; sum += e; }
                real inv = sum > 0 ? (real)1.0 / sum : (real)0;
                for (int t2 = 0; t2 <= t; t2++) a[t2] *= inv;
                for (int t2 = t + 1; t2 < T; t2++) { a[t2] = 0; pa[t2] = 0; }
                // weighted sum of values
                real* o = out + (size_t)(b * T + t) * C + h * hs;
                for (int i = 0; i < hs; i++) o[i] = 0;
                for (int t2 = 0; t2 <= t; t2++) {
                    const real* v = inp + (size_t)(b * T + t2) * 3 * C + 2 * C + h * hs;
                    real aw = a[t2];
                    for (int i = 0; i < hs; i++) o[i] += aw * v[i];
                }
        }
    }
}
inline void attention_backward(real* dinp, real* datt, const real* dout,
                               const real* inp, const real* att, int B, int T, int C, int NH) {
    int hs = C / NH;
    real scale = (real)(1.0 / std::sqrt((double)hs));
    int BNH = B * NH;
    #pragma omp parallel for
    for (int bh = 0; bh < BNH; bh++) {
        int b = bh / NH, h = bh % NH;
        for (int t = 0; t < T; t++) {
                const real* a  = att  + (((size_t)b * NH + h) * T + t) * T;
                real* da = datt + (((size_t)b * NH + h) * T + t) * T;
                const real* dO = dout + (size_t)(b * T + t) * C + h * hs;
                const real* q = inp + (size_t)(b * T + t) * 3 * C + h * hs;
                real* dq = dinp + (size_t)(b * T + t) * 3 * C + h * hs;
                // backprop value accumulation -> datt and dv
                for (int t2 = 0; t2 <= t; t2++) {
                    const real* v = inp + (size_t)(b * T + t2) * 3 * C + 2 * C + h * hs;
                    real* dv = dinp + (size_t)(b * T + t2) * 3 * C + 2 * C + h * hs;
                    real datt_acc = 0;
                    for (int i = 0; i < hs; i++) { datt_acc += dO[i] * v[i]; dv[i] += a[t2] * dO[i]; }
                    da[t2] = datt_acc;
                }
                // backprop softmax: dpre[t2] = a[t2]*(da[t2] - sum_j a[j]*da[j])
                real dsum = 0; for (int t2 = 0; t2 <= t; t2++) dsum += a[t2] * da[t2];
                for (int t2 = 0; t2 <= t; t2++) {
                    real dpre = a[t2] * (da[t2] - dsum) * scale;
                    const real* k = inp + (size_t)(b * T + t2) * 3 * C + C + h * hs;
                    real* dk = dinp + (size_t)(b * T + t2) * 3 * C + C + h * hs;
                    for (int i = 0; i < hs; i++) { dq[i] += dpre * k[i]; dk[i] += dpre * q[i]; }
                }
        }
    }
}

// GELU. tanh_approx=false: exact erf GELU (nanoGPT's nn.GELU() default).
// tanh_approx=true: gelu_new (what GPT-2 was actually trained with).
inline void gelu_forward(real* out, const real* inp, int N, bool tanh_approx = false) {
    const real inv_sqrt2 = (real)0.7071067811865476;
    const real gc = (real)0.7978845608028654; // sqrt(2/pi)
    for (int i = 0; i < N; i++) {
        real x = inp[i];
        if (tanh_approx) {
            real t = (real)std::tanh(gc * (x + (real)0.044715 * x * x * x));
            out[i] = (real)0.5 * x * ((real)1.0 + t);
        } else {
            out[i] = (real)0.5 * x * ((real)1.0 + (real)std::erf(x * inv_sqrt2));
        }
    }
}
inline void gelu_backward(real* dinp, const real* inp, const real* dout, int N, bool tanh_approx = false) {
    const real inv_sqrt2 = (real)0.7071067811865476;
    const real inv_sqrt2pi = (real)0.3989422804014327;
    const real gc = (real)0.7978845608028654;
    for (int i = 0; i < N; i++) {
        real x = inp[i];
        real dg;
        if (tanh_approx) {
            real inner = gc * (x + (real)0.044715 * x * x * x);
            real t = (real)std::tanh(inner);
            real sech2 = (real)1.0 - t * t;
            dg = (real)0.5 * ((real)1.0 + t) +
                 (real)0.5 * x * sech2 * gc * ((real)1.0 + (real)3 * (real)0.044715 * x * x);
        } else {
            real cdf = (real)0.5 * ((real)1.0 + (real)std::erf(x * inv_sqrt2));
            real pdf = inv_sqrt2pi * (real)std::exp((real)-0.5 * x * x);
            dg = cdf + x * pdf;
        }
        dinp[i] += dg * dout[i];
    }
}

inline void residual_forward(real* out, const real* a, const real* b, int N) {
    for (int i = 0; i < N; i++) out[i] = a[i] + b[i];
}
inline void residual_backward(real* da, real* db, const real* dout, int N) {
    for (int i = 0; i < N; i++) { da[i] += dout[i]; db[i] += dout[i]; }
}

// softmax + cross entropy over vocab. Returns mean loss over valid targets.
// Writes probs into `probs` (B,T,V) and, if dlogits!=nullptr, its gradient.
inline real softmax_crossentropy(real* dlogits, real* probs, const real* logits,
                                 const int* targets, int B, int T, int V) {
    double loss_sum = 0.0; int count = 0;
    for (int bt = 0; bt < B * T; bt++) {
        const real* l = logits + (size_t)bt * V;
        real* p = probs + (size_t)bt * V;
        real maxv = -1e30f; for (int i = 0; i < V; i++) if (l[i] > maxv) maxv = l[i];
        real sum = 0; for (int i = 0; i < V; i++) { real e = std::exp(l[i] - maxv); p[i] = e; sum += e; }
        real inv = (real)1.0 / sum;
        for (int i = 0; i < V; i++) p[i] *= inv;
        int tg = targets[bt];
        if (tg >= 0) { loss_sum += -std::log((double)p[tg] + 1e-30); count++; }
    }
    if (dlogits) {
        real dloss = (real)(1.0 / (count > 0 ? count : 1));
        for (int bt = 0; bt < B * T; bt++) {
            int tg = targets[bt];
            real* dl = dlogits + (size_t)bt * V;
            const real* p = probs + (size_t)bt * V;
            if (tg < 0) { for (int i = 0; i < V; i++) dl[i] = 0; continue; }
            for (int i = 0; i < V; i++) dl[i] = (p[i] - (i == tg ? (real)1 : (real)0)) * dloss;
        }
    }
    return (real)(loss_sum / (count > 0 ? count : 1));
}

// ===========================================================================
//  The GPT model: owns parameters, gradients, activations, AdamW state.
// ===========================================================================
struct GPT {
    GPTConfig config;
    size_t param_sizes[NUM_PARAM_TENSORS];
    size_t num_params = 0;
    std::vector<real> params_mem, grads_mem;
    ParameterTensors params{}, grads{};

    // activations (allocated for a given B,T)
    size_t act_sizes[NUM_ACT_TENSORS];
    size_t num_acts = 0;
    std::vector<real> acts_mem, grads_acts_mem;
    ActivationTensors acts{}, grads_acts{};
    int B = 0, T = 0;

    // AdamW state
    std::vector<real> m_mem, v_mem;
    int adam_t = 0;

    std::vector<int> inputs, targets; // cached batch (for backward)
    real mean_loss = 0;

    void set_param_pointers(ParameterTensors& p, std::vector<real>& mem) {
        real** ptrs[NUM_PARAM_TENSORS] = {
            &p.wte, &p.wpe, &p.ln1w, &p.ln1b, &p.qkvw, &p.qkvb, &p.attprojw,
            &p.attprojb, &p.ln2w, &p.ln2b, &p.fcw, &p.fcb, &p.fcprojw,
            &p.fcprojb, &p.lnfw, &p.lnfb };
        size_t off = 0;
        for (int i = 0; i < NUM_PARAM_TENSORS; i++) { *ptrs[i] = mem.data() + off; off += param_sizes[i]; }
    }
    void set_act_pointers(ActivationTensors& a, std::vector<real>& mem) {
        real** ptrs[NUM_ACT_TENSORS] = {
            &a.encoded, &a.ln1, &a.ln1_mean, &a.ln1_rstd, &a.qkv, &a.atty,
            &a.preatt, &a.att, &a.attproj, &a.residual2, &a.ln2, &a.ln2_mean,
            &a.ln2_rstd, &a.fch, &a.fch_gelu, &a.fcproj, &a.residual3, &a.lnf,
            &a.lnf_mean, &a.lnf_rstd, &a.logits };
        size_t off = 0;
        for (int i = 0; i < NUM_ACT_TENSORS; i++) { *ptrs[i] = mem.data() + off; off += act_sizes[i]; }
    }

    void build(const GPTConfig& c) {
        config = c;
        fill_param_sizes(param_sizes, c);
        num_params = 0; for (int i = 0; i < NUM_PARAM_TENSORS; i++) num_params += param_sizes[i];
        params_mem.assign(num_params, (real)0);
        grads_mem.assign(num_params, (real)0);
        set_param_pointers(params, params_mem);
        set_param_pointers(grads, grads_mem);
    }

    // random init as in nanoGPT: N(0,0.02); residual projections scaled by
    // 1/sqrt(2*n_layer); LayerNorm weights=1, biases=0.
    void init_random(uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> nd(0.0, 0.02);
        // default everything from N(0,0.02) then fix up layernorms/biases
        for (size_t i = 0; i < num_params; i++) params_mem[i] = (real)nd(rng);
        auto setc = [&](real* p, size_t n, real val) { for (size_t i = 0; i < n; i++) p[i] = val; };
        int L = config.n_layer, C = config.n_embd;
        setc(params.wpe, param_sizes[1], 0); // wpe also N(0,0.02) in nanoGPT; reset then fill
        for (size_t i = 0; i < param_sizes[1]; i++) params.wpe[i] = (real)nd(rng);
        // layernorm weights = 1, biases = 0
        setc(params.ln1w, (size_t)L * C, 1); setc(params.ln1b, (size_t)L * C, 0);
        setc(params.ln2w, (size_t)L * C, 1); setc(params.ln2b, (size_t)L * C, 0);
        setc(params.lnfw, C, 1); setc(params.lnfb, C, 0);
        // linear biases = 0
        setc(params.qkvb, param_sizes[5], 0);
        setc(params.attprojb, param_sizes[7], 0);
        setc(params.fcb, param_sizes[11], 0);
        setc(params.fcprojb, param_sizes[13], 0);
        // scaled init for residual projections (c_proj weights)
        std::normal_distribution<double> nd2(0.0, 0.02 / std::sqrt(2.0 * L));
        for (size_t i = 0; i < param_sizes[6]; i++)  params.attprojw[i] = (real)nd2(rng);
        for (size_t i = 0; i < param_sizes[12]; i++) params.fcprojw[i] = (real)nd2(rng);
    }

    void ensure_acts(int B_, int T_) {
        if (B == B_ && T == T_ && !acts_mem.empty()) return;
        B = B_; T = T_;
        fill_act_sizes(act_sizes, config, B, T);
        num_acts = 0; for (int i = 0; i < NUM_ACT_TENSORS; i++) num_acts += act_sizes[i];
        acts_mem.assign(num_acts, (real)0);
        set_act_pointers(acts, acts_mem);
        inputs.assign((size_t)B * T, 0);
        targets.assign((size_t)B * T, -1);
    }

    // Forward pass. If targets != nullptr, also computes mean_loss.
    void forward(const int* inp, const int* tgt, int B_, int T_) {
        ensure_acts(B_, T_);
        int C = config.n_embd, L = config.n_layer, NH = config.n_head, V = config.vocab_size;
        for (int i = 0; i < B * T; i++) inputs[i] = inp[i];

        ActivationTensors& a = acts;
        encoder_forward(a.encoded, inp, params.wte, params.wpe, B, T, C);
        real* resid = a.encoded;
        for (int l = 0; l < L; l++) {
            size_t bt = (size_t)B * T;
            real* ln1 = a.ln1 + l * bt * C;
            layernorm_forward(ln1, a.ln1_mean + l * bt, a.ln1_rstd + l * bt, resid,
                              params.ln1w + l * C, config.bias ? params.ln1b + l * C : nullptr, B, T, C);
            real* qkv = a.qkv + l * bt * 3 * C;
            matmul_forward(qkv, ln1, params.qkvw + (size_t)l * 3 * C * C,
                           config.bias ? params.qkvb + (size_t)l * 3 * C : nullptr, B, T, C, 3 * C);
            real* atty = a.atty + l * bt * C;
            attention_forward(atty, a.preatt + (size_t)l * B * NH * T * T,
                              a.att + (size_t)l * B * NH * T * T, qkv, B, T, C, NH);
            real* attproj = a.attproj + l * bt * C;
            matmul_forward(attproj, atty, params.attprojw + (size_t)l * C * C,
                           config.bias ? params.attprojb + (size_t)l * C : nullptr, B, T, C, C);
            real* resid2 = a.residual2 + l * bt * C;
            residual_forward(resid2, resid, attproj, (int)(bt * C));
            real* ln2 = a.ln2 + l * bt * C;
            layernorm_forward(ln2, a.ln2_mean + l * bt, a.ln2_rstd + l * bt, resid2,
                              params.ln2w + l * C, config.bias ? params.ln2b + l * C : nullptr, B, T, C);
            real* fch = a.fch + l * bt * 4 * C;
            matmul_forward(fch, ln2, params.fcw + (size_t)l * 4 * C * C,
                           config.bias ? params.fcb + (size_t)l * 4 * C : nullptr, B, T, C, 4 * C);
            real* fch_gelu = a.fch_gelu + l * bt * 4 * C;
            gelu_forward(fch_gelu, fch, (int)(bt * 4 * C), config.gelu_tanh);
            real* fcproj = a.fcproj + l * bt * C;
            matmul_forward(fcproj, fch_gelu, params.fcprojw + (size_t)l * C * 4 * C,
                           config.bias ? params.fcprojb + (size_t)l * C : nullptr, B, T, 4 * C, C);
            real* resid3 = a.residual3 + l * bt * C;
            residual_forward(resid3, resid2, fcproj, (int)(bt * C));
            resid = resid3;
        }
        layernorm_forward(a.lnf, a.lnf_mean, a.lnf_rstd, resid, params.lnfw,
                          config.bias ? params.lnfb : nullptr, B, T, C);
        // lm_head (tied to wte): logits = lnf @ wte^T
        matmul_forward(a.logits, a.lnf, params.wte, nullptr, B, T, C, V);

        if (tgt) {
            for (int i = 0; i < B * T; i++) targets[i] = tgt[i];
            // use grads_acts.logits region as scratch probs if allocated; else a temp
            static thread_local std::vector<real> probs;
            probs.assign((size_t)B * T * V, 0);
            mean_loss = softmax_crossentropy(nullptr, probs.data(), a.logits, targets.data(), B, T, V);
        } else {
            mean_loss = -1;
        }
    }

    void zero_grad() {
        std::fill(grads_mem.begin(), grads_mem.end(), (real)0);
        if (!grads_acts_mem.empty()) std::fill(grads_acts_mem.begin(), grads_acts_mem.end(), (real)0);
    }

    void backward() {
        if (grads_acts_mem.empty()) { grads_acts_mem.assign(num_acts, (real)0); set_act_pointers(grads_acts, grads_acts_mem); }
        int C = config.n_embd, L = config.n_layer, NH = config.n_head, V = config.vocab_size;
        size_t bt = (size_t)B * T;
        ActivationTensors& a = acts; ActivationTensors& g = grads_acts;

        // cross-entropy -> dlogits (reuse probs)
        static thread_local std::vector<real> probs;
        probs.assign((size_t)B * T * V, 0);
        softmax_crossentropy(g.logits, probs.data(), a.logits, targets.data(), B, T, V);

        // lm_head backward: logits = lnf @ wte^T  (weight = wte, tied)
        matmul_backward(g.lnf, grads.wte, nullptr, g.logits, a.lnf, params.wte, B, T, C, V);
        // final layernorm
        // dresidual (of last block's residual3) accumulates into g.residual3 of last layer
        real* dresid_final = (L > 0) ? g.residual3 + (size_t)(L - 1) * bt * C : g.encoded;
        real* resid_final = (L > 0) ? a.residual3 + (size_t)(L - 1) * bt * C : a.encoded;
        layernorm_backward(dresid_final, grads.lnfw, config.bias ? grads.lnfb : nullptr,
                           g.lnf, resid_final, params.lnfw, a.lnf_mean, a.lnf_rstd, B, T, C);

        for (int l = L - 1; l >= 0; l--) {
            real* resid  = (l == 0) ? a.encoded : a.residual3 + (size_t)(l - 1) * bt * C;
            real* dresid = (l == 0) ? g.encoded : g.residual3 + (size_t)(l - 1) * bt * C;
            real* dresid3 = g.residual3 + (size_t)l * bt * C;
            real* dresid2 = g.residual2 + (size_t)l * bt * C;
            real* dfcproj = g.fcproj + (size_t)l * bt * C;
            // residual3 = residual2 + fcproj
            residual_backward(dresid2, dfcproj, dresid3, (int)(bt * C));
            // mlp c_proj
            real* dfch_gelu = g.fch_gelu + (size_t)l * bt * 4 * C;
            matmul_backward(dfch_gelu, grads.fcprojw + (size_t)l * C * 4 * C,
                            config.bias ? grads.fcprojb + (size_t)l * C : nullptr, dfcproj,
                            a.fch_gelu + (size_t)l * bt * 4 * C, params.fcprojw + (size_t)l * C * 4 * C,
                            B, T, 4 * C, C);
            // gelu
            real* dfch = g.fch + (size_t)l * bt * 4 * C;
            gelu_backward(dfch, a.fch + (size_t)l * bt * 4 * C, dfch_gelu, (int)(bt * 4 * C), config.gelu_tanh);
            // mlp c_fc
            real* dln2 = g.ln2 + (size_t)l * bt * C;
            matmul_backward(dln2, grads.fcw + (size_t)l * 4 * C * C,
                            config.bias ? grads.fcb + (size_t)l * 4 * C : nullptr, dfch,
                            a.ln2 + (size_t)l * bt * C, params.fcw + (size_t)l * 4 * C * C,
                            B, T, C, 4 * C);
            // ln2
            layernorm_backward(dresid2, grads.ln2w + (size_t)l * C, config.bias ? grads.ln2b + (size_t)l * C : nullptr,
                               dln2, a.residual2 + (size_t)l * bt * C, params.ln2w + (size_t)l * C,
                               a.ln2_mean + l * bt, a.ln2_rstd + l * bt, B, T, C);
            // residual2 = resid + attproj
            real* dattproj = g.attproj + (size_t)l * bt * C;
            residual_backward(dresid, dattproj, dresid2, (int)(bt * C));
            // attn c_proj
            real* datty = g.atty + (size_t)l * bt * C;
            matmul_backward(datty, grads.attprojw + (size_t)l * C * C,
                            config.bias ? grads.attprojb + (size_t)l * C : nullptr, dattproj,
                            a.atty + (size_t)l * bt * C, params.attprojw + (size_t)l * C * C, B, T, C, C);
            // attention
            real* dqkv = g.qkv + (size_t)l * bt * 3 * C;
            real* datt = g.att + (size_t)l * B * NH * T * T;
            attention_backward(dqkv, datt, datty, a.qkv + (size_t)l * bt * 3 * C,
                               a.att + (size_t)l * B * NH * T * T, B, T, C, NH);
            // qkv matmul
            real* dln1 = g.ln1 + (size_t)l * bt * C;
            matmul_backward(dln1, grads.qkvw + (size_t)l * 3 * C * C,
                            config.bias ? grads.qkvb + (size_t)l * 3 * C : nullptr, dqkv,
                            a.ln1 + (size_t)l * bt * C, params.qkvw + (size_t)l * 3 * C * C, B, T, C, 3 * C);
            // ln1
            layernorm_backward(dresid, grads.ln1w + (size_t)l * C, config.bias ? grads.ln1b + (size_t)l * C : nullptr,
                               dln1, resid, params.ln1w + (size_t)l * C,
                               a.ln1_mean + l * bt, a.ln1_rstd + l * bt, B, T, C);
        }
        // encoder (wte gets contribution here too, in addition to lm_head tie)
        encoder_backward(grads.wte, grads.wpe, g.encoded, inputs.data(), B, T, C);
    }

    // AdamW update. decay applied only to 2D tensors (matmul weights + embeddings).
    void adamw(real lr, real beta1, real beta2, real eps, real weight_decay) {
        if (m_mem.empty()) { m_mem.assign(num_params, (real)0); v_mem.assign(num_params, (real)0); }
        adam_t++;
        real bc1 = (real)(1.0 - std::pow((double)beta1, adam_t));
        real bc2 = (real)(1.0 - std::pow((double)beta2, adam_t));
        // which flat ranges are 2D weights (decayed): wte, wpe? nanoGPT decays dim>=2.
        // wte(2D) decay, wpe(2D) decay, layernorms(1D) no, linear weights(2D) decay,
        // linear biases(1D) no. Build a per-parameter decay mask cheaply via sizes.
        // Tensors order: 0 wte(2D),1 wpe(2D),2 ln1w(1D),3 ln1b,4 qkvw(2D),5 qkvb,
        // 6 attprojw(2D),7 attprojb,8 ln2w,9 ln2b,10 fcw(2D),11 fcb,12 fcprojw(2D),
        // 13 fcprojb,14 lnfw,15 lnfb.
        const bool decay2d[NUM_PARAM_TENSORS] = {
            true, true, false, false, true, false, true, false,
            false, false, true, false, true, false, false, false };
        size_t off = 0;
        for (int ti = 0; ti < NUM_PARAM_TENSORS; ti++) {
            bool dec = decay2d[ti];
            for (size_t j = 0; j < param_sizes[ti]; j++) {
                size_t i = off + j;
                real gr = grads_mem[i];
                real m = beta1 * m_mem[i] + (1 - beta1) * gr;
                real v = beta2 * v_mem[i] + (1 - beta2) * gr * gr;
                m_mem[i] = m; v_mem[i] = v;
                real mh = m / bc1, vh = v / bc2;
                real upd = mh / ((real)std::sqrt((double)vh) + eps);
                if (dec) upd += weight_decay * params_mem[i];
                params_mem[i] -= lr * upd;
            }
            off += param_sizes[ti];
        }
    }
};

} // namespace gpt

#endif // NANOGPT_GPT_H
