[English](README.md) | [日本語](README.ja.md)

# nanoGPT — C++ port

A dependency-free **standard C++17** port of Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT). It implements the full GPT-2
transformer — **forward AND backward** — by hand, plus an AdamW optimiser, so it
can both **train** a character-level GPT from scratch and **run inference with
real pretrained GPT-2 weights**.

Ports [`model.py`](../model.py) (the network) and the training/sampling logic of
[`train.py`](../train.py) / [`sample.py`](../sample.py) to C++.

## Do we need a transformer C++ library? No.

A transformer is "just matmuls plus a few elementwise ops": token/position
embeddings, LayerNorm, causal multi-head self-attention (QKV matmuls + softmax),
an MLP with GELU, residual connections, and a linear head. All of that is
implemented here from scratch. No PyTorch, no Eigen, no BLAS, no ggml/llama.cpp
— **only the C++ standard library** (this mirrors Karpathy's own llm.c /
llama2.c). OpenMP is used automatically if present, but is optional.

- Builds and runs unchanged with **MSVC (Visual Studio 2022), GCC and Clang**,
  on Windows, Linux and macOS.
- Backprop verified against numerical gradients; the BPE tokenizer verified
  against real GPT-2 token ids; char-level training verified to converge.

## What's implemented

The exact GPT-2 architecture from `model.py`, forward and backward:

- token + position embeddings (with weight tying: `lm_head` shares `wte`)
- LayerNorm (with optional bias)
- causal multi-head self-attention
- MLP with GELU (exact `erf`, or the `tanh` approximation GPT-2 was trained with)
- residual connections
- cross-entropy loss
- AdamW with decoupled weight decay (decay only on 2D tensors, as in nanoGPT)

## Files

| File | Purpose |
|---|---|
| `gpt.h` | The GPT model: all ops forward+backward, AdamW (the core) |
| `tokenizer.h` | Character-level tokenizer (for training) |
| `bpe.h` | GPT-2 byte-level BPE tokenizer (for GPT-2 inference) |
| `main.cpp` | `train` / `sample` driver for the char-level model |
| `gpt2.cpp` | Loads real GPT-2 weights and generates text |
| `export_gpt2_safetensors.py` | Exports GPT-2 weights to the C++ format — **numpy only, no torch** (recommended) |
| `export_gpt2.py` | Same, via nanoGPT's `model.py` (needs torch + transformers) |
| `test_grad.cpp` | Gradient check (double precision) — backprop correctness |
| `bpe_test.cpp` | Verifies BPE output against known GPT-2 token ids |
| `CMakeLists.txt` | Cross-platform build (auto-detects OpenMP) |

## Build

### Visual Studio 2022 (CMake)

Open Visual Studio 2022 → **Open a Local Folder** → select this `cpp` folder.
VS auto-detects `CMakeLists.txt`; choose **`x64-Release`** → **Build → Build All**.
This produces `nanogpt`, `gpt2`, `nanogpt_gradcheck` and `bpe_test`.

### CMake (any compiler)

```bash
cmake -S . -B build
cmake --build build --config Release
```

### g++ / clang++ directly

```bash
g++ -O3 -ffast-math -fopenmp -std=c++17 -o nanogpt main.cpp
g++ -O3 -ffast-math -fopenmp -std=c++17 -o gpt2   gpt2.cpp
g++ -O3 -ffast-math -DGPT_USE_DOUBLE -std=c++17 -o nanogpt_gradcheck test_grad.cpp
```

(OpenMP is optional — drop `-fopenmp` / `/openmp` and it still builds and runs.)

---

## A. Train a character-level GPT (fully self-contained)

Get the tiny-shakespeare dataset (a ~1MB text file), then train:

```bash
# input.txt (already included; or re-download):
#   curl -sSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

nanogpt train input.txt --steps 5000 --layers 4 --embd 128 --heads 4 --block 64 --out ckpt.bin
```

It prints the train/val loss as it goes and writes `ckpt.bin`. Then sample:

```bash
nanogpt sample ckpt.bin --tokens 500 --temp 0.8 --topk 40 --prompt "ROMEO:"
```

Options — `train`: `--steps --lr --batch --block --layers --embd --heads --out
--eval-every --seed --init --ckpt --warmup --min-lr --decay-iters --no-lr-decay
--grad-clip`; `sample`: `--tokens --temp --topk --prompt --seed`.

By default the learning rate follows a **cosine decay with linear warmup** (the
same `get_lr` schedule as nanoGPT's `train.py`): it warms up over `--warmup`
steps (default `steps/10`), then decays from `--lr` down to `--min-lr` (default
`lr/10`) by `--decay-iters` (default: the final step). Pass `--no-lr-decay` for a
constant learning rate. Gradients are clipped to a global L2 norm of `--grad-clip`
(default `1.0`; set `0` to disable). On `resume` the schedule continues at the
restored global step, so warmup isn't repeated.

### Resume or fine-tune a char model

`train` can continue from a saved checkpoint instead of starting from scratch,
via `--init` and `--ckpt`:

```bash
# resume: pick up an interrupted run. Restores weights, AdamW momentum AND the
# step counter, so training continues exactly where it left off.
nanogpt train input.txt --init resume --ckpt ckpt.bin --steps 2000 --out ckpt.bin

# finetune: keep the trained weights, but start a fresh optimiser + step counter,
# and (typically) train on a different text file with a smaller learning rate.
nanogpt train other.txt --init finetune --ckpt ckpt.bin --steps 500 --lr 3e-4 --out ft.bin
```

- `--init scratch` (default) — random init, as before.
- `--init resume` — load params **and** AdamW state (`m`, `v`, step) and continue.
- `--init finetune` — load params only; the optimiser and step counter reset to 0.

The checkpoint format is now **v2** (params + AdamW moments + step count); old
**v1** checkpoints (params only) still load. Fine-tuning reuses the checkpoint's
own character vocabulary — any character in the new text that wasn't in the
original vocab is skipped on encode, so fine-tune on text from the same domain
(e.g. more English) for best results.

A short run already shows clear learning (measured here, 3 layers / 96 embd,
CPU only):

```
step     0 | train loss 4.2012 | val loss 4.2000
step   300 | train loss 2.3784 | val loss 2.3962
```

Loss keeps dropping with more steps / a bigger model (nanoGPT reaches ~1.5).

---

## B. Run real GPT-2 (124M … 1.5B) inference

Export the pretrained weights once. The recommended exporter needs **only numpy
+ requests** (no torch): it downloads GPT-2's `model.safetensors` from
HuggingFace, transposes the Conv1D weights into nn.Linear layout, and writes the
C++ format.

```bash
pip install numpy requests
python export_gpt2_safetensors.py gpt2 model_gpt2.bin   # ~548MB download -> 498MB model_gpt2.bin
```

(Alternatively `export_gpt2.py gpt2 model_gpt2.bin` uses nanoGPT's own
`model.py`, which needs `torch` + `transformers`.)

You also need the BPE files `encoder.json` and `vocab.bpe` (already included; or
`export_gpt2.py` downloads them). Then generate — pure C++, no Python:

```bash
gpt2 model_gpt2.bin encoder.json vocab.bpe --prompt "The meaning of life is" --tokens 100 --temp 0.7 --topk 40
```

Example output (GPT-2 124M, measured here):

```
The meaning of life is not the same as the meaning of life. It is not something
a mere human being, a human being, needs. A human being needs to live in the
way we want to live, to live ...
```

`gpt2-medium`, `gpt2-large`, `gpt2-xl` work too (pass the name to the exporter).
On CPU this is slow for the larger models (no KV cache, naive matmul); the point
is portability, not speed. GPT-2 124M generates ~2 tokens/s on 4 CPU threads.

### Fine-tune GPT-2 on your own text

`gpt2 finetune` continues training the real GPT-2 weights on a plain-text file
(BPE-tokenised on the fly) and writes a new `.bin` that `gpt2` can generate from.
It shares the same forward+backward, AdamW, cosine LR schedule and gradient
clipping as the char trainer.

```bash
# fine-tune 124M on some text, then generate from the result
gpt2 finetune model_gpt2.bin mytext.txt encoder.json vocab.bpe \
    --steps 200 --block 256 --batch 1 --lr 3e-5 --out model_ft.bin
gpt2 model_ft.bin encoder.json vocab.bpe --prompt "Once upon a time"
```

Options: `--steps --lr --batch --block --out --eval-every --warmup --min-lr
--decay-iters --no-lr-decay --grad-clip --init finetune|resume --seed`. The saved
`.bin` also carries the AdamW state, so `--init resume` picks fine-tuning back up
where it stopped (a plain `gpt2` generate ignores that trailing state).

Speed (this machine, GPT-2 124M, 4 CPU threads, `batch 1`): about **6 s/step at
`--block 128`** and **~14 s/step at `--block 256`**. So a ~200-step fine-tune is
roughly **20 min** (`block 128`) to **45 min** (`block 256`); ~1000 steps is a
few hours. It is CPU-only and un-optimised (no KV cache, naive matmul, no
gradient accumulation) — meant as a working demo, not a fast trainer. Larger
models (`gpt2-medium`/`large`/`xl`) are proportionally slower.

---

## C. Load a model trained by Python nanoGPT (weight-compatible)

You can load a checkpoint produced by nanoGPT's own `train.py` (e.g. the
shakespeare_char model) and run it in the C++ port. nanoGPT stores `nn.Linear`
weights as `[out, in]` (no Conv1D transpose needed), and this port uses the same
exact-erf GELU as nanoGPT's `nn.GELU()`, so the C++ logits match PyTorch to
floating-point rounding.

```bash
# after training in Python nanoGPT (writes out/ckpt.pt and data/.../meta.pkl):
python export_nanogpt.py out/ckpt.pt data/shakespeare_char/meta.pkl model.bin
nanogpt sample model.bin --prompt "ROMEO:" --tokens 500 --temp 0.8 --topk 40
```

**Verified numerically** with `verify_nanogpt_compat.py`, which builds a nanoGPT
model via the repo's `model.py`, exports it, and compares logits on the same
input:

```
PyTorch vs C++ logits (16 positions x 65 vocab):
  max abs diff  = 2.384e-07
  argmax agreement = 100.0%
  -> PASS (weight-compatible)
```

So greedy generation is identical, and sampled generation is functionally
identical (only the RNG differs). Note: bit-exactness with PyTorch is not
attempted (different kernels/summation order); ~1e-7 agreement is the practical
limit. For the tightest match, build without `-ffast-math`.

---

## Verification (all run on this machine)

| Check | How | Result |
|---|---|---|
| Backprop correctness | `nanogpt_gradcheck` (analytic vs numerical, double) | **PASS**, max rel err 7.5e-6 across every parameter tensor |
| BPE tokenizer | `bpe_test` vs known GPT-2 token ids | **PASS** (exact ids + perfect round-trip) |
| End-to-end training | `nanogpt train` on tiny-shakespeare | loss 4.20 → 2.38, learns English text |
| Real GPT-2 124M inference | `gpt2` on exported OpenAI weights | **works** — generates coherent English (see example above) |
| Weight-compat vs Python nanoGPT | `verify_nanogpt_compat.py` (logits vs PyTorch) | **PASS**, max abs diff 2.4e-7, argmax 100% |

Coherent GPT-2 output is strong evidence the 124M weights load in the correct
layout and the forward pass is right (any layout error produces garbage).

## Notes / caveats

- Compute type is `float` by default; `-DGPT_USE_DOUBLE` switches to double
  (used by the gradient-check target for a crisp check — finite-difference
  gradient checking is unreliable in float).
- Numerical gradient checking is only meaningful in the double build; that is
  why it is a separate target (`nanogpt_gradcheck`) rather than a mode of
  `nanogpt`.
- BPE fidelity: GPT-2's pre-tokenization uses a Unicode-property regex
  (`\p{L}`, `\p{N}`) that is impractical to reproduce in dependency-free C++.
  The splitter here is ASCII-faithful — it matches GPT-2 exactly for ASCII
  English (verified) and only diverges on non-ASCII letters. The byte-level BPE
  itself is exact.
- GELU: `gpt.h` defaults to exact `erf` GELU (nanoGPT's `nn.GELU()`); the
  GPT-2 export sets the `tanh` approximation (`gelu_new`, what GPT-2 was trained
  with).
- No KV cache: generation recomputes the whole context each step. Fine for a
  demo; not optimised for throughput.

`input.txt`, `encoder.json` and `vocab.bpe` are included for convenience; you
may prefer to `.gitignore` them and let users fetch them via the commands above.
