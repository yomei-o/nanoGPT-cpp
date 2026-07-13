"""
Export a pretrained GPT-2 checkpoint to the flat binary format read by gpt2.cpp.

Uses nanoGPT's own model.py (in the parent directory) to fetch and correctly
transpose the OpenAI weights into nn.Linear layout, then writes them in the
exact parameter order expected by gpt.h's ParameterTensors.

Requirements (only to run this exporter once): torch + transformers
    pip install torch transformers

Usage:
    python export_gpt2.py [gpt2|gpt2-medium|gpt2-large|gpt2-xl] [out.bin]

It also downloads the BPE files encoder.json and vocab.bpe next to out.bin.
"""
import os
import sys
import struct
import urllib.request

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import GPT  # nanoGPT's model.py

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "model_gpt2.bin"
GELU_TANH = 1  # GPT-2 was trained with gelu_new (tanh approx)

print(f"loading {MODEL} via nanoGPT.from_pretrained ...")
model = GPT.from_pretrained(MODEL)
model.eval()
sd = model.state_dict()
cfg = model.config
L = cfg.n_layer

def cat_layers(suffix):
    return np.concatenate([sd[f"transformer.h.{l}.{suffix}"].numpy().astype(np.float32).ravel()
                           for l in range(L)])

# assemble in the exact order of gpt.h ParameterTensors
parts = [
    sd["transformer.wte.weight"].numpy().astype(np.float32).ravel(),   # wte
    sd["transformer.wpe.weight"].numpy().astype(np.float32).ravel(),   # wpe
    cat_layers("ln_1.weight"),                                          # ln1w
    cat_layers("ln_1.bias"),                                            # ln1b
    cat_layers("attn.c_attn.weight"),                                   # qkvw
    cat_layers("attn.c_attn.bias"),                                     # qkvb
    cat_layers("attn.c_proj.weight"),                                   # attprojw
    cat_layers("attn.c_proj.bias"),                                     # attprojb
    cat_layers("ln_2.weight"),                                          # ln2w
    cat_layers("ln_2.bias"),                                            # ln2b
    cat_layers("mlp.c_fc.weight"),                                      # fcw
    cat_layers("mlp.c_fc.bias"),                                        # fcb
    cat_layers("mlp.c_proj.weight"),                                    # fcprojw
    cat_layers("mlp.c_proj.bias"),                                      # fcprojb
    sd["transformer.ln_f.weight"].numpy().astype(np.float32).ravel(),  # lnfw
    sd["transformer.ln_f.bias"].numpy().astype(np.float32).ravel(),    # lnfb
]
flat = np.concatenate(parts)
print(f"total parameters written: {flat.size:,}")

with open(OUT, "wb") as f:
    f.write(b"GPT2")
    f.write(struct.pack("<i", 1))  # version
    f.write(struct.pack("<7i", cfg.block_size, cfg.vocab_size, cfg.n_layer,
                        cfg.n_head, cfg.n_embd, 1 if cfg.bias else 0, GELU_TANH))
    f.write(struct.pack("<q", flat.size))
    f.write(flat.tobytes())
print(f"wrote {OUT}")

# also fetch the BPE files if not present
base = "https://openaipublic.blob.core.windows.net/gpt-2/models/124M"
outdir = os.path.dirname(os.path.abspath(OUT))
for name in ["encoder.json", "vocab.bpe"]:
    dst = os.path.join(outdir, name)
    if not os.path.exists(dst):
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(f"{base}/{name}", dst)
print("done. Run:  gpt2 model_gpt2.bin encoder.json vocab.bpe --prompt \"...\"")
