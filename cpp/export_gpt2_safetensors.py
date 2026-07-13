"""
torch-free exporter: reads GPT-2's model.safetensors (HuggingFace) with numpy
only and writes the flat binary format read by gpt2.cpp.

Needs only numpy (+ requests to download). No torch, no transformers.

Usage:
    python export_gpt2_safetensors.py [gpt2|gpt2-medium|gpt2-large|gpt2-xl] [out.bin]
"""
import os, sys, json, struct
import numpy as np

NAME = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "model_gpt2.bin"
GELU_TANH = 1  # GPT-2 was trained with gelu_new (tanh approx)

HF_REPO = {"gpt2": "gpt2", "gpt2-medium": "gpt2-medium",
           "gpt2-large": "gpt2-large", "gpt2-xl": "gpt2-xl"}[NAME]
HEAD_MAP = {768: 12, 1024: 16, 1280: 20, 1600: 25}

here = os.path.dirname(os.path.abspath(OUT))
st_path = os.path.join(here, f"{NAME}.safetensors")
if not os.path.exists(st_path):
    import requests
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/model.safetensors"
    print(f"downloading {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)); done = 0
        with open(st_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); done += len(chunk)
                if total: print(f"\r  {done/1e6:.0f} / {total/1e6:.0f} MB", end="")
    print()

# ---- parse safetensors (8-byte header len, JSON header, then raw bytes) ----
def load_safetensors(path):
    dtmap = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    n = int(np.frombuffer(mm[:8].tobytes(), dtype="<u8")[0])
    header = json.loads(mm[8:8 + n].tobytes())
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        arr = np.frombuffer(mm[base + a: base + b].tobytes(), dtype=dtmap[meta["dtype"]])
        out[name] = arr.reshape(meta["shape"])
    return out

T = load_safetensors(st_path)
C = T["wte.weight"].shape[1]
V = T["wte.weight"].shape[0]
block = T["wpe.weight"].shape[0]
L = 1 + max(int(k.split(".")[1]) for k in T if k.startswith("h."))
H = HEAD_MAP[C]
print(f"{NAME}: L={L} C={C} H={H} vocab={V} block={block}")

def cat(suffix, transpose=False):
    out = []
    for l in range(L):
        a = np.asarray(T[f"h.{l}.{suffix}"], dtype=np.float32)
        if transpose:  # HF Conv1D weight is [in, out]; nn.Linear wants [out, in]
            a = a.T
        out.append(np.ascontiguousarray(a).ravel())
    return np.concatenate(out)

parts = [
    np.asarray(T["wte.weight"], np.float32).ravel(),   # wte
    np.asarray(T["wpe.weight"], np.float32).ravel(),   # wpe
    cat("ln_1.weight"), cat("ln_1.bias"),                          # ln1w, ln1b
    cat("attn.c_attn.weight", True), cat("attn.c_attn.bias"),      # qkvw, qkvb
    cat("attn.c_proj.weight", True), cat("attn.c_proj.bias"),      # attprojw, attprojb
    cat("ln_2.weight"), cat("ln_2.bias"),                          # ln2w, ln2b
    cat("mlp.c_fc.weight", True), cat("mlp.c_fc.bias"),            # fcw, fcb
    cat("mlp.c_proj.weight", True), cat("mlp.c_proj.bias"),        # fcprojw, fcprojb
    np.asarray(T["ln_f.weight"], np.float32).ravel(),  # lnfw
    np.asarray(T["ln_f.bias"], np.float32).ravel(),    # lnfb
]
flat = np.concatenate(parts).astype(np.float32)
print(f"total parameters: {flat.size:,}")

with open(OUT, "wb") as f:
    f.write(b"GPT2")
    f.write(struct.pack("<i", 1))
    f.write(struct.pack("<7i", block, V, L, H, C, 1, GELU_TANH))
    f.write(struct.pack("<q", flat.size))
    f.write(flat.tobytes())
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB)")
print("run:  ./gpt2 model_gpt2.bin encoder.json vocab.bpe --prompt \"...\"")
