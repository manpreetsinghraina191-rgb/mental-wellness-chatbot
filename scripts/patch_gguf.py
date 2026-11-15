"""Make the fine-tuned model loadable by Ollama by patching its weights into Ollama's own base GGUF.

Why: the Colab export used a newer llama.cpp converter that FUSES some Qwen3.5 tensors
(attn_qkv + attn_gate -> attn_qkv_gate, ssm_beta + ssm_alpha -> ssm_beta_alpha). Ollama 0.40.16
(latest stable on 2026-11-15) expects them separate and refuses to load the file.

Those fused tensors belong to the linear-attention layers, which LoRA never touched (the adapter
targets q/k/v/o and the MLP gate/up/down projections only). So: copy Ollama's base qwen3.5:4b
GGUF and overwrite every tensor that exists in BOTH files with the same name and shape by the
fine-tuned one. Same quantisation type -> raw bytes copied. Different type (4 attn_v tensors:
Q4_K in the export, Q6_K in the base) -> dequantised and re-encoded as Q6_K (encoder below,
checked by round-trip through the official gguf decoder).

Checks printed before writing:
  - an untouched tensor (blk.0 attn_qkv) matches the base up to quantisation noise
  - fine-tuned tensors (MLP) differ from the base by clearly more than that

  BASE=$(ollama show qwen3.5:4b --modelfile | grep ^FROM | cut -d' ' -f2-)
  uv run --with gguf python scripts/patch_gguf.py models/Qwen3.5-4B.Q4_K_M.gguf "$BASE" OUT.gguf
  uv run python scripts/make_ollama_model.py OUT.gguf
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader
from gguf.quants import dequantize

QK = 256  # values per Q6_K super-block


def quantize_q6_k(x: np.ndarray) -> np.ndarray:
    """Encode float rows as ggml Q6_K blocks: ql[128] qh[64] scales[16](int8) d(f16) = 210 bytes.
    Value = d * scale[sub-block of 16] * (q - 32), q in 0..63 (same layout as ggml's
    quantize_row_q6_K_ref; simpler max-abs scale search)."""
    blocks = x.reshape(-1, QK).astype(np.float32)
    n = blocks.shape[0]
    sub = blocks.reshape(n, 16, 16)
    amax = np.abs(sub).max(axis=2)  # (n, 16)
    sub_scale = amax / 31.0
    d = sub_scale.max(axis=1) / 127.0  # (n,)
    d16 = d.astype(np.float16)
    dd = d16.astype(np.float32)
    safe_d = np.where(dd == 0, 1, dd)
    scales = np.clip(np.round(sub_scale / safe_d[:, None]), -128, 127).astype(np.int8)
    step = dd[:, None] * scales.astype(np.float32)  # (n, 16)
    safe_step = np.where(step == 0, 1, step)
    q = np.clip(np.round(sub / safe_step[:, :, None]), -32, 31)
    q = np.where(step[:, :, None] == 0, 0, q)
    L = (q + 32).astype(np.uint8).reshape(n, QK)

    ql = np.zeros((n, 128), np.uint8)
    qh = np.zeros((n, 64), np.uint8)
    for half in range(2):  # two halves of 128 values
        base = half * 128
        q1 = L[:, base : base + 32]
        q2 = L[:, base + 32 : base + 64]
        q3 = L[:, base + 64 : base + 96]
        q4 = L[:, base + 96 : base + 128]
        ql[:, half * 64 : half * 64 + 32] = (q1 & 0xF) | ((q3 & 0xF) << 4)
        ql[:, half * 64 + 32 : half * 64 + 64] = (q2 & 0xF) | ((q4 & 0xF) << 4)
        qh[:, half * 32 : half * 32 + 32] = (
            (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)
        )
    out = np.concatenate(
        [ql, qh, scales.view(np.uint8), d16.reshape(n, 1).view(np.uint8).reshape(n, 2)], axis=1
    )
    return out.reshape(x.shape[0], -1)


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1))


def deq(t: object) -> np.ndarray:
    return np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)  # type: ignore[attr-defined]


def main() -> None:
    ft_path, base_path, out_path = (Path(a) for a in sys.argv[1:4])
    ft = {t.name: t for t in GGUFReader(ft_path).tensors}
    base = {t.name: t for t in GGUFReader(base_path).tensors}

    # 1. the Q6_K encoder round-trips through the official decoder
    sample = deq(ft["blk.3.attn_v.weight"])
    enc = quantize_q6_k(sample)
    err = rel_diff(np.asarray(dequantize(enc, Q.Q6_K)), sample)
    print(f"Q6_K encoder round-trip relative error: {err:.4f}")
    assert err < 0.02, "Q6_K encoder is wrong"

    # 2. untouched tensors match the base; fine-tuned ones differ by more
    qkv = deq(ft["blk.0.attn_qkv_gate.weight"])[: base["blk.0.attn_qkv.weight"].shape[1]]
    untouched = rel_diff(qkv, deq(base["blk.0.attn_qkv.weight"]))
    touched = [
        rel_diff(deq(ft[n]), deq(base[n])) for n in ("blk.0.ffn_up.weight", "blk.3.attn_q.weight")
    ]
    print(f"untouched blk.0 attn_qkv vs base: {untouched:.4f}  (quantisation noise)")
    print(f"fine-tuned blk.0 ffn_up / blk.3 attn_q vs base: {touched[0]:.4f} / {touched[1]:.4f}")

    shared = [n for n in ft if n in base and list(ft[n].shape) == list(base[n].shape)]
    shutil.copyfile(base_path, out_path)
    copied = reencoded = 0
    with open(out_path, "r+b") as f:
        for n in shared:
            src, dst = ft[n], base[n]
            if src.tensor_type == dst.tensor_type:
                data = np.asarray(src.data).tobytes()
                copied += 1
            elif dst.tensor_type == Q.Q6_K:
                data = quantize_q6_k(deq(src)).tobytes()
                reencoded += 1
            else:
                sys.exit(f"{n}: no encoder for {dst.tensor_type.name}")
            assert len(data) == dst.n_bytes, f"{n}: size mismatch"
            f.seek(dst.data_offset)
            f.write(data)
    print(f"patched {copied} tensors byte-for-byte, re-encoded {reencoded} -> {out_path}")


if __name__ == "__main__":
    main()
