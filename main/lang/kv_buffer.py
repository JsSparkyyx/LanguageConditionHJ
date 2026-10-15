"""KvBuffer (V6, qwen_impl.md 4): layer 0's keys / values of the image prefix, per captured tick.

``<root>/kv_buffer/`` is a RingSidecar (sidecar.py) in lockstep with ``buffer/`` and ``item_buffer/``. Row i holds
what HoloBrain-Qwen's layer 0 computed over the 325-token prefix of frame i (the three camera images inside the
prompt, up to the "." of "You are a robot."): keys after the 3D rotary embedding (transformers' cache layout) and
values, [n_kv_heads, prefix_len, head_dim] each. The words of any sentence are then one layer-0 pass of the
sentence's tokens against these rows (HoloBrainQwenEncoder.encode_text), at training time, with the relabeled
sentence.

numpy has no bfloat16 here, so ``k`` / ``v`` hold the raw bf16 bit patterns as int16; ``rows_to_kv`` reinterprets
them with ``torch.Tensor.view(torch.bfloat16)`` (exact), never a numeric cast. The header records the prefix
layout and the checkpoint the rows came from: a reader that passes ``ckpt_id`` refuses rows of another checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .sidecar import RingSidecar

PREFIX_LEN = 325   # positions through "You are a robot." with 3 x 99 image pads (qwen_impl.md 2)
ROPE_NEXT = 61     # the 3D rotary position of the first sentence token
N_KV_HEADS = 2
HEAD_DIM = 128
KV_VERSION = 1


class KvBuffer(RingSidecar):
    KIND = "kv_version"
    VERSION = KV_VERSION
    DIRNAME = "kv_buffer"

    @classmethod
    def make_fields(cls, prefix_len: int = PREFIX_LEN, n_kv_heads: int = N_KV_HEADS, head_dim: int = HEAD_DIM,
                    **_) -> dict:
        shape = (int(n_kv_heads), int(prefix_len), int(head_dim))
        return {"k": (shape, np.dtype(np.int16)), "v": (shape, np.dtype(np.int16))}

    @classmethod
    def make_meta(cls, prefix_len: int = PREFIX_LEN, n_kv_heads: int = N_KV_HEADS, head_dim: int = HEAD_DIM,
                  rope_next: int = ROPE_NEXT, ckpt_id: str | None = None, attn: str | None = None,
                  encoder_version: int | None = None) -> dict:
        meta = {"prefix_len": int(prefix_len), "rope_next": int(rope_next), "n_kv_heads": int(n_kv_heads),
                "head_dim": int(head_dim), "value_dtype": "bfloat16 bits in int16", "key_rope": "applied"}
        if ckpt_id is not None:
            meta["ckpt_id"] = str(ckpt_id)
        if attn is not None:
            meta["attn"] = str(attn)
        if encoder_version is not None:   # HoloBrainQwenEncoder.ENCODER_VERSION; absent = 1 (rows of 2026-10-14)
            meta["encoder_version"] = int(encoder_version)
        return meta

    @classmethod
    def open_read(cls, root: str | Path, ckpt_id: str | None = None) -> "KvBuffer":
        """Read-only open; with ckpt_id, refuse rows written by another HoloBrain-Qwen checkpoint."""
        kb = cls(root)
        if ckpt_id is not None and kb.header.get("ckpt_id") not in (None, ckpt_id):
            raise ValueError(f"{kb.root}: kv rows of checkpoint {kb.header.get('ckpt_id')}, this run uses {ckpt_id}")
        return kb


def kv_to_row(k, v) -> dict:
    """Torch bf16 keys / values [1, n_kv, P, d] or [n_kv, P, d] -> {"k", "v"} int16 numpy copies (the bits)."""
    import torch

    out = {}
    for name, x in (("k", k), ("v", v)):
        if x.dtype != torch.bfloat16:
            raise TypeError(f"kv_to_row: {name} is {x.dtype}, expected bfloat16")
        if x.dim() == 4:
            if x.shape[0] != 1:
                raise ValueError(f"kv_to_row: {name} batch {x.shape[0]}, one frame per row")
            x = x[0]
        if not torch.isfinite(x).all():
            raise ValueError(f"kv_to_row: non-finite {name}")
        out[name] = x.detach().contiguous().view(torch.int16).cpu().numpy().copy()
    return out


def rows_to_kv(k, v, device="cuda"):
    """int16 bit arrays [U, n_kv, P, d] (numpy or torch) -> (K, V) torch bfloat16 on the device."""
    import torch

    out = []
    for x in (k, v):
        t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x))
        if t.dtype != torch.int16:
            raise TypeError(f"rows_to_kv: {t.dtype}, expected int16 bit patterns")
        out.append(t.to(device, non_blocking=True).view(torch.bfloat16))
    return out[0], out[1]
