"""HoloBrain's frozen text-image fusion stage (FUSION_PLAN V5; the contract is fusion_impl.md 4.1).

HoloBrain-0 GD runs, per frame, backbone -> neck (4 maps, strides 8 / 16 / 32 / 64) -> ``feature_enhancer`` (6
layers of text-image fusion with the sentence) -> ``spatial_enhancer`` (depth, 3D positions). ``HoloBrainEncoder``
skips the enhancer; this module puts it back, frozen, on stored neck maps:

- ``FusionStage(ckpt_dir)``: the checkpoint's ``feature_enhancer`` (277 keys, strict load) and ``spatial_enhancer``
  (the HoloBrainEncoder's when given), eval mode, no grad. ``fuse(neck, words, wmask)`` fuses frame f with sentence
  f and returns the 1200 scene tokens, their 3D positions (the spatial enhancer predicts depth from the fused maps,
  so the positions depend on the sentence too) and the fused words. ``unfused(neck)`` is the V0 path (the spatial
  enhancer on the raw maps): today's stored tokens.
- The enhancer's own forward runs unchanged, one sentence length per call. HoloBrain does not mask padded words
  (``BiMultiHeadAttention`` adds a bool mask, and ``bool.masked_fill(-9e15)`` stays all True) and tiles the text
  mask head-major, so a padded or mixed-length batch differs from HoloBrain's single-sentence inference. ``fuse``
  therefore groups frames by the sentence length n, slices the words to n and builds ``position_ids`` / ``masks``
  from n alone (``text_dict_for_length``): within a group every mask is the same and nothing is padded.
- ``msda_lowmem``: the deformable attention summed level by level. transformers 4.57 has no compiled kernel, and
  its pure-PyTorch fallback stacks the 4 levels (~250 MB per frame at peak); exact up to summation order.
- ``encode_obs_with_neck(encoder, obs)``: one backbone pass that returns today's tokens bit for bit plus the frame's
  neck dict; ``neck_to_row`` / ``rows_to_neck`` convert it to and from a ``neck_buffer/`` row.

Neck dict (fusion_impl.md section 3), torch tensors with a leading frame dim F:
``neck_maps`` [F, 3, 1700, 256] (the 4 levels per camera, level-major, row-major within a level),
``neck_3d`` [F, 3, 400, 128] (neck_3d levels 1 and 2), ``projection_mat`` [F, 3, 4, 4] (ego frame),
``image_wh`` [F, 3, 2], ``T_ego2world`` [F, 4, 4] (the positions' out_extrinsic).
"""

from __future__ import annotations

import contextlib
import json
import os
from functools import partial

import numpy as np
import torch
from torch import nn

# the neck layout and the row dtypes are defined once, in the torch-free neck_buffer (check_shards reads it on the
# login node): LEVEL_SHAPES ((h, w) per neck level at 320 x 256), TOKEN_LEVELS (1, 2: 320 + 80 tokens per camera)
from .neck_buffer import LEVEL_SHAPES, NECK_FIELDS, ROW_FIELDS, TOKEN_LEVELS

__all__ = [
    "LEVEL_SHAPES", "N_CELLS", "TOKEN_LEVELS", "N_TOK_CAM", "FUSION_DTYPES",
    "text_dict_for_length", "msda_lowmem", "tf32_flags", "token_positions", "bi_attention_softmax",
    "FusionStage", "BiAttentionCapture", "encode_obs_with_neck", "neck_to_row", "rows_to_neck", "synthetic_neck",
]

N_CELLS = sum(h * w for h, w in LEVEL_SHAPES)          # 1700 per camera
LEVEL_START = tuple(int(s) for s in np.cumsum([0] + [h * w for h, w in LEVEL_SHAPES[:-1]]))  # 0, 1280, 1600, 1680
N_TOK_CAM = sum(LEVEL_SHAPES[i][0] * LEVEL_SHAPES[i][1] for i in TOKEN_LEVELS)             # 400
FUSION_DTYPES = ("fp32", "tf32", "bf16")
N_ENHANCER_KEYS = 277  # feature_enhancer.* in pretrain/model.safetensors (recon, 2026-10-09)
# frames per spatial-enhancer call: its per-depth-bin point features are [3, 400, 128, 128] fp32 = 79 MB per frame
# (twice that at peak), so a fusion chunk of 32 would hold 5 GB there alone. Per-token ops: no effect on values.
SPATIAL_CHUNK = 8

# neck_buffer row dtypes (fusion_impl.md section 3: fp16 maps, fp32 geometry); episode_id / t_in_ep are added by
# the lockstep writer
_ROW_DTYPES = {k: NECK_FIELDS[k][1].type for k in ROW_FIELDS}


# ---------------------------------------------------------------------------
# text masks, deformable attention, precision
# ---------------------------------------------------------------------------


def text_dict_for_length(n: int, L: int, device=None) -> dict:
    """HoloBrain's text masks for ONE sentence of n tokens ([CLS] w1..wk [SEP], n = k + 2), truncated or padded to L.

    ``bip3d.bert.generate_masks_with_special_tokens_and_transfer_map`` (special tokens [CLS], [SEP]) gives for such a
    sentence: ``position_ids`` [CLS] = 0, w1 = 0, ..., [SEP] = n - 2, padding 0; ``masks`` the identity plus one
    block over positions 1..n-1 ([CLS] attends to itself only); ``text_token_mask`` True on the n real tokens. All
    three follow from n, so the text cache needs no new fields. Returns position_ids [L] long, masks [L, L] bool,
    text_token_mask [L] bool (``fuse`` uses L = n).
    """
    ar = torch.arange(L, device=device)
    body = (ar >= 1) & (ar < n)
    return {
        "position_ids": torch.where(body, ar - 1, torch.zeros_like(ar)),
        "masks": (body[:, None] & body[None, :]) | torch.eye(L, dtype=torch.bool, device=device),
        "text_token_mask": ar < n,
    }


def msda_lowmem(value, spatial_shapes, spatial_shapes_list, level_start_index, sampling_locations,
                attention_weights, im2col_step):
    """transformers' pure-PyTorch ``MultiScaleDeformableAttention.forward`` (same signature and result), summing the
    levels one at a time instead of stacking them.

    The original keeps the 4 per-level samples [B * heads, d, Q, P] alive, stacks them (a copy) and multiplies by the
    weights (a third copy): ~250 MB per frame at 3 x 1700 queries. Here one level's samples are alive at a time and
    are weighted in place (out of place when autograd records). Exact up to the summation order of the L * P
    weighted samples per head. ``spatial_shapes``, ``level_start_index`` and ``im2col_step`` are unused, as in the
    original.
    """
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in spatial_shapes_list], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    # (B, Q, heads, L, P) -> (B * heads, 1, Q, L * P), as the original
    weights = attention_weights.transpose(1, 2).reshape(batch_size * num_heads, 1, num_queries, num_levels * num_points)
    in_place = not torch.is_grad_enabled()
    out = None
    for lvl, (h, w) in enumerate(spatial_shapes_list):
        value_l = value_list[lvl].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, h, w)
        grid_l = sampling_grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)   # (B * heads, Q, P, 2)
        if value_l.dtype != grid_l.dtype:  # bf16 values, fp32 grid: autocast promotes grid_sample; do it explicitly
            dt = torch.promote_types(value_l.dtype, grid_l.dtype)
            value_l, grid_l = value_l.to(dt), grid_l.to(dt)
        s = nn.functional.grid_sample(value_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False)
        w_l = weights[..., lvl * num_points:(lvl + 1) * num_points]           # (B * heads, 1, Q, P)
        if in_place and s.dtype == torch.promote_types(s.dtype, w_l.dtype):
            s = s.mul_(w_l)
        else:
            s = s * w_l
        part = s.sum(-1)                                                      # (B * heads, d, Q)
        del s
        out = part if out is None else (out.add_(part) if in_place and out.dtype == part.dtype else out + part)
    return out.view(batch_size, num_heads * hidden_dim, num_queries).transpose(1, 2).contiguous()


class _LowMemMSDA(nn.Module):
    """``msda_lowmem`` as a module: ``self_attn.func`` is a registered child module (no parameters), which
    nn.Module only lets another module replace."""

    def forward(self, *args):
        return msda_lowmem(*args)


@contextlib.contextmanager
def tf32_flags(on: bool):
    """TF32 for CUDA matmuls AND cuDNN inside the block only, both restored on exit (the trainer calls the stage from
    inside its own ``matmul_tf32`` block; collection and evaluation keep PyTorch's defaults)."""
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    torch.backends.cudnn.allow_tf32 = bool(on)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


# ---------------------------------------------------------------------------
# geometry and layout helpers (the math of HoloBrainEncoder, per frame)
# ---------------------------------------------------------------------------


def _maps_to_cells(maps):
    """list of [B, n_cam, C, h, w] -> [B, n_cam, sum(h * w), C], level-major, row-major within a level (the
    flattening of HoloBrainEncoder.encode_visual_tokens)."""
    return torch.cat([f.flatten(3).transpose(2, 3) for f in maps], dim=2)


def _cells_to_maps(cells, shapes):
    """[B, n_cam, sum(h * w), C] cells -> list of contiguous [B, n_cam, C, h, w] (inverse of _maps_to_cells)."""
    out, s = [], 0
    for h, w in shapes:
        out.append(cells[:, :, s:s + h * w].transpose(2, 3).unflatten(3, (h, w)).contiguous())
        s += h * w
    return out


def _take(x, idx, device):
    """Frames idx (list of ints) of a neck field -> fp32 on device (indexed on the field's own device first)."""
    return x[torch.as_tensor(idx, dtype=torch.long, device=x.device)].to(device, torch.float32)


def _as_4x4(T):
    """[..., 4, 4] or [..., 3, 4] (as spatial_token_positions accepts) -> [..., 4, 4]."""
    T = torch.as_tensor(T)
    if T.shape[-2:] == (3, 4):
        last = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=T.dtype, device=T.device).expand(*T.shape[:-2], 1, 4)
        T = torch.cat([T, last], dim=-2)
    return T


def token_positions(spatial_enhancer, depth_prob, image_wh, projection_mat, fused, T_ego2world):
    """HoloBrainEncoder.spatial_token_positions (pos_mean over all levels of ``fused``) with one out_extrinsic per
    frame, since the frames of one call can come from different episodes.

    depth_prob [g, n_cam, N, D], image_wh / projection_mat as the spatial enhancer got them, T_ego2world [g, 4, 4]
    -> [g, n_cam, N, 3] world metres.
    """
    spatial_shapes = torch.tensor([[f.shape[-2], f.shape[-1]] for f in fused], device=depth_prob.device)
    pts = spatial_enhancer.get_pts(
        spatial_shapes, image_wh.to(depth_prob.device), projection_mat.to(depth_prob),
        depth_prob.device, depth_prob.dtype,
    )  # [g, n_cam, N, num_depth, 3]
    pos = (depth_prob.unsqueeze(-1) * pts).sum(dim=-2)
    T = T_ego2world.to(depth_prob)
    return torch.stack([pos[f] @ T[f, :3, :3].T + T[f, :3, 3] for f in range(pos.shape[0])])


def bi_attention_softmax(attn, vision, lang, attention_mask_v=None, attention_mask_l=None):
    """The two softmaxes ``bip3d.layers.BiMultiHeadAttention.forward`` computes and does not return, recomputed with
    its exact operations (scale, the +-50000 clamps, the per-word max shift, the bool text mask that adds +1 to every
    logit): (word [bsz, heads, tgt, src], each vision row's softmax over the words; cell [bsz, heads, src, tgt], each
    word's softmax over the vision rows)."""
    bsz, tgt_len, _ = vision.size()
    query_states = attn.v_proj(vision) * attn.scale
    key_states = attn._shape(attn.l_proj(lang), -1, bsz)
    proj_shape = (bsz * attn.num_heads, -1, attn.head_dim)
    query_states = attn._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    src_len = key_states.size(1)
    w = torch.bmm(query_states, key_states.transpose(1, 2))
    if attn.stable_softmax_2d:
        w = w - w.max()
    if attn.clamp_min_for_underflow:
        w = torch.clamp(w, min=-attn.MAX_CLAMP_VALUE)
    if attn.clamp_max_for_overflow:
        w = torch.clamp(w, max=attn.MAX_CLAMP_VALUE)
    w_t = w.transpose(1, 2)
    w_l = w_t - torch.max(w_t, dim=-1, keepdim=True)[0]
    if attn.clamp_min_for_underflow:
        w_l = torch.clamp(w_l, min=-attn.MAX_CLAMP_VALUE)
    if attn.clamp_max_for_overflow:
        w_l = torch.clamp(w_l, max=attn.MAX_CLAMP_VALUE)
    if attention_mask_v is not None:
        mask_v = attention_mask_v[:, None, None, :].repeat(1, attn.num_heads, 1, 1).flatten(0, 1)
        w_l = w_l.masked_fill(mask_v, float("-inf"))
    cell = w_l.softmax(dim=-1)
    if attention_mask_l is not None:
        mask = attention_mask_l.unsqueeze(1).unsqueeze(1).expand(bsz, 1, tgt_len, src_len)
        mask = mask.masked_fill(mask == 0, -9e15)
        w = (w.view(bsz, attn.num_heads, tgt_len, src_len) + mask).view(bsz * attn.num_heads, tgt_len, src_len)
    word = nn.functional.softmax(w, dim=-1)
    return (word.view(bsz, attn.num_heads, tgt_len, src_len), cell.view(bsz, attn.num_heads, src_len, tgt_len))


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


class BiAttentionCapture:
    """The bi-attention softmaxes of chosen fusion layers (E4), recorded by ``FusionStage.capture_bi_attention``.

    One entry per enhancer call (``fuse`` makes one per sentence length and chunk): ``calls[i]`` = {"fuse_call",
    "frames" (indices into that fuse call's F), "n"}; ``word[layer][i]`` [g, heads, n_cam * 20, n], each stride-64
    cell's softmax over the n words (what the cell reads); ``cell[layer][i]`` [g, heads, n, n_cam * 20], each word's
    softmax over the stride-64 cells of all cameras jointly (camera-major, row-major 4 x 5 per camera). CPU fp32.
    Only ``fuse`` calls should run inside the block (a direct enhancer call records softmaxes without a ``calls``
    entry).
    """

    def __init__(self, layers):
        self.layers = tuple(int(i) for i in layers)
        self.calls: list[dict] = []
        self.word: dict[int, list] = {i: [] for i in self.layers}
        self.cell: dict[int, list] = {i: [] for i in self.layers}

    def _begin(self, fuse_call: int, frames, n: int) -> None:
        self.calls.append({"fuse_call": int(fuse_call), "frames": list(frames), "n": int(n)})

    def _hook(self, layer, module, args, kwargs):
        vision = args[0] if len(args) > 0 else kwargs["vision"]
        lang = args[1] if len(args) > 1 else kwargs["lang"]
        word, cell = bi_attention_softmax(module, vision, lang, kwargs.get("attention_mask_v"),
                                          kwargs.get("attention_mask_l"))
        self.word[layer].append(word.float().cpu())
        self.cell[layer].append(cell.float().cpu())

    def per_frame(self, layer: int, fuse_call: int | None = None):
        """One fuse call's softmaxes (default: the last recorded) in its frame order: lists over frames of word
        [heads, n_cam * 20, n_f] and cell [heads, n_f, n_cam * 20]."""
        if not self.calls:
            return [], []
        fc = self.calls[-1]["fuse_call"] if fuse_call is None else int(fuse_call)
        sel = [i for i, c in enumerate(self.calls) if c["fuse_call"] == fc]
        n_frames = sum(len(self.calls[i]["frames"]) for i in sel)
        word, cell = [None] * n_frames, [None] * n_frames
        for i in sel:
            for j, f in enumerate(self.calls[i]["frames"]):
                word[f], cell[f] = self.word[layer][i][j], self.cell[layer][i][j]
        return word, cell


class FusionStage:
    """HoloBrain's frozen feature_enhancer + spatial_enhancer, from the pretrain/ checkpoint.

    Never a submodule of LangPolicyEncoder (deep-copied for the target, optimised, saved): a plain object holding
    eval-mode, requires_grad-free modules, called under no_grad. ``dtype`` (LangConfig.fusion_dtype) and ``chunk``
    (fusion_chunk) are read at call time.
    """

    def __init__(self, ckpt_dir: str, device="cuda", dtype: str = "fp32", chunk: int = 32,
                 spatial_enhancer=None, lowmem: bool = True):
        from safetensors import safe_open

        from main.network.build import build  # the builder HoloBrainEncoder uses

        if dtype not in FUSION_DTYPES:
            raise ValueError(f"fusion dtype {dtype!r}: expected one of {FUSION_DTYPES}")
        if int(chunk) < 1:
            raise ValueError(f"fusion chunk {chunk}: expected >= 1 frames")
        ckpt_dir = str(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, "pretrain")):  # the root or pretrain/ itself, as HoloBrainEncoder
            ckpt_dir = os.path.join(ckpt_dir, "pretrain")
        with open(os.path.join(ckpt_dir, "model.config.json")) as f:
            cfg = json.load(f)
        if "BIP3D" not in cfg["class_type"]:
            raise ValueError(f"FusionStage targets the GD/BIP3D variant (HoloBrain_v0.0_GD); got {cfg['class_type']}")
        self.ckpt_dir, self.cfg = ckpt_dir, cfg
        self.device = torch.device(device)
        self.dtype, self.chunk, self.lowmem = dtype, int(chunk), bool(lowmem)

        # weights: the same model.safetensors as HoloBrainEncoder's neck / spatial enhancer and the text cache's
        # text_feat_map (never mix checkpoints, fusion_impl.md R2)
        enhancer = build(cfg["feature_enhancer"])
        with safe_open(os.path.join(ckpt_dir, "model.safetensors"), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            sd = {k[len("feature_enhancer."):]: f.get_tensor(k) for k in keys if k.startswith("feature_enhancer.")}
            sd_se = None if spatial_enhancer is not None else {
                k[len("spatial_enhancer."):]: f.get_tensor(k) for k in keys if k.startswith("spatial_enhancer.")
            }
        enhancer.load_state_dict(sd, strict=True)  # raises on any missing / unexpected key
        self.n_enhancer_keys = len(sd)
        if spatial_enhancer is None:
            spatial_enhancer = build(cfg["spatial_enhancer"])
            spatial_enhancer.load_state_dict(sd_se, strict=True)
            spatial_enhancer.to(self.device)
        else:  # the rollout's HoloBrainEncoder.spatial_enhancer: the same weights, already on its device
            p = next(spatial_enhancer.parameters())
            if p.device.type != self.device.type:
                raise ValueError(f"spatial_enhancer is on {p.device}, the stage on {self.device}")
        self.enhancer = enhancer.to(self.device).eval().requires_grad_(False)
        self.spatial_enhancer = spatial_enhancer.eval().requires_grad_(False)

        # the deformable attention: transformers' pure-PyTorch fallback lives in self_attn.func (bip3d/layers.py
        # 29-39, 287-296); with a compiled kernel there is no func and nothing to patch
        self.n_lowmem = 0
        if self.lowmem:
            for blk in self.enhancer.img_attn_blocks:
                if hasattr(blk.self_attn, "func"):
                    blk.self_attn.func = _LowMemMSDA()
                    self.n_lowmem += 1
        self._captures: list[BiAttentionCapture] = []
        self._n_fuse = 0

    # -- precision ---------------------------------------------------------

    def precision(self):
        """The stage's matmul precision (``self.dtype``) for the whole stage: TF32 for CUDA matmuls and cuDNN on for
        "tf32", off for "fp32" and "bf16" (whose spatial enhancer and positions run in fp32), restored on exit."""
        return tf32_flags(self.dtype == "tf32")

    def _enhancer_autocast(self):
        """bf16: autocast around the enhancer only. On CPU every dtype runs fp32."""
        if self.dtype == "bf16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # -- checks ------------------------------------------------------------

    def _check(self, neck: dict, F: int) -> int:
        """Neck field shapes for F frames -> n_cam."""
        m = neck["neck_maps"]
        if m.dim() != 4 or m.shape[0] != F or m.shape[2] != N_CELLS:
            raise ValueError(f"neck_maps {tuple(m.shape)}: expected [F={F}, n_cam, {N_CELLS}, C]")
        n_cam = int(m.shape[1])
        want = {"neck_3d": (F, n_cam, N_TOK_CAM), "projection_mat": (F, n_cam, 4, 4),
                "image_wh": (F, n_cam, 2), "T_ego2world": (F, 4, 4)}
        for k, shape in want.items():
            if tuple(neck[k].shape[:len(shape)]) != shape:
                raise ValueError(f"{k} {tuple(neck[k].shape)}: expected {shape}{'' if k != 'neck_3d' else ' + [C]'}")
        return n_cam

    @staticmethod
    def _lengths(wmask) -> list[int]:
        """Sentence length n per frame; the mask must be [CLS] w1..wk [SEP] then padding (TextStore's)."""
        wm = wmask.bool()
        n = wm.sum(1)
        ar = torch.arange(wm.shape[1], device=wm.device)
        if bool((n < 1).any()) or not torch.equal(wm, ar[None] < n[:, None]):
            raise ValueError("wmask must be a non-empty prefix mask per frame ([CLS] w1..wk [SEP], then padding)")
        return [int(x) for x in n.tolist()]

    # -- forward -----------------------------------------------------------

    def _spatial(self, maps, neck: dict, idx: list):
        """The spatial enhancer on levels 1, 2 of frames idx (``maps``: their two [g, n_cam, 256, h, w], raw or
        fused) with the stored neck_3d -> tokens [g, n_cam * 400, 256], positions [g, n_cam * 400, 3], flattened
        camera-major as rollout.py stores scene_tokens. Inputs as HoloBrainEncoder.encode_spatial passes them."""
        toks, poss = [], []
        for s in range(0, len(idx), SPATIAL_CHUNK):
            sub = idx[s:s + SPATIAL_CHUNK]
            m = [x[s:s + SPATIAL_CHUNK] for x in maps]
            f3d = _cells_to_maps(_take(neck["neck_3d"], sub, self.device), [LEVEL_SHAPES[i] for i in TOKEN_LEVELS])
            wh = _take(neck["image_wh"], sub, self.device)
            proj = _take(neck["projection_mat"], sub, self.device)
            fused, depth_prob, _ = self.spatial_enhancer(
                feature_maps=m, inputs={"image_wh": wh, "projection_mat": proj}, feature_3d=f3d,
            )
            toks.append(_maps_to_cells(fused).flatten(1, 2))
            pos = token_positions(self.spatial_enhancer, depth_prob, wh, proj, fused,
                                  _take(neck["T_ego2world"], sub, self.device))
            poss.append(pos.flatten(1, 2))
        return torch.cat(toks), torch.cat(poss)

    @torch.no_grad()
    def unfused(self, neck: dict):
        """V0 tokens: the spatial enhancer on the raw levels 1, 2 -> tokens [F, 1200, 256] fp32, pos [F, 1200, 3]
        fp32 (rollout.py's stored scene_tokens / scene_pos for a live fp32 neck)."""
        F = int(neck["neck_maps"].shape[0])
        n_cam = self._check(neck, F)
        if F == 0:
            return (torch.zeros(0, n_cam * N_TOK_CAM, 256, device=self.device),
                    torch.zeros(0, n_cam * N_TOK_CAM, 3, device=self.device))
        cells = neck["neck_maps"][:, :, LEVEL_START[TOKEN_LEVELS[0]]:LEVEL_START[TOKEN_LEVELS[-1] + 1]]
        toks, poss = [], []
        with self.precision():
            for s in range(0, F, self.chunk):
                idx = list(range(s, min(F, s + self.chunk)))
                maps = _cells_to_maps(_take(cells, idx, self.device), [LEVEL_SHAPES[i] for i in TOKEN_LEVELS])
                t, p = self._spatial(maps, neck, idx)
                toks.append(t)
                poss.append(p)
        return torch.cat(toks), torch.cat(poss)

    @torch.no_grad()
    def fuse(self, neck: dict, words, wmask):
        """Frame f fused with sentence f.

        words [F, L, 256] (TextStore.lookup), wmask [F, L] bool -> tokens [F, 1200, 256] fp32, pos [F, 1200, 3] fp32,
        w_fused [F, L, 256] fp32 (zeros past each frame's n). Frames are grouped by n = wmask.sum(1); each group runs
        the official enhancer forward on unpadded words (words[:, :n], text_dict_for_length(n, n)) in chunks of
        ``self.chunk`` frames ([g, n_cam, 256, h, w] maps, the cameras read the words jointly), then the spatial
        enhancer on the fused levels 1, 2. The output keeps the input frame order.
        """
        F, L, C = (int(x) for x in words.shape)
        if tuple(wmask.shape) != (F, L):
            raise ValueError(f"wmask {tuple(wmask.shape)} for words {tuple(words.shape)}")
        n_cam = self._check(neck, F)
        n = self._lengths(wmask)
        tok = torch.zeros(F, n_cam * N_TOK_CAM, 256, device=self.device)
        pos = torch.zeros(F, n_cam * N_TOK_CAM, 3, device=self.device)
        w_fused = torch.zeros(F, L, C, device=self.device)
        self._n_fuse += 1
        with self.precision():
            for length in sorted(set(n)):
                td = text_dict_for_length(length, length, self.device)
                frames = [f for f in range(F) if n[f] == length]
                for s in range(0, len(frames), self.chunk):
                    idx = frames[s:s + self.chunk]
                    g = len(idx)
                    text_dict = {
                        "embedded": _take(words, idx, self.device)[:, :length].contiguous(),
                        "position_ids": td["position_ids"].expand(g, -1),
                        "masks": td["masks"].expand(g, -1, -1),
                        "text_token_mask": td["text_token_mask"].expand(g, -1),
                    }
                    maps = _cells_to_maps(_take(neck["neck_maps"], idx, self.device), LEVEL_SHAPES)
                    for cap in self._captures:
                        cap._begin(self._n_fuse, idx, length)
                    with self._enhancer_autocast():
                        fused, w_f = self.enhancer(feature_maps=maps, text_dict=text_dict)
                    del maps
                    t, p = self._spatial([fused[i].float() for i in TOKEN_LEVELS], neck, idx)
                    del fused
                    out = torch.as_tensor(idx, dtype=torch.long, device=self.device)
                    tok[out], pos[out] = t, p
                    w_fused[out, :length] = w_f.float()
        return tok, pos, w_fused

    @contextlib.contextmanager
    def capture_bi_attention(self, layers=(0, 5)):
        """E4: record, for the given enhancer layers, the bi-attention softmaxes of every ``fuse`` inside the block
        (a forward pre-hook on text_img_attn_blocks[i].attn recomputes them from its inputs, bi_attention_softmax).
        Yields a BiAttentionCapture."""
        n_layers = len(self.enhancer.text_img_attn_blocks)
        for i in layers:
            if not 0 <= int(i) < n_layers:
                raise ValueError(f"fusion layer {i}: the enhancer has layers 0..{n_layers - 1}")
        cap = BiAttentionCapture(layers)
        blocks = self.enhancer.text_img_attn_blocks
        handles = [blocks[i].attn.register_forward_pre_hook(partial(cap._hook, i), with_kwargs=True)
                   for i in cap.layers]
        self._captures.append(cap)
        try:
            yield cap
        finally:
            for h in handles:
                h.remove()
            self._captures.remove(cap)


# ---------------------------------------------------------------------------
# observation -> tokens + neck, and the neck_buffer row format
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_obs_with_neck(encoder, obs: dict):
    """ONE backbone / neck pass -> ((tokens [B, 3, 400, 256], pos_mean [B, 3, 400, 3]), neck dict with F = B).

    tokens / pos_mean are bit-identical to ``encoder.encode_visual_tokens(imgs, depths, image_wh, projection_mat,
    out_extrinsic=T_ego2world)`` (the base rollout's _encode_obs): the same encoder methods on the same inputs, in the
    same order (levels 1, 2 dropped before the spatial enhancer). The neck dict holds the section-3 fields on the
    encoder's device, fp32: all 4 neck maps, neck_3d levels 1, 2, the camera geometry cast as every current use
    casts it (projection_mat and T_ego2world fp64 -> fp32; image_wh int -> fp32, which leaves get_pts' pixel grid
    unchanged).
    """
    feature_maps, feature_3d = encoder.extract_vision_features(obs["imgs"], obs["depths"])
    shapes = tuple(tuple(int(x) for x in f.shape[-2:]) for f in feature_maps)
    if shapes != LEVEL_SHAPES:
        raise ValueError(f"neck levels {shapes}: the neck format expects {LEVEL_SHAPES} (320 x 256 images)")
    maps = [feature_maps[i] for i in TOKEN_LEVELS]
    f3d = [feature_3d[i] for i in TOKEN_LEVELS]
    fused, depth_prob = encoder.encode_spatial(maps, obs["image_wh"], obs["projection_mat"], f3d)
    tokens = torch.cat([f.flatten(3).transpose(2, 3) for f in fused], dim=2)   # [B, n_cam, 400, 256]
    pos_mean, _ = encoder.spatial_token_positions(
        depth_prob, obs["image_wh"], obs["projection_mat"], fused, out_extrinsic=obs["T_ego2world"],
    )
    B, n_cam = int(tokens.shape[0]), int(tokens.shape[1])
    dev = tokens.device
    wh = torch.as_tensor(obs["image_wh"]).to(dev, torch.float32).reshape(-1, 2)
    if wh.shape[0] == 1:  # one (w, h) for every camera, as HoloBrain's processor gives it
        wh = wh.expand(n_cam, 2)
    T = _as_4x4(obs["T_ego2world"]).to(dev, torch.float32).reshape(-1, 4, 4)
    neck = {
        "neck_maps": _maps_to_cells(feature_maps),                                  # [B, n_cam, 1700, 256]
        "neck_3d": _maps_to_cells(f3d),                                             # [B, n_cam, 400, 128]
        "projection_mat": torch.as_tensor(obs["projection_mat"]).to(dev, torch.float32).reshape(B, n_cam, 4, 4),
        "image_wh": wh.reshape(-1, n_cam, 2).expand(B, n_cam, 2).contiguous(),
        "T_ego2world": T.expand(B, 4, 4).contiguous(),
    }
    return (tokens, pos_mean), neck


def neck_to_row(neck: dict) -> dict:
    """Neck dict with F = 1 -> numpy row for neck_buffer/ (F squeezed): neck_maps / neck_3d fp16 (cast on the
    device, round to nearest like the StepBuffer's scene tokens), the camera geometry fp32. A neck value beyond the
    fp16 range raises instead of storing inf."""
    row = {}
    for k, dt in _ROW_DTYPES.items():
        t = neck[k]
        if t.shape[0] != 1:
            raise ValueError(f"neck_to_row: {k} has {t.shape[0]} frames, expected 1")
        t = t[0].detach()
        if dt == np.float16:
            t = t.half()
            if not bool(torch.isfinite(t).all()):
                raise ValueError(f"neck_to_row: {k} overflows fp16 (max |x| {float(neck[k].abs().max()):.4g})")
        row[k] = np.array(t.cpu().numpy(), dtype=dt)  # own copy: the row outlives this tick's neck dict
    return row


def rows_to_neck(arrays: dict, device) -> dict:
    """numpy [F, ...] (NeckBuffer.read) -> torch dict on device: each field uploaded in its own dtype (fp16 maps are
    never widened on the CPU), then float fields cast to fp32 on the device; integer fields (episode_id, t_in_ep)
    keep theirs."""
    out = {}
    for k, a in arrays.items():
        a = np.asarray(a)
        if not a.flags.writeable:
            a = a.copy()
        t = torch.from_numpy(np.ascontiguousarray(a)).to(device)
        out[k] = t.float() if t.is_floating_point() else t
    return out


def synthetic_neck(n_frames: int, seed: int = 0, device="cpu", n_cam: int = 3) -> dict:
    """A random neck dict with valid camera geometry, for tests and ``bench_fusion.py --synthetic``: maps and neck_3d
    ~ N(0, 1); pinhole cameras at 320 x 256 (the last one is the ego / head camera, the others at small random poses
    relative to it); a random ego -> world pose per frame."""
    g = torch.Generator().manual_seed(int(seed))

    def rigid(rot: float, trans: float):
        aa = (torch.rand(3, generator=g) * 2 - 1) * rot
        skew = torch.tensor([[0.0, -aa[2], aa[1]], [aa[2], 0.0, -aa[0]], [-aa[1], aa[0], 0.0]])
        T = torch.eye(4)
        T[:3, :3] = torch.linalg.matrix_exp(skew)
        T[:3, 3] = (torch.rand(3, generator=g) * 2 - 1) * trans
        return T

    K = torch.tensor([[250.0, 0.0, 160.0, 0.0], [0.0, 250.0, 128.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    proj = torch.stack([
        torch.stack([K @ (torch.eye(4) if c == n_cam - 1 else rigid(0.4, 0.3)) for c in range(n_cam)])
        for _ in range(n_frames)
    ])
    neck = {
        "neck_maps": torch.randn(n_frames, n_cam, N_CELLS, 256, generator=g),
        "neck_3d": torch.randn(n_frames, n_cam, N_TOK_CAM, 128, generator=g),
        "projection_mat": proj,
        "image_wh": torch.tensor([320.0, 256.0]).expand(n_frames, n_cam, 2).contiguous(),
        "T_ego2world": torch.stack([rigid(3.1, 1.0) for _ in range(n_frames)]),
    }
    return {k: v.to(device) for k, v in neck.items()}
