"""HoloBrain-Qwen's frozen front end, composed (V6, qwen_impl.md 2).

HoloBrain_v0.0_Qwen replaces the GD front end (Swin-T + neck + BERT + text_feat_map + feature_enhancer) by one
bf16 Qwen2.5-VL-3B cut to language layer 0: the three camera images sit inside the prompt

    <|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n
    <|vision_start|><|image_pad|><|vision_end|> x 3 (space separated)\\nYou are a robot. {sentence}<|im_end|>...

each image pad expanded to 99 vision tokens (a 9 x 11 grid of 28-pixel cells at 308 x 252). Every position's
feature is a mix of two hidden states, the input embedding and norm(layer0(.)), each through its own bf16
Linear(2048, 256), weighted by softmax(weight) (``foward_feat_mapping``, structure.py:236-250). The image slots
then go through the depth Swin (stride 28) and the spatial enhancer, as the GD maps do: 297 scene tokens per
frame, each with a world position. The sentence's own positions are the words.

The prompt is causal and the images come first, so the scene tokens never depend on the sentence. Splitting the
prompt at the "." of "You are a robot." (325 positions) gives:
- ``encode_frame``: the images -> scene tokens, positions, and layer 0's keys / values of the 325-token prefix;
- ``encode_text``: any sentence against stored keys / values -> its words, one layer-0 pass of 3-15 tokens.
So training relabels the sentence without re-running the vision tower: the shards keep the K / V (kv_buffer.py).

Built here, not through HoloBrain's class: that class hard-codes flash_attention_2, loads the full 36-layer base
before cutting it and builds the action decoder. Everything needed is in the HoloBrain pretrain checkpoint
(``vlm.*``, ``feat_mapping.*``, ``weight``, ``backbone_3d.*``, ``spatial_enhancer.*``); the Qwen base folder gives
only the config, tokenizer and image processor files.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

DELTA_HF_HOME = "/work/nvme/bifq/project/tmp/hf-cache"


def hf_snapshot(model: str, rev: str, sub: str = "") -> str:
    """Snapshot ``rev`` of the Hugging Face model ``model`` ("org/name") in Delta's cache (the only place looked at
    before 2026-11-09), else in $HF_HOME/hub (another machine, e.g. Bridges-2: download the same revision there,
    scripts/lang/bridges/README.md). Neither there: the $HF_HOME path (Delta's without HF_HOME), and loading fails
    naming it."""
    tail = Path("hub") / ("models--" + model.replace("/", "--")) / "snapshots" / rev / sub
    homes = [Path(DELTA_HF_HOME)] + ([Path(os.environ["HF_HOME"])] if os.environ.get("HF_HOME") else [])
    for home in homes:
        if (home / tail).exists():
            return str(home / tail)
    return str(homes[-1] / tail)


QWEN_CKPT = os.environ.get("HOLOBRAIN_QWEN_CKPT") or hf_snapshot(
    "HorizonRobotics/HoloBrain_v0.0_Qwen", "b82becd9b5f0cd72ef22796d894de292ac2b0333", "pretrain")
QWEN_BASE = os.environ.get("QWEN_BASE_DIR") or hf_snapshot(
    "Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3")

SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
ROBOT = "\nYou are a robot."
TAIL_END = "<|im_end|>\n<|im_start|>assistant\n"
TEMPLATE_END_IDS = [151645, 198, 151644, 77091, 198]
PAD_ID = 151643          # <|endoftext|>, the tokenizer's pad
IMAGE_PAD_ID = 151655    # <|image_pad|>
ATTNS = ("sdpa", "flash_attention_2", "eager")


def full_prompt(sentence: str, n_cam: int = 3) -> str:
    """HoloBrain's TextTemplate (structure.py:55-67) for one instruction."""
    return SYSTEM + " ".join([IMAGE_BLOCK] * n_cam) + ROBOT + " " + sentence + TAIL_END


def prefix_prompt(n_cam: int = 3) -> str:
    """The prompt through the "." of "You are a robot." (no trailing space: it belongs to the sentence)."""
    return SYSTEM + " ".join([IMAGE_BLOCK] * n_cam) + ROBOT


def checkpoint_id(path: str | Path) -> str:
    """sha256[:16] of a safetensors file's header bytes and its size (the tensor index; cheap to read)."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(8)
        (n,) = struct.unpack("<Q", head)
        h = hashlib.sha256(head + f.read(n))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:16]


class FeatMapping(nn.Module):
    """HoloBrain's two bf16 Linear(2048, 256) and the bf16 mixing logits, computed as foward_feat_mapping."""

    def __init__(self, hidden: int = 2048, dim: int = 256, n: int = 2, dtype=torch.bfloat16):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, dim, bias=True, dtype=dtype) for _ in range(n)])
        self.weight = nn.Parameter(torch.zeros(n, dtype=dtype))

    def forward(self, hidden_states) -> torch.Tensor:
        x = torch.stack(list(hidden_states), dim=0)                       # [n, B, L, hidden]
        w = torch.stack([m.weight for m in self.layers], dim=0)[:, None]  # [n, 1, dim, hidden]
        x = x @ w.mT
        x = x + torch.stack([m.bias for m in self.layers], dim=0)[:, None, None]
        x = x.permute(1, 2, 3, 0)                                         # [B, L, dim, n]
        return (x @ torch.softmax(self.weight, dim=0)).float()            # bf16 arithmetic, then fp32


def _as_tensor(x, device, dtype=None):
    t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x))
    return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device)


class HoloBrainQwenEncoder:
    """A plain object holding frozen modules (never a submodule of the trained model); every call is no_grad."""

    IMAGE_WH = (308, 252)
    N_CAM = 3
    GRID = (9, 11)
    N_TOK_CAM = 99
    N_TOK = 297
    PREFIX_LEN = 325
    ROPE_NEXT = 61
    N_KV_HEADS = 2
    HEAD_DIM = 128
    MAX_WORDS = 24
    PATCH = 28
    # 1: the collection-time encoder of 2026-10-14 (rotary inv_freq rounded to bf16); 2: float32 inv_freq, as
    # HoloBrain-Qwen (bit-identical to its own model with flash_attention_2 on one GPU type). Stored in kv_buffer/
    # headers; the trainer refuses rows of another version (scripts/lang/qwen_recompute.py rewrites them)
    ENCODER_VERSION = 2

    def __init__(self, ckpt_dir: str = QWEN_CKPT, base_dir: str = QWEN_BASE, device="cuda", attn: str = "sdpa",
                 vision: bool = True, text_chunk: int = 256):
        from safetensors import safe_open
        from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

        from main.network.build import build

        if attn not in ATTNS:
            raise ValueError(f"qwen attn {attn!r}: expected one of {ATTNS}")
        ckpt_dir = str(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, "pretrain")):
            ckpt_dir = os.path.join(ckpt_dir, "pretrain")
        with open(os.path.join(ckpt_dir, "model.config.json")) as f:
            cfg = json.load(f)
        if "Qwen" not in cfg["class_type"]:
            raise ValueError(f"HoloBrainQwenEncoder needs HoloBrain_v0.0_Qwen's pretrain/; got {cfg['class_type']}")
        st_path = os.path.join(ckpt_dir, "model.safetensors")

        qcfg = Qwen2_5_VLConfig.from_pretrained(base_dir)
        qcfg.text_config.num_hidden_layers = 1
        vlm = Qwen2_5_VLForConditionalGeneration._from_config(qcfg, attn_implementation=attn, dtype=torch.bfloat16)
        # parameters bf16 (HoloBrain's precision; the load below copies into these dtypes), buffers untouched: the
        # rotary inv_freq buffers stay float32, as in HoloBrain's from_pretrained build. A whole-model .to(bf16)
        # (encoder version 1) rounded them and moved the vision RoPE angles by up to 0.06 rad: image features 6-9 %
        # off HoloBrain's own on the same frame (pre-run review 2026-10-15)
        with torch.no_grad():
            for prm in vlm.parameters():
                if prm.dtype != torch.bfloat16:
                    prm.data = prm.data.to(torch.bfloat16)
        groups = {"vlm.": {}, "feat_mapping.": {}, "backbone_3d.": {}, "spatial_enhancer.": {}}
        mix = None
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k == "weight":
                    mix = f.get_tensor(k)
                    continue
                for p, d in groups.items():
                    if k.startswith(p):
                        d[k[len(p):]] = f.get_tensor(k)
                        break
        res = vlm.load_state_dict(groups["vlm."], strict=False)
        if list(res.missing_keys) != ["model.language_model.embed_tokens.weight"] or res.unexpected_keys:
            raise RuntimeError(f"HoloBrain-Qwen vlm load: missing {res.missing_keys}, unexpected {res.unexpected_keys}")
        if vlm.lm_head.weight is not vlm.model.language_model.embed_tokens.weight:
            raise RuntimeError("HoloBrain-Qwen: the token embedding is not tied to lm_head (the checkpoint stores the "
                               "fine-tuned table once, as vlm.lm_head.weight)")
        fm = FeatMapping(hidden=qcfg.text_config.hidden_size)
        fm.load_state_dict({**{f"layers.{k}": v for k, v in groups["feat_mapping."].items()}, "weight": mix},
                           strict=True)
        backbone_3d = build(cfg["backbone_3d"])
        backbone_3d.load_state_dict(groups["backbone_3d."], strict=True)
        spatial = build(cfg["spatial_enhancer"])
        spatial.load_state_dict(groups["spatial_enhancer."], strict=True)
        processor = AutoProcessor.from_pretrained(base_dir, use_fast=True)
        self._setup(vlm, fm, backbone_3d, spatial, processor, device=device, attn=attn, vision=vision,
                    text_chunk=text_chunk, ckpt_id=checkpoint_id(st_path))
        self.ckpt_dir, self.base_dir = ckpt_dir, str(base_dir)

    @classmethod
    def from_parts(cls, vlm, feat_mapping, backbone_3d, spatial_enhancer, processor, device="cpu", attn="sdpa",
                   vision=True, text_chunk=256, ckpt_id="parts"):
        """The same object around given modules (the CPU tests' tiny random Qwen2.5-VL)."""
        self = cls.__new__(cls)
        self._setup(vlm, feat_mapping, backbone_3d, spatial_enhancer, processor, device=device, attn=attn,
                    vision=vision, text_chunk=text_chunk, ckpt_id=ckpt_id)
        self.ckpt_dir = self.base_dir = None
        return self

    def _setup(self, vlm, feat_mapping, backbone_3d, spatial_enhancer, processor, *, device, attn, vision,
               text_chunk, ckpt_id):
        self.device = torch.device(device)
        self.attn, self.ckpt_id, self.text_chunk = attn, str(ckpt_id), int(text_chunk)
        rope = {n: b.dtype for n, b in vlm.named_buffers() if n.endswith("inv_freq")}
        if any(dt != torch.float32 for dt in rope.values()):
            raise RuntimeError(f"HoloBrain-Qwen: rotary inv_freq buffers must stay float32 (HoloBrain's build), got {rope}")
        for m in (vlm, feat_mapping, backbone_3d, spatial_enhancer):
            m.to(self.device)
            m.eval()            # one call per step: robo_orchard_lab's SwinTransformer.train returns None
            m.requires_grad_(False)
        backbone_3d.float()
        spatial_enhancer.float()
        self.vlm, self.feat_mapping = vlm, feat_mapping
        self.backbone_3d, self.spatial_enhancer = backbone_3d, spatial_enhancer
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self._tail_cache: dict[str, list[int]] = {}

        ids = self.tokenizer(prefix_prompt(self.N_CAM).replace("<|image_pad|>", "<|image_pad|>" * self.N_TOK_CAM),
                             add_special_tokens=False)["input_ids"]
        if len(ids) != self.PREFIX_LEN:
            raise RuntimeError(f"HoloBrain-Qwen prefix is {len(ids)} tokens, expected {self.PREFIX_LEN}")
        self.prefix_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self.image_mask = self.prefix_ids == IMAGE_PAD_ID
        slots = torch.nonzero(self.image_mask).flatten().cpu().tolist()
        want = [i for c in range(self.N_CAM) for i in range(15 + 102 * c, 15 + 102 * c + self.N_TOK_CAM)]
        if slots != want:
            raise RuntimeError("HoloBrain-Qwen prefix: image slots are not 15-113, 117-215, 219-317")
        gh, gw = self.IMAGE_WH[1] // 14, self.IMAGE_WH[0] // 14
        self.grid_thw = torch.tensor([[1, gh, gw]] * self.N_CAM, dtype=torch.long, device=self.device)
        pos, _ = self.vlm.model.get_rope_index(self.prefix_ids[None], image_grid_thw=self.grid_thw,
                                               attention_mask=torch.ones_like(self.prefix_ids[None]))
        self.prefix_pos = pos.to(self.device)                        # [3, 1, 325]
        if int(self.prefix_pos.max()) + 1 != self.ROPE_NEXT:
            raise RuntimeError(f"HoloBrain-Qwen prefix: next rotary position {int(self.prefix_pos.max()) + 1}, "
                               f"expected {self.ROPE_NEXT}")
        self._blank = None
        self.vision = True
        if not vision:                 # the trainer's text pass: keep the blank prefix, then drop the tower
            self.blank_kv()
            del self.vlm.model.visual
            self.vision = False
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------- text ----
    @property
    def embed_tokens(self):
        return self.vlm.model.language_model.embed_tokens

    @property
    def language_model(self):
        return self.vlm.model.language_model

    def instruction_ids(self, sentence: str) -> list[int]:
        """The sentence's own token ids inside the prompt (the space before it included in its first token)."""
        ids = self._tail_cache.get(sentence)
        if ids is None:
            tail = self.tokenizer(" " + sentence + TAIL_END, add_special_tokens=False)["input_ids"]
            if tail[-5:] != TEMPLATE_END_IDS:
                raise RuntimeError(f"HoloBrain-Qwen tail of {sentence!r} does not end with the template end")
            ids = tail[:-5]
            if not 1 <= len(ids) <= self.MAX_WORDS:
                raise ValueError(f"sentence {sentence!r} is {len(ids)} Qwen tokens; the word budget is "
                                 f"1..{self.MAX_WORDS}")
            self._tail_cache[sentence] = ids
        return ids

    @torch.no_grad()
    def encode_text(self, kv, sentences, kv_index=None):
        """kv = (K, V) bf16 [U, 2, 325, 128] on the device; sentences list[str] (N); kv_index [N] into U (default:
        arange(N), or zeros when U == 1) -> words fp32 [N, 24, 256], mask bool [N, 24]."""
        from transformers import DynamicCache

        K, V = kv
        U, N = int(K.shape[0]), len(sentences)
        if kv_index is None:
            kv_index = torch.zeros(N, dtype=torch.long) if U == 1 else torch.arange(N)
            if U not in (1, N):
                raise ValueError(f"encode_text: {U} prefixes for {N} sentences and no kv_index")
        kv_index = torch.as_tensor(kv_index, dtype=torch.long, device=K.device).reshape(-1)
        if kv_index.numel() != N:
            raise ValueError(f"encode_text: kv_index has {kv_index.numel()} entries for {N} sentences")
        if tuple(K.shape[1:]) != (self.N_KV_HEADS, self.PREFIX_LEN, self.HEAD_DIM) or K.shape != V.shape:
            raise ValueError(f"encode_text: kv {tuple(K.shape)} / {tuple(V.shape)}")
        words = torch.zeros(N, self.MAX_WORDS, self.feat_dim, dtype=torch.float32, device=self.device)
        mask = torch.zeros(N, self.MAX_WORDS, dtype=torch.bool, device=self.device)
        ids_all = [self.instruction_ids(s) for s in sentences]
        for c0 in range(0, N, self.text_chunk):
            sl = slice(c0, min(N, c0 + self.text_chunk))
            ids = ids_all[sl]
            n = len(ids)
            T = max(len(x) for x in ids)
            tok = torch.full((n, T), PAD_ID, dtype=torch.long, device=self.device)
            for i, x in enumerate(ids):
                tok[i, :len(x)] = torch.tensor(x, dtype=torch.long)
            idx = kv_index[sl]
            cache = DynamicCache(ddp_cache_data=[(K[idx].to(self.device), V[idx].to(self.device))])
            pos = (self.ROPE_NEXT + torch.arange(T, device=self.device)).view(1, 1, T).expand(3, n, T)
            out = self.language_model(
                inputs_embeds=self.embed_tokens(tok), position_ids=pos, past_key_values=cache,
                cache_position=torch.arange(self.PREFIX_LEN, self.PREFIX_LEN + T, device=self.device),
                attention_mask=None, use_cache=True, output_hidden_states=True)
            feat = self.feat_mapping(out.hidden_states)               # [n, T, 256] fp32
            L = torch.tensor([len(x) for x in ids], device=self.device)
            keep = torch.arange(T, device=self.device)[None] < L[:, None]
            w = min(T, self.MAX_WORDS)
            words[sl, :w] = torch.where(keep[:, :w, None], feat[:, :w], 0.0)
            mask[sl, :w] = keep[:, :w]
        return words, mask

    @property
    def feat_dim(self) -> int:
        return int(self.feat_mapping.layers[0].out_features)

    # --------------------------------------------------------------- frames ----
    def pixel_values(self, rgb) -> torch.Tensor:
        """uint8 RGB [M, 252, 308, 3] -> the processor's patches [M * 396, 1176] (CLIP normalisation, no resize)."""
        imgs = _as_tensor(rgb, self.device)
        if tuple(imgs.shape[1:]) != (self.IMAGE_WH[1], self.IMAGE_WH[0], 3):
            raise ValueError(f"rgb frames must be [M, {self.IMAGE_WH[1]}, {self.IMAGE_WH[0]}, 3], got "
                             f"{tuple(imgs.shape)}")
        out = self.processor.image_processor(images=list(imgs.permute(0, 3, 1, 2)), return_tensors="pt",
                                             device=str(self.device))
        grid = out["image_grid_thw"].to(self.device)
        if not torch.equal(grid, self.grid_thw[:1].expand(len(imgs), 3)):
            raise RuntimeError(f"image processor grid {grid[0].tolist()}, expected {self.grid_thw[0].tolist()}")
        return out["pixel_values"].to(self.device)

    def image_maps(self, feat) -> torch.Tensor:
        """Prefix features [F, 325, C] -> the image slots as maps [F, 3, C, 9, 11] (cameras left, right, head; each
        slot block a row-major 9 x 11 raster), the layout HoloBrain's _vlm_outputs_handler gives the enhancer."""
        F = int(feat.shape[0])
        gh, gw = self.GRID
        return feat[:, self.image_mask].view(F, self.N_CAM, gh, gw, -1).permute(0, 1, 4, 2, 3).contiguous()

    @torch.no_grad()
    def prefix_pass(self, rgb):
        """rgb uint8 [F, 3, 252, 308, 3] -> (hidden states (2 x [F, 325, 2048] bf16), K, V [F, 2, 325, 128] bf16)."""
        if not self.vision:
            raise RuntimeError("HoloBrainQwenEncoder(vision=False): the vision tower was dropped (text pass only)")
        rgb = _as_tensor(rgb, self.device)
        F = int(rgb.shape[0])
        if tuple(rgb.shape[1:3]) != (self.N_CAM, self.IMAGE_WH[1]):
            raise ValueError(f"rgb must be [F, {self.N_CAM}, 252, 308, 3], got {tuple(rgb.shape)}")
        pv = self.pixel_values(rgb.flatten(0, 1)).to(self.vlm.model.visual.dtype)
        feats = self.vlm.model.visual(pv, grid_thw=self.grid_thw.repeat(F, 1))    # [F * 297, hidden]
        emb = self.embed_tokens(self.prefix_ids)[None].expand(F, -1, -1).clone()
        emb[:, self.image_mask] = feats.view(F, self.N_TOK, -1).to(emb.dtype)
        out = self.language_model(inputs_embeds=emb, position_ids=self.prefix_pos.expand(3, F, -1),
                                  use_cache=True, output_hidden_states=True)
        layer = out.past_key_values.layers[0]
        return out.hidden_states, layer.keys, layer.values

    @torch.no_grad()
    def encode_frame(self, rgb, depth, projection_mat, T_ego2world):
        """rgb uint8 [F,3,252,308,3]; depth float32 [F,3,252,308] metres; projection_mat [F,3,4,4] (ego frame at
        308 x 252); T_ego2world [F,4,4] -> tokens fp32 [F,3,99,256], pos fp32 [F,3,99,3] world metres,
        (K, V) bf16 [F,2,325,128]."""
        from .fusion import token_positions

        hs, K, V = self.prefix_pass(rgb)
        F = int(K.shape[0])
        maps = [self.image_maps(self.feat_mapping(hs))]
        d = _as_tensor(depth, self.device, torch.float32)
        if tuple(d.shape) != (F, self.N_CAM, self.IMAGE_WH[1], self.IMAGE_WH[0]):
            raise ValueError(f"depth must be [F, 3, 252, 308], got {tuple(d.shape)}")
        f3d = [x.unflatten(0, (F, self.N_CAM)) for x in self.backbone_3d(d.flatten(0, 1)[:, None])]
        image_wh = torch.tensor([list(self.IMAGE_WH)] * self.N_CAM, dtype=torch.float32, device=self.device)
        proj = _as_tensor(projection_mat, self.device, torch.float32)
        fused, depth_prob, _ = self.spatial_enhancer(
            feature_maps=maps, inputs={"image_wh": image_wh, "projection_mat": proj}, feature_3d=f3d)
        tokens = fused[0].flatten(3).transpose(2, 3).contiguous()              # [F, 3, 99, 256], raster per camera
        pos = token_positions(self.spatial_enhancer, depth_prob, image_wh, proj, fused,
                              _as_tensor(T_ego2world, self.device, torch.float32))
        return tokens.float(), pos.float(), (K, V)

    @torch.no_grad()
    def blank_kv(self):
        """(K, V) [1, 2, 325, 128] of the prefix with black images (RGB 0): the ``qwen_words_prefix = "blank"``
        control (words that read no scene). Cached."""
        if self._blank is None:
            black = torch.zeros(1, self.N_CAM, self.IMAGE_WH[1], self.IMAGE_WH[0], 3, dtype=torch.uint8)
            _, K, V = self.prefix_pass(black)
            self._blank = (K.contiguous(), V.contiguous())
        return self._blank

    def meta(self) -> dict:
        return {"prefix_len": self.PREFIX_LEN, "rope_next": self.ROPE_NEXT, "ckpt_id": self.ckpt_id, "attn": self.attn,
                "encoder_version": self.ENCODER_VERSION}
