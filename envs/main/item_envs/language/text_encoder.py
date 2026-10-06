"""The frozen HoloBrain GD text path: sentence -> fp16-ready [L, 256] tokens.

HoloBrain's text encoder is robo_orchard_lab's bip3d ``BertModel``
(model.config.json ``text_encoder``: bert-base-uncased, sub-sentence
representation split on [CLS] / [SEP], no pooling layer, one embedded
layer) followed by ``text_feat_map`` (Linear 768 -> 256). The deployed
encoder (main/network/encoder.py) drops it; this module rebuilds it from
the checkpoint's own weights (``text_encoder.language_backbone.body.model.*``
and ``text_feat_map.*`` in model.safetensors) and runs the same forward as
``bip3d.bert.BertModel`` / ``BertEncoder`` with robo_orchard_lab's own
tokenizer / mask helpers. Only the tokenizer + BERT config come from the
HF cache (no pretrained BERT weights are downloaded).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["HoloBrainText", "ckpt_dir"]

_TEXT_PREFIX = "text_encoder.language_backbone.body.model."
_MAP_PREFIX = "text_feat_map."


def ckpt_dir() -> Path:
    """HOLOBRAIN_CKPT (the root or its pretrain/ subdir holding model.config.json)."""
    root = Path(os.environ.get("HOLOBRAIN_CKPT", "/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD"))
    return root if (root / "model.config.json").exists() else root / "pretrain"


class HoloBrainText:
    """``embed(sentences) -> (emb [B, L, 256] float32, valid [B, L] bool)``."""

    def __init__(self, ckpt: str | Path | None = None, device: str = "cpu"):
        import torch
        from safetensors import safe_open
        from transformers import AutoTokenizer, BertConfig
        from transformers import BertModel as HFBertModel

        self.ckpt = Path(ckpt) if ckpt is not None else ckpt_dir()
        cfg = json.loads((self.ckpt / "model.config.json").read_text())["text_encoder"]
        name = "bert-base-uncased" if "bert-base-uncased" in str(cfg.get("name", "")) else str(cfg["name"])
        self.max_tokens = int(cfg.get("max_tokens", 256))
        self.pad_to_max = bool(cfg.get("pad_to_max", True))
        self.sub_sentence = bool(cfg.get("use_sub_sentence_represent", False))
        self.n_layers = int(cfg.get("num_layers_of_embedded", 1))
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.special_ids = self.tokenizer.convert_tokens_to_ids(list(cfg.get("special_tokens_list") or []))
        self.bert = HFBertModel(BertConfig.from_pretrained(name), add_pooling_layer=bool(cfg.get("add_pooling_layer", False)))
        with safe_open(str(self.ckpt / "model.safetensors"), framework="pt") as f:
            bert_sd = {k[len(_TEXT_PREFIX):]: f.get_tensor(k) for k in f.keys() if k.startswith(_TEXT_PREFIX)}
            map_sd = {k[len(_MAP_PREFIX):]: f.get_tensor(k) for k in f.keys() if k.startswith(_MAP_PREFIX)}
        missing, unexpected = self.bert.load_state_dict(bert_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"text encoder weights do not match BERT: missing {missing[:3]}, unexpected {unexpected[:3]}")
        w = map_sd["weight"]
        self.feat_map = torch.nn.Linear(w.shape[1], w.shape[0])
        self.feat_map.load_state_dict(map_sd)
        for m in (self.bert, self.feat_map):
            m.to(device).eval().requires_grad_(False)

    def embed(self, sentences: list[str]):
        import torch
        from robo_orchard_lab.models.bip3d.bert import generate_masks_with_special_tokens_and_transfer_map
        from robo_orchard_lab.utils.transformers_compat import batch_encode_tokenizer, prepare_hf_attention_mask

        tok = batch_encode_tokenizer(
            self.tokenizer, list(sentences), max_length=self.max_tokens,
            padding="max_length" if self.pad_to_max else "longest",
            return_special_tokens_mask=True, return_tensors="pt", truncation=True,
        ).to(self.device)
        if self.sub_sentence:
            mask, position_ids = generate_masks_with_special_tokens_and_transfer_map(tok, self.special_ids)
            token_type_ids = tok["token_type_ids"]
        else:
            mask, position_ids, token_type_ids = tok.attention_mask, None, None
        with torch.no_grad():
            out = self.bert(
                input_ids=tok.input_ids, attention_mask=prepare_hf_attention_mask(self.bert, mask, tok.input_ids.shape),
                position_ids=position_ids, token_type_ids=token_type_ids, output_hidden_states=True,
            )
            layers = out.hidden_states[1:]  # 12 hidden layers (BertEncoder.forward)
            feats = torch.stack(layers[-self.n_layers:], 1).mean(1) / self.n_layers
            if mask.dim() == 2:
                feats = feats * mask.unsqueeze(-1).float()
            emb = self.feat_map(feats)
        return emb.float(), tok.attention_mask.bool()
