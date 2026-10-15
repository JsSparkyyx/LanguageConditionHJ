"""HoloBrainQwenEncoder on a tiny random Qwen2.5-VL with HoloBrain-Qwen's layout (qwen_impl.md 2): the real
tokenizer and image processor files of the Qwen base folder, 2 KV heads of 128 dims, the mrope sections, a 2-block
vision tower with the same patching; the real depth Swin and spatial enhancer classes, random weights. fp32, so
the cached-prefix path can be compared tightly with one full-prompt pass. Nothing is downloaded. CPU only.

(a) parity: encode_text rows = the instruction rows of the full prompt; (b) causality: the image slots do not
depend on the sentence; (c) a batch of two sentences of different lengths = two batches of one; (d) the prefix
ids are the full prompt's first 325 ids (the processor's own expansion; the trailing-space trap); (e) the grid
order 297 = 3 x 9 x 11, cameras left, right, head."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from main.lang import qwen_encoder as qe

BASE = Path(qe.QWEN_BASE)
HB = Path(qe.QWEN_CKPT)
pytestmark = pytest.mark.skipif(not (BASE / "tokenizer.json").exists() or not (HB / "model.config.json").exists(),
                                reason="Qwen base / HoloBrain-Qwen config files not on this machine")

SENTENCES = ["Don't touch the red mug.", "keep the beads in the bowl", "Keep the tall blue block upright.",
             "3 cups must stay put", "'Mind the can'"]


def tiny_parts(dtype=torch.float32, seed=0):
    from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    from main.network.build import build

    torch.manual_seed(seed)
    cfg = Qwen2_5_VLConfig.from_pretrained(BASE)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_attention_heads, t.num_key_value_heads = 256, 512, 2, 2
    t.num_hidden_layers = 1
    v.hidden_size, v.intermediate_size, v.num_heads, v.depth, v.out_hidden_size = 64, 128, 2, 2, 256
    v.fullatt_block_indexes = [1]
    vlm = Qwen2_5_VLForConditionalGeneration._from_config(cfg, attn_implementation="sdpa", dtype=dtype)
    vlm.to(dtype)                      # the text config carries bf16 and wins over dtype= for the language model
    fm = qe.FeatMapping(hidden=256, dtype=dtype)
    with torch.no_grad():
        for m in fm.layers:
            m.weight.normal_(0, 0.05), m.bias.normal_(0, 0.1)
        fm.weight.copy_(torch.tensor([0.30078125, 0.451171875]))
    hb = json.loads((HB / "model.config.json").read_text())
    return vlm, fm, build(hb["backbone_3d"]), build(hb["spatial_enhancer"]), AutoProcessor.from_pretrained(
        BASE, use_fast=True)


@pytest.fixture(scope="module")
def enc():
    torch.set_num_threads(4)     # a shared login node: oversubscribed threads make these passes crawl
    return qe.HoloBrainQwenEncoder.from_parts(*tiny_parts(), device="cpu", text_chunk=2)


def frame(seed=0, F=1):
    g = np.random.default_rng(seed)
    rgb = torch.from_numpy(g.integers(0, 256, (F, 3, 252, 308, 3), dtype=np.uint8))
    depth = torch.from_numpy(g.uniform(0.3, 1.0, (F, 3, 252, 308)).astype(np.float32))
    K = torch.eye(4)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = 260.0, 260.0, 154.0, 126.0
    proj = K.expand(F, 3, 4, 4).clone()
    return rgb, depth, proj, torch.eye(4).expand(F, 4, 4).clone()


def full_pass(enc, rgb, sentence):
    """One pass over the whole prompt, as HoloBrain runs it (B = 1): -> (features [L, 256], ids, positions)."""
    ids = torch.tensor(enc.prefix_ids.tolist() + enc.instruction_ids(sentence) + qe.TEMPLATE_END_IDS)[None]
    with torch.no_grad():
        feats = enc.vlm.model.visual(enc.pixel_values(rgb.flatten(0, 1)), grid_thw=enc.grid_thw)
        emb = enc.embed_tokens(ids).clone()
        emb[0, ids[0] == qe.IMAGE_PAD_ID] = feats
        pos, _ = enc.vlm.model.get_rope_index(ids, image_grid_thw=enc.grid_thw, attention_mask=torch.ones_like(ids))
        out = enc.language_model(inputs_embeds=emb, position_ids=pos, output_hidden_states=True)
        return enc.feat_mapping(out.hidden_states)[0], ids[0], pos[:, 0]


def test_prefix_ids_and_positions_match_the_processor(enc):
    black = [np.zeros((252, 308, 3), np.uint8)] * 3
    for s in SENTENCES:
        got = enc.processor(text=[qe.full_prompt(s)], images=black, return_tensors="pt")["input_ids"][0].tolist()
        L = len(enc.instruction_ids(s))
        assert got[:325] == enc.prefix_ids.tolist()                        # (d)
        assert got[325:325 + L] == enc.instruction_ids(s) and got[325 + L:] == qe.TEMPLATE_END_IDS
        assert 3 <= L <= 15
    assert enc.tokenizer(" Don", add_special_tokens=False)["input_ids"] == enc.instruction_ids("Don")
    _, ids, pos = full_pass(enc, frame()[0], SENTENCES[0])
    L = len(enc.instruction_ids(SENTENCES[0]))
    assert torch.equal(pos[:, :325], enc.prefix_pos[:, 0])
    assert torch.equal(pos[:, 325:325 + L], (61 + torch.arange(L)).expand(3, L))
    with pytest.raises(ValueError, match="word budget"):
        enc.instruction_ids(" ".join(["block"] * 30))


def test_text_pass_equals_the_full_prompt(enc):
    rgb = frame(1)[0]
    _, K, V = enc.prefix_pass(rgb)
    assert K.shape == V.shape == (1, 2, 325, 128)
    words, mask = enc.encode_text((K, V), SENTENCES)                       # one prefix for all (U == 1), chunks of 2
    for i, s in enumerate(SENTENCES):
        ref, _, _ = full_pass(enc, rgb, s)
        L = len(enc.instruction_ids(s))
        assert mask[i].sum() == L and mask[i, :L].all()
        assert torch.allclose(words[i, :L], ref[325:325 + L], atol=1e-5, rtol=1e-4)      # (a)
        assert not words[i, L:].any()


def test_image_slots_do_not_read_the_sentence(enc):
    rgb = frame(2)[0]
    a, ids, _ = full_pass(enc, rgb, SENTENCES[0])
    b, _, _ = full_pass(enc, rgb, SENTENCES[2])
    hs, _, _ = enc.prefix_pass(rgb)
    pre = enc.feat_mapping(hs)[0]
    img = ids[:325] == qe.IMAGE_PAD_ID
    assert torch.equal(a[:325][img], b[:325][img])                         # (b)
    assert torch.allclose(pre[img], a[:325][img], atol=1e-5, rtol=1e-4)


def test_batches_and_kv_index(enc):
    rgb, _, _, _ = frame(3, F=2)
    _, K, V = enc.prefix_pass(rgb)
    s = [SENTENCES[0], SENTENCES[3]]                                       # 6 and 5 tokens: padded batch
    w2, m2 = enc.encode_text((K, V), s)                                    # U == N: row i reads prefix i
    for i in range(2):
        w1, m1 = enc.encode_text((K[i:i + 1], V[i:i + 1]), [s[i]])
        assert torch.equal(m1[0], m2[i]) and torch.allclose(w1[0], w2[i], atol=1e-5, rtol=1e-4)    # (c)
    w3, _ = enc.encode_text((K, V), [s[0], s[0], s[0]], kv_index=torch.tensor([1, 0, 1]))
    assert torch.equal(w3[0], w3[2]) and torch.allclose(w3[1], w2[0], atol=1e-5, rtol=1e-4)
    # two noise frames have nearly the same mean features (a random layer 0 attends almost uniformly); a black and
    # a white frame do not: the words read the frame they are given
    flat = torch.stack([torch.zeros_like(rgb[0]), torch.full_like(rgb[0], 255)])
    _, Kf, Vf = enc.prefix_pass(flat)
    wf, _ = enc.encode_text((Kf, Vf), [s[0], s[0]], kv_index=torch.tensor([0, 1]))
    assert (wf[0] - wf[1]).abs().max() > 1e-4
    with pytest.raises(ValueError, match="kv_index"):
        enc.encode_text((K, V), SENTENCES[:3])


def test_grid_order(enc):
    seq = torch.arange(325, dtype=torch.float32).view(1, 325, 1).expand(1, 325, 4)
    maps = enc.image_maps(seq)                                             # [1, 3, 4, 9, 11]
    tokens = maps.flatten(3).transpose(2, 3)                               # the enhancer's output order
    for c, r, q in [(0, 0, 0), (0, 8, 10), (1, 3, 4), (2, 0, 1), (2, 8, 10)]:
        assert int(maps[0, c, 0, r, q]) == 15 + 102 * c + 11 * r + q       # (e)
        assert int(tokens[0, c, 11 * r + q, 0]) == 15 + 102 * c + 11 * r + q


def test_encode_frame_shapes_and_blank(enc):
    rgb, depth, proj, T = frame(4, F=2)
    tokens, pos, (K, V) = enc.encode_frame(rgb, depth, proj, T)
    assert tokens.shape == (2, 3, 99, 256) and pos.shape == (2, 3, 99, 3)
    assert tokens.dtype == pos.dtype == torch.float32 and torch.isfinite(tokens).all() and torch.isfinite(pos).all()
    assert K.shape == (2, 2, 325, 128)
    t1, p1, (K1, _) = enc.encode_frame(rgb[1:], depth[1:], proj[1:], T[1:])
    assert torch.allclose(t1[0], tokens[1], atol=1e-4, rtol=1e-4) and torch.allclose(K1[0], K[1], atol=1e-5)
    Kb, Vb = enc.blank_kv()
    assert Kb.shape == (1, 2, 325, 128) and enc.blank_kv()[0] is Kb
    assert not torch.allclose(Kb[0], K[0])
    w, m = enc.encode_text(enc.blank_kv(), SENTENCES[:2])
    assert m.sum(1).tolist() == [len(enc.instruction_ids(s)) for s in SENTENCES[:2]]


def test_text_only_encoder_keeps_the_blank_prefix():
    vlm, fm, b3, se, proc = tiny_parts(seed=1)
    full = qe.HoloBrainQwenEncoder.from_parts(vlm, fm, b3, se, proc, device="cpu")
    Kb = full.blank_kv()[0].clone()
    vlm2, fm2, b32, se2, _ = tiny_parts(seed=1)
    txt = qe.HoloBrainQwenEncoder.from_parts(vlm2, fm2, b32, se2, proc, device="cpu", vision=False)
    assert not hasattr(txt.vlm.model, "visual") and torch.equal(txt.blank_kv()[0], Kb)
    with pytest.raises(RuntimeError, match="vision tower"):
        txt.prefix_pass(frame()[0])


def test_feat_mapping_is_holobrains_arithmetic():
    torch.manual_seed(0)
    fm = qe.FeatMapping(hidden=16, dim=8)
    with torch.no_grad():
        for m in fm.layers:
            m.weight.normal_(), m.bias.normal_()
        fm.weight.copy_(torch.tensor([0.30078125, 0.451171875]))
    x = [torch.randn(2, 5, 16).to(torch.bfloat16) for _ in range(2)]
    w = torch.softmax(fm.weight, dim=0)
    assert w.dtype == torch.bfloat16 and w.tolist() == [0.462890625, 0.5390625]

    def foward_feat_mapping(self, x):   # structure.py:236-250, verbatim
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
        weight = torch.stack(
            [layer.weight for layer in self.feat_mapping], dim=0
        )
        weight = weight[:, None]
        x = x @ weight.mT

        bias = torch.stack([layer.bias for layer in self.feat_mapping], dim=0)
        bias = bias[:, None, None]
        x = x + bias
        x = x.permute(1, 2, 3, 0)
        x = x @ torch.nn.functional.softmax(self.weight, dim=0)
        return x

    holo = type("H", (), {"feat_mapping": fm.layers, "weight": fm.weight})()
    with torch.no_grad():
        assert torch.equal(fm(x), foward_feat_mapping(holo, x).to(torch.float32))
        ref = torch.stack([torch.nn.functional.linear(x[i].float(), fm.layers[i].weight.float(), fm.layers[i].bias.float())
                           for i in range(2)], -1) @ w.float()
        assert fm(x).dtype == torch.float32 and torch.allclose(fm(x), ref, rtol=2e-2, atol=5e-2)


def test_checkpoint_id_reads_the_header(tmp_path):
    from safetensors.torch import save_file

    p = tmp_path / "a.safetensors"
    save_file({"x": torch.zeros(3)}, str(p))
    a = qe.checkpoint_id(p)
    save_file({"x": torch.ones(3)}, str(p))
    assert qe.checkpoint_id(p) == a                       # same index: same id (the header, not the values)
    save_file({"x": torch.ones(4)}, str(p))
    assert qe.checkpoint_id(p) != a and len(a) == 16
