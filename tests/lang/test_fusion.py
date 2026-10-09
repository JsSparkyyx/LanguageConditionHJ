"""main/lang/fusion.py (fusion_impl.md 4.1, section 5): the frozen HoloBrain fusion stage.

CPU tests load the real feature_enhancer / spatial_enhancer weights from the checkpoint (safetensors, ~90 MB) and
skip when it is missing; inputs are synthetic neck dicts of the real level shapes. References are built from the
official modules (the enhancer's own forward, bip3d's mask helper, HoloBrainEncoder's own spatial methods called on
a stand-in object), not from fusion.py's helpers.

``gpu`` tests need the HoloBrain encoder / BERT and CUDA; they skip without CUDA unless LANG_FUSION_TEST_CPU=1 (runs
them on the CPU, a few minutes).
"""

import contextlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from main.lang import fusion as fz

CKPT = Path(os.environ.get("HOLOBRAIN_CKPT", "/work/nvme/bifq/project/tmp/holobrain/HoloBrain_v0.0_GD"))
HAS_CKPT = (CKPT / "pretrain" / "model.safetensors").exists() or (CKPT / "model.safetensors").exists()
needs_ckpt = pytest.mark.skipif(not HAS_CKPT, reason=f"HoloBrain checkpoint not found at {CKPT}")


def _gpu_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("LANG_FUSION_TEST_CPU") == "1":
        return "cpu"
    pytest.skip("needs CUDA (LANG_FUSION_TEST_CPU=1 runs it on the CPU)")


def _tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        pytest.skip("bert-base-uncased tokenizer not available offline")


@pytest.fixture(scope="module")
def stage():
    if not HAS_CKPT:
        pytest.skip(f"HoloBrain checkpoint not found at {CKPT}")
    return fz.FusionStage(str(CKPT), device="cpu", chunk=32)


def _words(F, lengths, seed=0, pad_seed=None):
    """Random words [F, 24, 256] with prefix masks of the given lengths; padding rows random (pad_seed) or zero."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(F, 24, 256, generator=g)
    m = torch.arange(24)[None] < torch.tensor(lengths)[:, None]
    gp = torch.Generator().manual_seed(pad_seed) if pad_seed is not None else None
    pad = torch.randn(F, 24, 256, generator=gp) if gp is not None else torch.zeros(F, 24, 256)
    return torch.where(m[..., None], w, pad), m


def _ref_maps(cells, shapes):
    """[1, n_cam, cells, C] -> list of [1, n_cam, C, h, w], written independently of fusion._cells_to_maps."""
    out, s = [], 0
    for h, w in shapes:
        blk = cells[:, :, s:s + h * w]                       # [1, n_cam, h*w, C]
        out.append(blk.permute(0, 1, 3, 2).reshape(blk.shape[0], blk.shape[1], blk.shape[3], h, w).contiguous())
        s += h * w
    return out


def _ref_text_dict(words_n):
    """text_dict of one unpadded sentence via bip3d's own mask helper on [CLS] ids.. [SEP] token ids."""
    from robo_orchard_lab.models.bip3d.bert import generate_masks_with_special_tokens_and_transfer_map

    n = words_n.shape[1]
    ids = torch.tensor([[101] + [2000 + i for i in range(n - 2)] + [102]])
    masks, position_ids = generate_masks_with_special_tokens_and_transfer_map({"input_ids": ids}, [101, 102])
    return {"embedded": words_n, "position_ids": position_ids, "masks": masks,
            "text_token_mask": torch.ones(1, n, dtype=torch.bool)}


def _encoder_standin(stage):
    """An object HoloBrainEncoder's encode_spatial / spatial_token_positions can run on (they read only these)."""
    return SimpleNamespace(spatial_enhancer=stage.spatial_enhancer, _device="cpu", _dtype=torch.float32)


def _ref_spatial(stage, maps12, neck, f):
    """HoloBrainEncoder.encode_spatial + the encode_visual_tokens flatten + spatial_token_positions, frame f alone,
    with the live inputs' types (image_wh int64 [n_cam, 2], one out_extrinsic) -> tokens [1200, 256], pos [1200, 3]."""
    from main.network.encoder import HoloBrainEncoder

    ns = _encoder_standin(stage)
    wh = neck["image_wh"][f].long()
    proj = neck["projection_mat"][f:f + 1].double()
    f3d = _ref_maps(neck["neck_3d"][f:f + 1], [fz.LEVEL_SHAPES[i] for i in fz.TOKEN_LEVELS])
    fused, depth_prob = HoloBrainEncoder.encode_spatial(ns, maps12, wh, proj, f3d)
    tokens = torch.cat([x.flatten(3).transpose(2, 3) for x in fused], dim=2)
    pos, _ = HoloBrainEncoder.spatial_token_positions(ns, depth_prob, wh, proj, fused,
                                                      out_extrinsic=neck["T_ego2world"][f].double().numpy())
    return tokens.flatten(1, 2)[0], pos.flatten(1, 2)[0]


def _ref_fuse_frame(stage, neck, words, wmask, f):
    """Frame f the HoloBrain way: the official enhancer forward on its unpadded sentence, then the spatial enhancer."""
    n = int(wmask[f].sum())
    with fz.tf32_flags(False), torch.no_grad():
        maps = _ref_maps(neck["neck_maps"][f:f + 1], fz.LEVEL_SHAPES)
        fused, w_f = stage.enhancer(feature_maps=maps, text_dict=_ref_text_dict(words[f:f + 1, :n].clone()))
        tok, pos = _ref_spatial(stage, [fused[i] for i in fz.TOKEN_LEVELS], neck, f)
    return tok, pos, w_f[0]


# ---------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------


def test_msda_lowmem_matches_hf():
    """Same signature and result as transformers' pure-PyTorch function, values and gradients, real level shapes."""
    from transformers.models.deformable_detr.modeling_deformable_detr import MultiScaleDeformableAttention

    g = torch.Generator().manual_seed(0)
    B, heads, d, P = 2, 8, 32, 4
    shapes_list = [tuple(s) for s in fz.LEVEL_SHAPES]
    shapes = torch.tensor(shapes_list)
    start = torch.tensor(fz.LEVEL_START)
    Q = 50
    value = torch.randn(B, fz.N_CELLS, heads, d, generator=g)
    loc = torch.rand(B, Q, heads, len(shapes_list), P, 2, generator=g) * 1.2 - 0.1   # some samples off the map
    aw = torch.randn(B, Q, heads, len(shapes_list) * P, generator=g).softmax(-1).view(B, Q, heads, len(shapes_list), P)
    hf = MultiScaleDeformableAttention()
    with torch.no_grad():
        ref = hf(value, shapes, shapes_list, start, loc, aw, 1)
        out = fz.msda_lowmem(value, shapes, shapes_list, start, loc, aw, 1)
    assert out.shape == ref.shape == (B, Q, heads * d)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    v1, v2 = value.clone().requires_grad_(True), value.clone().requires_grad_(True)
    hf(v1, shapes, shapes_list, start, loc, aw, 1).square().sum().backward()
    fz.msda_lowmem(v2, shapes, shapes_list, start, loc, aw, 1).square().sum().backward()
    torch.testing.assert_close(v2.grad, v1.grad, atol=1e-4, rtol=1e-5)


def test_text_dict_for_length_matches_bert_masks():
    """position_ids / masks / text_token_mask equal bip3d's helper on real tokenizer output, padded and truncated."""
    from robo_orchard_lab.models.bip3d.bert import generate_masks_with_special_tokens_and_transfer_map

    tok = _tokenizer()
    special = tok.convert_tokens_to_ids(["[CLS]", "[SEP]"])
    sentences = ["", "don't touch the red mug", "keep the small green box upright while you slide the tray",
                 "the beads must stay inside the bowl", "avoid the pill bottle"]
    for padding, max_length in (("longest", 768), ("max_length", 24)):
        t = tok(sentences, padding=padding, max_length=max_length, truncation=True, return_tensors="pt")
        masks, pos = generate_masks_with_special_tokens_and_transfer_map(t, special)
        L = t["input_ids"].shape[1]
        for i in range(len(sentences)):
            n = int(t["attention_mask"][i].sum())
            td = fz.text_dict_for_length(n, L)
            assert torch.equal(td["position_ids"], pos[i]), (sentences[i], padding)
            assert torch.equal(td["masks"], masks[i]), (sentences[i], padding)
            assert torch.equal(td["text_token_mask"], t["attention_mask"][i].bool())
            for Lt in (2, max(2, n - 1)):  # truncated rows are the slice of the full masks
                tt = fz.text_dict_for_length(n, Lt)
                assert torch.equal(tt["masks"], masks[i, :Lt, :Lt]) and torch.equal(tt["position_ids"], pos[i, :Lt])
    assert int(t["attention_mask"][0].sum()) == 2  # "" -> [CLS] [SEP], the content-free sentence of E4


def test_precision_context_restores_tf32_flags():
    m0, c0 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        for outer in ((True, False), (False, True), (True, True)):
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = outer
            for on in (False, True):
                with fz.tf32_flags(on):
                    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == (on, on)
                assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == outer
            with pytest.raises(RuntimeError):
                with fz.tf32_flags(True):
                    raise RuntimeError("inside")
            assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == outer
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m0, c0


def test_stage_precision_per_dtype(stage):
    """fp32 and bf16 run the stage with TF32 off, tf32 with it on; dtype is read at call time; on CPU no autocast."""
    m0, c0 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # a trainer's TF32 block
        for dt, on in (("fp32", False), ("tf32", True), ("bf16", False)):
            stage.dtype = dt
            with stage.precision():
                assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == (on, on)
            assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
            assert isinstance(stage._enhancer_autocast(), contextlib.nullcontext)
    finally:
        stage.dtype = "fp32"
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m0, c0
    with pytest.raises(ValueError):
        fz.FusionStage(str(CKPT), device="cpu", dtype="fp16")


@needs_ckpt
def test_strict_load_and_frozen(stage):
    """277 feature_enhancer keys loaded strictly from pretrain/, the deformable attention patched in all 6 layers,
    every module frozen and in eval mode; a given spatial enhancer is reused, not rebuilt."""
    assert stage.n_enhancer_keys == fz.N_ENHANCER_KEYS == 277
    assert stage.ckpt_dir.endswith("pretrain")
    assert stage.n_lowmem == 6
    assert all(isinstance(b.self_attn.func, fz._LowMemMSDA) for b in stage.enhancer.img_attn_blocks)
    for m in (stage.enhancer, stage.spatial_enhancer):
        assert not m.training and not any(p.requires_grad for p in m.parameters())
    from safetensors import safe_open

    with safe_open(str(Path(stage.ckpt_dir) / "model.safetensors"), framework="pt") as f:
        for prefix, mod in (("feature_enhancer.", stage.enhancer), ("spatial_enhancer.", stage.spatial_enhancer)):
            sd = mod.state_dict()
            assert set(sd) == {k[len(prefix):] for k in f.keys() if k.startswith(prefix)}
            assert all(torch.equal(v, f.get_tensor(prefix + k)) for k, v in sd.items())
    reuse = fz.FusionStage(str(CKPT), device="cpu", spatial_enhancer=stage.spatial_enhancer, lowmem=False)
    assert reuse.spatial_enhancer is stage.spatial_enhancer and reuse.n_lowmem == 0


# ---------------------------------------------------------------------------
# fuse / unfused behaviour
# ---------------------------------------------------------------------------


@needs_ckpt
def test_fuse_grouping_equals_per_frame_official_forward(stage):
    """Lengths 9, 5, 14, 9 with chunk 2: the two 9-word frames share one enhancer call, each frame equals the
    official forward on its own unpadded sentence (tokens, positions with its own T_ego2world, fused words)."""
    lengths = [9, 5, 14, 9]
    neck = fz.synthetic_neck(len(lengths), seed=3)
    words, wmask = _words(len(lengths), lengths, seed=4, pad_seed=5)
    stage.chunk = 2
    try:
        with stage.capture_bi_attention(layers=(0,)) as cap:
            tok, pos, w_fused = stage.fuse(neck, words, wmask)
    finally:
        stage.chunk = 32
    assert tok.shape == (4, 1200, 256) and pos.shape == (4, 1200, 3) and w_fused.shape == (4, 24, 256)
    assert sorted((c["n"], tuple(c["frames"])) for c in cap.calls) == [(5, (1,)), (9, (0, 3)), (14, (2,))]
    for f, n in enumerate(lengths):  # alone in its group: bit-identical; two frames per call: ~4e-6 (GEMM rounding)
        rt, rp, rw = _ref_fuse_frame(stage, neck, words, wmask, f)
        torch.testing.assert_close(tok[f], rt, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(pos[f], rp, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(w_fused[f, :n], rw, atol=2e-5, rtol=1e-5)
        assert not w_fused[f, n:].any()
    # the sentence and the frame matter: frames 0 and 3 share a length, not a scene or a sentence
    assert (tok[0] - tok[3]).abs().mean() > 1e-2


@needs_ckpt
def test_fuse_frame_independent_of_batch_mates_padding_and_chunk(stage):
    lengths = [7, 7, 12]
    neck = fz.synthetic_neck(3, seed=11)
    words, wmask = _words(3, lengths, seed=12)
    tok, pos, wf = stage.fuse(neck, words, wmask)
    first = {k: v[:1] for k, v in neck.items()}
    t1, p1, w1 = stage.fuse(first, words[:1], wmask[:1])                     # frame 0 alone
    torch.testing.assert_close(t1[0], tok[0], atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(p1[0], pos[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w1[0], wf[0], atol=2e-5, rtol=1e-5)
    garbage = words[:1].clone()                                               # same words, random padding rows
    garbage[0, 7:] = torch.randn(17, 256, generator=torch.Generator().manual_seed(99))
    t2, p2, w2 = stage.fuse(first, garbage, wmask[:1])
    assert torch.equal(t2, t1) and torch.equal(p2, p1) and torch.equal(w2, w1)
    stage.chunk = 1
    try:
        t3, p3, _ = stage.fuse(neck, words, wmask)
    finally:
        stage.chunk = 32
    torch.testing.assert_close(t3, tok, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(p3, pos, atol=1e-5, rtol=1e-5)
    other = words[:1].clone()                                                 # another sentence of the same length
    other[0, 1:6] = torch.randn(5, 256, generator=torch.Generator().manual_seed(13))
    t4, _, _ = stage.fuse(first, other, wmask[:1])
    assert (t4 - t1).abs().max() > 1e-4


@needs_ckpt
def test_fuse_rejects_bad_masks(stage):
    neck = fz.synthetic_neck(1, seed=0)
    words, wmask = _words(1, [6])
    holes = wmask.clone()
    holes[0, 2] = False
    for bad in (holes, torch.zeros_like(wmask)):
        with pytest.raises(ValueError):
            stage.fuse(neck, words, bad)
    with pytest.raises(ValueError):
        stage.fuse({**neck, "neck_maps": neck["neck_maps"][:, :, :1600]}, words, wmask)


@needs_ckpt
def test_unfused_matches_encoder_math_camera_major(stage):
    """unfused == HoloBrainEncoder.encode_spatial on the raw levels 1, 2 + its flatten + spatial_token_positions,
    flattened camera-major (cam 0 stride 16 then 32, cam 1, cam 2) exactly as rollout.py stores scene_tokens."""
    neck = fz.synthetic_neck(3, seed=21)
    tok, pos = stage.unfused(neck)
    assert tok.shape == (3, 1200, 256) and pos.shape == (3, 1200, 3) and tok.dtype == pos.dtype == torch.float32
    for f in range(3):
        cells = neck["neck_maps"][f:f + 1, :, fz.LEVEL_START[1]:fz.LEVEL_START[3]]
        rt, rp = _ref_spatial(stage, _ref_maps(cells, fz.LEVEL_SHAPES[1:3]), neck, f)
        torch.testing.assert_close(tok[f], rt, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pos[f], rp, atol=1e-5, rtol=1e-5)
    # camera-major: permuting the cameras' inputs permutes the 400-token blocks
    perm = [2, 0, 1]
    per_cam = ("neck_maps", "neck_3d", "projection_mat", "image_wh")
    pn = {k: (v[:, perm] if k in per_cam else v) for k, v in neck.items()}
    tp, _ = stage.unfused(pn)
    for i, c in enumerate(perm):
        torch.testing.assert_close(tp[:, 400 * i:400 * (i + 1)], tok[:, 400 * c:400 * (c + 1)], atol=1e-5, rtol=1e-5)
    # fusion rewrites the tokens and moves their positions (the depth comes from the fused maps)
    words, wmask = _words(3, [8, 8, 8], seed=22)
    ft, fp, _ = stage.fuse(neck, words, wmask)
    assert (ft - tok).abs().mean() > 1e-2 and (fp - pos).abs().max() > 1e-6


@needs_ckpt
def test_encode_obs_with_neck_and_row_roundtrip(stage):
    """With a stand-in encoder (random neck levels, HoloBrainEncoder's own spatial methods): the tokens equal
    encode_visual_tokens bit for bit, the neck dict has the section-3 layout, unfused(neck) gives the same tokens,
    and the buffer row keeps fp16 maps / fp32 geometry."""
    from main.network.encoder import HoloBrainEncoder

    g = torch.Generator().manual_seed(31)
    levels = [torch.randn(1, 3, 256, h, w, generator=g) for h, w in fz.LEVEL_SHAPES]
    levels3d = [torch.randn(1, 3, 128, h, w, generator=g) for h, w in fz.LEVEL_SHAPES]

    class StandIn(SimpleNamespace):
        encode_spatial = HoloBrainEncoder.encode_spatial
        spatial_token_positions = HoloBrainEncoder.spatial_token_positions
        encode_visual_tokens = HoloBrainEncoder.encode_visual_tokens

        def extract_vision_features(self, imgs, depths):
            return [x.clone() for x in levels], [x.clone() for x in levels3d]

    enc = StandIn(spatial_enhancer=stage.spatial_enhancer, _device="cpu", _dtype=torch.float32)
    geo = fz.synthetic_neck(1, seed=32)
    obs = {"imgs": torch.zeros(1, 3, 3, 256, 320), "depths": torch.ones(1, 3, 1, 256, 320),
           "image_wh": np.array([[320, 256]] * 3, dtype=np.int64),
           "projection_mat": geo["projection_mat"].double(), "T_ego2world": geo["T_ego2world"][0].double().numpy()}
    (tok, pos), neck = fz.encode_obs_with_neck(enc, obs)
    rt, rp, _ = enc.encode_visual_tokens(obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"],
                                         out_extrinsic=obs["T_ego2world"])
    assert torch.equal(tok, rt) and torch.equal(pos, rp)
    assert {k: tuple(v.shape) for k, v in neck.items()} == {
        "neck_maps": (1, 3, 1700, 256), "neck_3d": (1, 3, 400, 128), "projection_mat": (1, 3, 4, 4),
        "image_wh": (1, 3, 2), "T_ego2world": (1, 4, 4)}
    assert all(v.dtype == torch.float32 for v in neck.values())
    assert torch.equal(neck["neck_maps"][0, 1, 1280 + 20 * 3 + 7], levels[1][0, 1, :, 3, 7])   # level 1, row 3, col 7
    assert torch.equal(neck["neck_3d"][0, 2, 320 + 10 * 5 + 2], levels3d[2][0, 2, :, 5, 2])    # level 2, row 5, col 2
    ut, up = stage.unfused(neck)
    torch.testing.assert_close(ut[0], tok.flatten(1, 2)[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(up[0], pos.flatten(1, 2)[0], atol=1e-5, rtol=1e-5)

    row = fz.neck_to_row(neck)
    assert {k: (v.shape, v.dtype) for k, v in row.items()} == {
        "neck_maps": ((3, 1700, 256), np.float16), "neck_3d": ((3, 400, 128), np.float16),
        "projection_mat": ((3, 4, 4), np.float32), "image_wh": ((3, 2), np.float32),
        "T_ego2world": ((4, 4), np.float32)}
    batch = {k: np.stack([v, v]) for k, v in row.items()}
    batch["episode_id"] = np.array([5, 5], np.int64)
    back = fz.rows_to_neck(batch, "cpu")
    assert back["neck_maps"].dtype == torch.float32 and back["episode_id"].dtype == torch.int64
    assert torch.equal(back["neck_maps"][1], neck["neck_maps"][0].half().float())
    st, sp = stage.unfused(back)
    assert (st[0] - ut[0]).abs().max() < 5e-2 and (sp[0] - up[0]).abs().max() < 1e-2   # fp16 storage only
    with pytest.raises(ValueError):
        fz.neck_to_row({**neck, "neck_maps": neck["neck_maps"] * 1e6})


def test_neck_row_is_the_neck_buffer_row(tmp_path):
    """The two halves of the format agree (fusion_impl.md section 3): a neck_to_row row is exactly NeckBuffer's
    ROW_FIELDS, stores without a cast beyond its dtypes, reads back unchanged, and rows_to_neck of a NeckBuffer read
    passes the stage's shape check. No checkpoint needed."""
    from main.lang.neck_buffer import NECK_FIELDS, ROW_FIELDS, NeckBuffer

    neck = fz.synthetic_neck(1, seed=5)
    row = fz.neck_to_row(neck)
    assert tuple(row) == ROW_FIELDS
    assert {k: (v.shape, v.dtype) for k, v in row.items()} == {k: NECK_FIELDS[k] for k in ROW_FIELDS}
    assert fz.N_CELLS == NECK_FIELDS["neck_maps"][0][1] and fz.N_TOK_CAM == NECK_FIELDS["neck_3d"][0][1]
    nb = NeckBuffer(tmp_path / "neck_buffer", capacity=3)
    nb.append(dict(row, episode_id=7, t_in_ep=2))
    nb.flush()
    got = NeckBuffer(tmp_path / "neck_buffer").read(np.array([0, 0]))
    for k in ROW_FIELDS:
        assert np.array_equal(got[k][1], row[k])
    back = fz.rows_to_neck(got, "cpu")
    assert fz.FusionStage._check(None, back, 2) == 3
    assert torch.equal(back["neck_maps"][0], neck["neck_maps"][0].half().float())


@needs_ckpt
def test_capture_bi_attention_reproduces_the_module(stage):
    """The captured softmaxes, applied to the module's own value projections, give back the module's outputs."""
    lengths = [6, 10]
    neck = fz.synthetic_neck(2, seed=41)
    words, wmask = _words(2, lengths, seed=42)
    outs = []
    attn = stage.enhancer.text_img_attn_blocks[5].attn
    h = attn.register_forward_hook(lambda m, a, o: outs.append((a[0].clone(), a[1].clone(), o[0].clone(),
                                                                o[1].clone())))
    try:
        with stage.capture_bi_attention(layers=(0, 5)) as cap:
            stage.fuse(neck, words, wmask)
    finally:
        h.remove()
    assert not stage._captures and len(cap.calls) == 2
    word, cell = cap.per_frame(5)
    for f, n in enumerate(lengths):
        assert word[f].shape == (4, 60, n) and cell[f].shape == (4, n, 60)
        torch.testing.assert_close(word[f].sum(-1), torch.ones(4, 60))
        torch.testing.assert_close(cell[f].sum(-1), torch.ones(4, n))
    for i, (vision, lang, dv, dl) in enumerate(outs):  # one enhancer call per length, in sorted-length order
        g, tgt, _ = vision.shape
        src = lang.shape[1]
        H, hd = attn.num_heads, attn.head_dim
        val_l = attn.values_l_proj(lang).view(g, src, H, hd).transpose(1, 2)
        val_v = attn.values_v_proj(vision).view(g, tgt, H, hd).transpose(1, 2)
        re_v = attn.out_v_proj((cap.word[5][i] @ val_l).transpose(1, 2).reshape(g, tgt, H * hd))
        re_l = attn.out_l_proj((cap.cell[5][i] @ val_v).transpose(1, 2).reshape(g, src, H * hd))
        torch.testing.assert_close(re_v, dv, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(re_l, dl, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError):
        with stage.capture_bi_attention(layers=(6,)):
            pass


# ---------------------------------------------------------------------------
# gpu: the real encoder and the BIP3D order
# ---------------------------------------------------------------------------


def _synthetic_obs(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    geo = fz.synthetic_neck(1, seed=seed)
    return {
        "imgs": torch.randn(1, 3, 3, 256, 320, generator=g),
        "depths": 0.2 + torch.rand(1, 3, 1, 256, 320, generator=g),
        "image_wh": torch.tensor([[320, 256]] * 3, dtype=torch.long),
        "projection_mat": geo["projection_mat"].double(),
        "T_ego2world": geo["T_ego2world"][0].double().numpy(),
    }


@pytest.mark.gpu
@needs_ckpt
def test_gpu_unfused_matches_encode_visual_tokens():
    """Live fp32 neck: encode_obs_with_neck is bit-identical to encode_visual_tokens, unfused(neck) within 1e-5;
    the fp16 buffer row is reported (stored-row parity, R6) and bounded loosely."""
    from main.network.encoder import HoloBrainEncoder

    dev = _gpu_device()
    enc = HoloBrainEncoder(str(CKPT), device=dev)
    obs = _synthetic_obs(dev, seed=1)
    ref_t, ref_p, _ = enc.encode_visual_tokens(obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"],
                                               out_extrinsic=obs["T_ego2world"])
    (tok, pos), neck = fz.encode_obs_with_neck(enc, obs)
    assert torch.equal(tok, ref_t) and torch.equal(pos, ref_p)
    stage = fz.FusionStage(str(CKPT), device=dev, spatial_enhancer=enc.spatial_enhancer)
    ut, up = stage.unfused(neck)
    torch.testing.assert_close(ut[0], ref_t.flatten(1, 2)[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(up[0], ref_p.flatten(1, 2)[0], atol=1e-5, rtol=0)
    st, sp = stage.unfused(fz.rows_to_neck({k: v[None] for k, v in fz.neck_to_row(neck).items()}, dev))
    dt, dp = float((st - ut).abs().max()), float((sp - up).abs().max())
    print(f"fp16 neck row: max |d tokens| {dt:.3g}, max |d pos| {dp:.3g} m")
    assert dt < 5e-2 and dp < 1e-2


@pytest.mark.gpu
@needs_ckpt
def test_gpu_fuse_matches_bip3d_order(monkeypatch):
    """BIP3D's own modules in BIP3D._forward's order on the same maps: bip3d BertModel (checkpoint weights) ->
    text_feat_map -> official enhancer -> spatial enhancer on all 4 levels, levels 1, 2 kept; against fuse() fed by
    our text path (HoloBrainText.embed, padded to 24 with garbage), two sentences of different length in one call."""
    import transformers
    from safetensors import safe_open

    from main.item_envs.language.text_encoder import HoloBrainText
    from main.network.encoder import HoloBrainEncoder
    from robo_orchard_lab.models.bip3d.bert import BertModel

    dev = _gpu_device()
    _tokenizer()
    stage = fz.FusionStage(str(CKPT), device=dev)
    cfg = dict(stage.cfg["text_encoder"])
    cfg.pop("type")
    cfg["name"] = "bert-base-uncased"   # the config's ./ckpt/ path does not exist; HoloBrainText uses the hub name
    # only the tokenizer and config are cached: build BERT from its config, then load the checkpoint's own weights
    monkeypatch.setattr(transformers.BertModel, "from_pretrained", classmethod(
        lambda cls, name, add_pooling_layer=True, config=None, **kw: cls(config, add_pooling_layer=add_pooling_layer)))
    bert = BertModel(**cfg)
    with safe_open(str(Path(stage.ckpt_dir) / "model.safetensors"), framework="pt") as f:
        bert.load_state_dict({k[len("text_encoder."):]: f.get_tensor(k) for k in f.keys()
                              if k.startswith("text_encoder.")}, strict=True)
        fmap = torch.nn.Linear(768, 256)
        fmap.load_state_dict({k[len("text_feat_map."):]: f.get_tensor(k) for k in f.keys()
                              if k.startswith("text_feat_map.")}, strict=True)
    bert.to(dev).eval()
    fmap.to(dev).eval()
    ns = SimpleNamespace(spatial_enhancer=stage.spatial_enhancer, _device=dev, _dtype=torch.float32)

    sentences = ["don't touch the red mug", "keep the small green box upright while the tray slides"]
    text = HoloBrainText(device=dev)
    g = torch.Generator().manual_seed(51)
    words = torch.randn(2, 24, 256, generator=g).to(dev)                    # garbage padding
    wmask = torch.zeros(2, 24, dtype=torch.bool, device=dev)
    for i, s in enumerate(sentences):
        e, m = text.embed([s])
        words[i, :e.shape[1]] = e[0]
        wmask[i, :e.shape[1]] = m[0]
    geo = fz.synthetic_neck(2, seed=52, device=dev)
    levels = [torch.randn(2, 3, 256, h, w, generator=g).to(dev) for h, w in fz.LEVEL_SHAPES]
    levels3d = [torch.randn(2, 3, 128, h, w, generator=g).to(dev) for h, w in fz.LEVEL_SHAPES]
    neck = {**geo, "neck_maps": fz._maps_to_cells(levels), "neck_3d": fz._maps_to_cells(levels3d[1:3])}
    tok, pos, w_fused = stage.fuse(neck, words, wmask)

    for i, s in enumerate(sentences):
        with torch.no_grad(), fz.tf32_flags(False):
            td = bert([s])
            td["embedded"] = fmap(td["embedded"])
            inputs = {"image_wh": torch.tensor([[320, 256]] * 3, device=dev),
                      "projection_mat": geo["projection_mat"][i:i + 1]}
            maps = [x[i:i + 1] for x in levels]
            f3d = [x[i:i + 1] for x in levels3d]
            fm, tf = stage.enhancer(feature_maps=maps, feature_3d=f3d, text_dict=td, inputs=inputs)
            fused, depth_prob, _ = stage.spatial_enhancer(feature_maps=fm, feature_3d=f3d, text_dict=td, inputs=inputs)
            ref_t = torch.cat([fused[j].flatten(3).transpose(2, 3) for j in fz.TOKEN_LEVELS], dim=2).flatten(1, 2)[0]
            ref_p, _ = HoloBrainEncoder.spatial_token_positions(
                ns, depth_prob, inputs["image_wh"], inputs["projection_mat"], fused,
                out_extrinsic=geo["T_ego2world"][i], feature_level=list(fz.TOKEN_LEVELS))
        n = tf.shape[1]
        assert n == int(wmask[i].sum())
        torch.testing.assert_close(tok[i], ref_t, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pos[i], ref_p.flatten(1, 2)[0], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(w_fused[i, :n], tf[0], atol=1e-4, rtol=1e-4)
