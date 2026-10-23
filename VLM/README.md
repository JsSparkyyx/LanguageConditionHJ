# VLMs kept for later use (2026-10-22)

| folder | model | size | status |
|---|---|---|---|
| `Qwen2.5-VL-32B-Instruct/` | Qwen/Qwen2.5-VL-32B-Instruct | 64 GB (bf16) | downloaded; load test passed on 2 x A40 |
| `RoboBrain2.0-32B/` | BAAI/RoboBrain2.0-32B (fine-tuned from Qwen2.5-VL-32B-Instruct, Apache-2.0) | 63 GB (bf16) | downloaded; load test passed on 2 x A40 (gated repo: the token is in `~/hugginface`, pass it as `HF_TOKEN`) |

The weight folders are gitignored (`VLM/*/`); this README and `test_vlm.py` are tracked. Qwen2.5-VL-3B-Instruct
(HoloBrain-Qwen's base) is in the shared Hugging Face cache, `/work/nvme/bifq/project/tmp/hf-cache/hub`.

## Download

```
export HF_HUB_DISABLE_XET=1 HF_HOME=/work/nvme/bifq/project/tmp/hf-cache   # plain HTTP: little login-node CPU
cd VLM
/work/nvme/bifq/conda/envs/RoboTwin/bin/hf download Qwen/Qwen2.5-VL-32B-Instruct --local-dir Qwen2.5-VL-32B-Instruct --max-workers 4
/work/nvme/bifq/conda/envs/RoboTwin/bin/hf download BAAI/RoboBrain2.0-32B --local-dir RoboBrain2.0-32B --max-workers 4
```

## Run

Inside the project container (transformers 4.57.6 has Qwen2.5-VL; both models load with
`Qwen2_5_VLForConditionalGeneration`). bf16 needs about 66 GB of GPU memory for the weights: two A40s (48 GB each),
one A100 80 GB, or an H200.

```
srun -A bifq-delta-gpu -p gpuA40x4-interactive --gpus-per-node=2 --mem=160g --cpus-per-task=16 --time=00:30:00 \
    bash scripts/lang/in_container.sh python VLM/test_vlm.py VLM/Qwen2.5-VL-32B-Instruct [image] [question]
```

Load test (2026-10-22, 2 x A40, the T3 scene picture of the corpus page, "List the objects on the table and their
colours."): it named the black goal frame, the blue mat, the yellow cylinder and the green cube. RoboBrain2.0-32B,
same test: loaded in 56 s, answered in 5.4 s: "a black square, a blue square, a yellow cylinder, and a green cube".
