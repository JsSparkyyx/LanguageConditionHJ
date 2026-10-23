"""Load one of the VLMs under VLM/ and answer one question about one camera frame (a load / sanity test).

    bash scripts/lang/in_container.sh python VLM/test_vlm.py VLM/Qwen2.5-VL-32B-Instruct [image] [question]

Both models are Qwen2.5-VL (RoboBrain 2.0 32B is fine-tuned from Qwen2.5-VL-32B-Instruct), so one loader serves
both. bf16, device_map="auto": a 32B model needs about 66 GB of weights plus activations (one A100 80 GB / H200,
or two A40 48 GB).
"""

import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

REPO = Path(__file__).resolve().parents[1]
model_dir = sys.argv[1]
image = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "docs/lang_filter/corpus_page/img/scene_push_zone.jpg")
question = sys.argv[3] if len(sys.argv) > 3 else "List the objects on the table and their colours."

t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_dir)
print(f"loaded {model_dir} in {time.time() - t0:.0f} s; GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[Image.open(image).convert("RGB")], return_tensors="pt").to(model.device)
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
answer = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(f"[{time.time() - t0:.1f} s] Q: {question}\nA: {answer}")
