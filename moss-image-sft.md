# MOSS-Image-SFT Quick Launch Guide

Playbook for rapidly provisioning, configuring, and serving [`OpenMOSS-Team/MOSS-Image-SFT`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) on a Google Colab Pro A100 instance using the `colab` CLI.

---

## 1. Hardware & Memory Specifications

| Component | Value | Notes |
|---|---|---|
| **Target GPU** | NVIDIA A100 SXM4 (40GB or 80GB) | Standard Colab shape (`--gpu A100`) |
| **Model Weights** | **27.2 GB total** | Text encoder: 16.8 GB, Transformer: 10.2 GB, VAE: 0.25 GB |
| **VRAM Footprint** | **~25.4 GB** (allocated), **~32 GB** (reserved) | Precision: `bfloat16` (transformer/text encoder) + `float32` (FLUX.2 VAE) |
| **Download Time** | **~60–90 seconds** | On Google data center backbone |
| **Inference Time** | **~19s (25 steps)** / **~38s (50 steps)** | On A100-SXM4-40GB |

---

## 2. Fast Deployment (Copy-Paste Sequence)

Run these commands from your local terminal:

```bash
# 1. Provision A100 session
colab new -s moss-image-sft --gpu A100

# 2. Install base dependencies
colab install -s moss-image-sft "diffusers>=0.38" "transformers>=5.3" "accelerate>=1.10" "safetensors>=0.6" "Pillow>=10" gradio

# 3. Upgrade torchao (fixes Colab 0.10.0 import conflict with diffusers)
echo "!pip install -U torchao" | colab exec -s moss-image-sft

# 4. (Optional but recommended) Install high-speed CUDA kernels
echo "!pip install flash-linear-attention causal-conv1d --no-build-isolation" | colab exec -s moss-image-sft --timeout 180

# 5. Launch Gradio app with public HTTPS share link (first run downloads 27GB weights)
colab exec -s moss-image-sft -f app_gradio.py --timeout 600
```

The script will output:
```text
GRADIO_PUBLIC_URL=https://<unique-hash>.gradio.live
```

---

## 3. Minimal Standalone Inference Script

```python
import torch
from diffusers import DiffusionPipeline

# 1. Load pipeline (cached automatically at /root/.cache/huggingface/)
if "pipe" not in globals() or pipe is None:
    pipe = DiffusionPipeline.from_pretrained(
        "OpenMOSS-Team/MOSS-Image-SFT",
        custom_pipeline="pipeline_moss_image",
        trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
    )
    pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    pipe.to("cuda")

# 2. Generate (CRITICAL: negative_prompt must be "" string, never None)
image = pipe(
    prompt="A cute fluffy red panda in a mossy forest, soft cinematic sunlight, 8k",
    negative_prompt="",
    height=1024,
    width=1024,
    num_inference_steps=25,   # 25 steps is 2x faster than 50 with identical quality
    guidance_scale=3.0,
    uncond_skip_layers=[22],  # Accelerates CFG negative branch
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]

image.save("/content/output.png")
```

---

## 4. Critical Gotchas & Troubleshooting

| Issue / Error | Cause | Solution |
|---|---|---|
| **`TypeError: object of type 'NoneType' has no len()`** | OpenMOSS's `pipeline_moss_image.py` evaluates `len(negatives)` inside CFG. If `negative_prompt=None`, `negatives` evaluates to `None` and crashes. | **Always pass `negative_prompt=""` (empty string)**, never `None`. |
| **`cannot import name 'FqnToConfig' from 'torchao'`** | Colab's default `torchao==0.10.0` has a breaking API mismatch with `diffusers>=0.38`. | Upgrade torchao: `pip install -U torchao` (resolves to `torchao>=0.18`). |
| **`Timeout waiting for reply` on first run** | Downloading 27.2 GB of safetensors weights exceeds the default 30s timeout. | Run first execution with `--timeout 600`. Subsequent runs take <1s to load from disk cache. |
| **Gradio hangs or blocks CLI** | Default `demo.launch()` blocks the Python thread. | Use `demo.queue().launch(share=True, prevent_thread_lock=True)` so execution returns immediately with the public URL. |
| **Slow inference (~38s)** | Default 50 steps with full dual CFG passes. | Set `num_inference_steps=25` (drops to **19s**) and keep `uncond_skip_layers=[22]`. |

---

## 5. Teardown & Cost Management

Always shut down the session when finished to stop compute billing:

```bash
# Check status and active assignments
colab usage
colab sessions

# Terminate session
colab stop -s moss-image-sft
```
