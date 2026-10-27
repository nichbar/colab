# Colab Model Deployment Toolkit

A collection of operational playbooks, scripts, and guidelines for provisioning, running, and serving machine learning models on remote Google Colab runtimes using the `colab` CLI.

## Overview

This repository provides reproducible workflows for both AI coding agents and developers to manage headless Google Colab sessions, execute remote inference, and serve models via Gradio or API tunnels without interactive notebook friction.

Included is a complete, tested deployment guide and Gradio application for **[MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)** (a 27GB bilingual text-to-image diffusion model) running on Colab Pro A100 GPUs.

---

## Repository Contents

| File | Description |
|---|---|
| [`AGENTS.md`](./AGENTS.md) | Standard operational guidelines, failure mode mitigations, and best practices for automated/agentic Colab CLI workflows. |
| [`moss-image-sft.md`](./moss-image-sft.md) | Playbook for deploying `OpenMOSS-Team/MOSS-Image-SFT` on an NVIDIA A100 GPU instance. |
| [`app_gradio.py`](./app_gradio.py) | Interactive Gradio Web UI with bilingual prompt examples and public HTTPS share link. |
| [`test_moss.py`](./test_moss.py) | Minimal standalone Python script for headless test generation. |

---

## Quick Start: Deploying MOSS-Image-SFT

### 1. Prerequisites
Ensure you have the `colab` CLI installed and authenticated:
```bash
colab whoami
colab usage
```

### 2. Provision and Run
Launch an A100 instance, install requirements, and start the Gradio app:

```bash
# 1. Provision an A100 instance
colab new -s moss-image-sft --gpu A100

# 2. Install dependencies & patch torchao compatibility
colab install -s moss-image-sft "diffusers>=0.38" "transformers>=5.3" "accelerate>=1.10" "safetensors>=0.6" "Pillow>=10" gradio
echo "!pip install -U torchao" | colab exec -s moss-image-sft

# 3. Launch the Gradio Web UI (public link generated automatically)
colab exec -s moss-image-sft -f app_gradio.py --timeout 600
```

### 3. Cleanup
When finished, stop the session to release compute units:
```bash
colab stop -s moss-image-sft
```

---

## General Workflow

For deploying any model using this toolkit:

```
[1. Pre-flight] ──> [2. Provision] ──> [3. Install] ──> [4. Author Script] ──> [5. Execute] ──> [6. Expose/Serve] ──> [7. Teardown]
```

Refer to [AGENTS.md](./AGENTS.md) for detailed instructions on session lifecycle, memory caching, environment configuration, and handling headless execution timeouts.
