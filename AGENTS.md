# AGENTS.md: Deploying Models via Google Colab CLI

This operational guide instructs AI coding agents on how to provision, configure, deploy, and manage machine learning models on remote Google Colab runtimes using the `colab` CLI.

---

## 1. Mental Model for Autonomous Agents

- **Session = Live Jupyter Kernel**: `colab new` provisions a billable remote VM running an `ipykernel` process.
- **Kernel State Persists**: Code executed via `colab exec -s <name>` runs inside the **same continuous Python kernel**. Variables, imported modules, and models loaded into GPU memory persist across distinct `colab exec` calls until the kernel is reset (`colab restart-kernel`) or the session is terminated (`colab stop`).
- **Remote Filesystem**: The default working directory is `/content`. Model caches live at `/root/.cache/huggingface/` (or `/content/drive/MyDrive/.cache/huggingface/` if Google Drive is mounted).
- **Headless Execution**: `colab` commands are fire-and-forget CLI calls. **Never invoke interactive TTY commands** (`colab repl`, `colab console`, `colab auth`, `colab drivemount`) without piped input or non-interactive flags—they will hang the agent waiting for stdin.

---

## 2. Standard Model Deployment Workflow

Follow this 7-step sequence when asked to deploy or evaluate any model:

```
[1. Pre-flight] ──> [2. Provision] ──> [3. Install] ──> [4. Author Script] ──> [5. Execute] ──> [6. Expose/Serve] ──> [7. Teardown]
```

> **Model-Specific Playbooks:**
> - [MOSS-Image-SFT (`OpenMOSS-Team/MOSS-Image-SFT`)](./moss-image-sft.md): A100 (40GB) fast setup, dependency fixes, custom pipeline caveats, and Gradio share configuration.


### Step 1: Pre-Flight Check
Before spinning up resources, inspect active sessions, compute balance, and auth validity:
```bash
colab whoami      # Verify OAuth2/ADC token validity and scopes
colab usage       # Check compute unit balance and burn rate
colab sessions    # Check for existing running sessions to avoid duplicate billing
```

### Step 2: Provision Accelerator
Always specify an explicit session name `-s <name>` and target accelerator:
```bash
# Supported GPUs: T4 (15GB), L4 (24GB), A100 (40/80GB)
# Supported TPUs: v5e1, v6e1
colab new -s <session_name> --gpu T4
```
*Note*: Free accounts generally have access to T4 GPUs. An invalid GPU flag defaults to `A100`, which will fail with HTTP 400 if the account lacks quota.

### Step 3: Install Remote Dependencies
Install Python packages on the VM using `colab install` (which uses `uv` by default for fast installation):
```bash
colab install -s <session_name> "transformers>=5.20.0" torch torchvision accelerate gradio
```

### Step 4: Author Idempotent Model Scripts
When writing Python scripts to be executed on Colab, adhere strictly to these rules:

1. **Kernel Cache Guard**: Prevent reloading models into GPU memory if already present:
   ```python
   if "model" not in globals() or model is None:
       processor = AutoProcessor.from_pretrained(MODEL_ID)
       model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, device_map="auto")
   ```
2. **Read Config via Environment Variables, NOT `sys.argv`**:
   The Colab ipykernel passes Jupyter runtime flags in `sys.argv` (e.g. `-f /root/.../kernel-xxx.json`). Always read inputs from `os.environ`:
   ```python
   prompt = os.environ.get("PROMPT", "Default prompt")
   ```

### Step 5: Execute with Explicit Timeouts
The default timeout for `colab exec` is **30.0 seconds**. Downloading weights and compiling model graphs requires higher thresholds:
```bash
# Model download & initial execution (use 300s to 600s)
colab exec -s <session_name> -f script.py --env PROMPT="Your prompt" --timeout 300

# Subsequent cached inference runs
colab exec -s <session_name> -f script.py --env PROMPT="Second prompt" --timeout 120
```

### Step 6: Exposing Web Interfaces or APIs
Colab VMs are located in a private Google network without inbound public IPs. Direct `localhost:8000` ports are not reachable from outside the VM.
- **For Web UIs (Gradio)**: Always specify `share=True` to create a public HTTPS tunnel:
  ```python
  demo.launch(share=True, inline=False)
  ```
  Run with indefinite timeout:
  ```bash
  colab exec -s <session_name> -f app_gradio.py --timeout 0
  ```
- **For REST APIs (FastAPI / vLLM)**: Use tunneling tools like `cloudflared` or `ngrok` inside the session to expose the local port.

### Step 7: Teardown & Cleanup
**Never leave compute sessions running unattended.** Always terminate when finished:
```bash
colab stop -s <session_name>
```

---

## 3. Critical Failure Modes & Workarounds

| Failure Mode | Root Cause | Agent Action / Fix |
|---|---|---|
| **TimeoutError: Timeout waiting for reply** | `colab exec` hit the default 30-second timeout while downloading or generating. | Add `--timeout 300` or `--timeout 600` to the `colab exec` command. |
| **`RuntimeError: Tensor on device meta is not on expected device cuda:0`** | Loading a model twice in the same session exhausted VRAM, forcing `accelerate` to offload layers to meta/CPU. | 1. Use the `if "model" not in globals()` guard.<br>2. Reset memory: `colab restart-kernel -s <session_name>`. |
| **Script reads `-f /root/.../kernel.json` as prompt** | Script inspected `sys.argv[1]` which contains the Jupyter kernel connection config. | Read arguments via `os.environ.get()` and pass with `--env KEY=VALUE`. |
| **Agent hangs indefinitely** | Agent ran `colab repl`, `colab console`, `colab auth`, or `colab drivemount` expecting TTY. | Kill the process. Pipe batch commands via `echo "..." \| colab exec` or pass non-interactive scripts via `-f`. |
| **HTTP 400 on `colab new --gpu <type>`** | Selected accelerator is unavailable or tier-gated for the active account. | Fall back to `--gpu T4` or omit accelerator flags to run CPU. |
| **HTTP 401 / 403 on API calls** | Missing OAuth2 scopes or expired session. | Run `colab whoami` to diagnose; re-authenticate via terminal. |

---

## 4. Ephemeral One-Shot Jobs (`colab run`)

For batch scripts that should provision a VM, execute, and automatically shut down (preventing accidental billing):

```bash
colab run --gpu T4 --timeout 600 job.py arg1 arg2
```
- Provisions a fresh VM.
- Maps `arg1 arg2` directly to `sys.argv` (in `colab run`, arguments pass normally).
- Emits Colab logs to `stderr` and script output to `stdout`.
- Propagates exit codes (`sys.exit(0)` on success, non-zero on error).
- Tears down the VM automatically on exit.

---

## 5. Command Reference Cheat Sheet

```bash
# Session Lifecycle
colab new -s <name> --gpu T4    # Create session with T4 GPU
colab status -s <name>           # Check hardware & idle/busy state
colab restart-kernel -s <name>   # Flush GPU VRAM without releasing VM
colab stop -s <name>             # Release VM and stop billing

# Execution & File Transfer
colab exec -s <name> -f script.py --env KEY=VAL --timeout 180
colab upload -s <name> <local_file> <remote_path>
colab download -s <name> <remote_path> <local_file>

# Inspection & Monitoring
colab usage                      # Check compute units balance & burn rate
colab sessions                   # List all active server assignments
echo "!nvidia-smi" | colab exec -s <name>
colab url -s <name> --open       # Attach Colab web UI in browser
```

---

## 6. Model-Specific Playbooks

- **[MOSS-Image-SFT Quick Launch Guide](./moss-image-sft.md)**: End-to-end operational guide for deploying `OpenMOSS-Team/MOSS-Image-SFT` on A100 40GB, including:
  - Copy-paste bash sequence for fast provisioning.
  - Fix for the `torchao` dependency conflict.
  - Critical gotcha: `negative_prompt` must be an empty string `""`, never `None`.
  - Non-blocking Gradio configuration with `share=True`.
  - Step tuning (25 vs 50 steps) for 2x faster inference.

