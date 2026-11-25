# Vāgdhenu on AMD (ROCm / WSL2) — Setup Guide

This fork adds AMD GPU support via ROCm, letting you run Vāgdhenu on an AMD
RDNA3 GPU (RX 7900 XTX / XT / 7800 XT / 7700 XT) through WSL2 on Windows 11.
The original repo only ships CUDA (NVIDIA) and MLX (Apple Silicon) paths.

**Target hardware for this fork:**
- AMD RX 7900 XTX (24 GB) — gfx1100 — the sweet spot
- Any RDNA3 card (change `HSA_OVERRIDE_GFX_VERSION` for gfx1101, gfx1102)
- Windows 11 host with WSL2 Ubuntu 22.04 or 24.04

## Quick start

From a WSL2 Ubuntu shell, inside this repo:

```bash
bash scripts/setup_amd.sh
# (installs Python 3.10, ROCm PyTorch, BigVGAN, weights)

# Then render a sample:
source .venv/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"
python src/render.py --shard examples/sample_shard.json \
                     --results /tmp/res.json --outdir out
# -> out/sample_anushtubh.wav
```

## Why this is needed

Vāgdhenu's upstream `scripts/setup.sh` hard-codes CUDA 12.1 PyTorch wheels and
assumes an NVIDIA GPU. On AMD cards this fails immediately. This fork:

1. Swaps the torch install to the **ROCm 6.2** wheels of the same `torch==2.4.1`
2. Installs **Python 3.10** on Ubuntu 24.04 (which ships 3.12) via deadsnakes
3. Adds env vars (`HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_ROCM_ARCH`) that pin
   PyTorch to the right RDNA3 ISA (gfx1100)
4. Runs a GPU smoke test before downloading 2+ GB of weights so you know it
   actually works

## Prerequisites on the Windows host

1. **Latest AMD Adrenalin driver** (25.9.2+ WHQL recommended — do NOT jump to
   26.x branch without reason; mixed-branch driver stacks cause PCIe/Oculink
   instability on multi-GPU systems)
2. **WSL2 with GPU passthrough** — check inside WSL: `ls /dev/dxg` should
   exist. If not: `wsl --update` from PowerShell
3. **Ubuntu 22.04 or 24.04** distro installed in WSL

## Prerequisites inside WSL

The `setup_amd.sh` script installs these for you:
- Python 3.10 (via deadsnakes PPA)
- git, curl, build-essential, cmake, libsndfile1, ffmpeg
- torch 2.4.1 + torchaudio 2.4.1 (ROCm 6.2 wheels)
- All Vāgdhenu Python requirements
- NVIDIA BigVGAN (cloned, torch path only — no CUDA compilation)

## Supported AMD GPUs

| Card | gfx arch | `HSA_OVERRIDE_GFX_VERSION` |
|------|----------|----------------------------|
| RX 7900 XTX / XT | gfx1100 | `11.0.0` (default) |
| RX 7800 XT       | gfx1101 | `11.0.1` |
| RX 7700 XT       | gfx1101 | `11.0.1` |
| RX 7600 XT       | gfx1102 | `11.0.2` |

For cards not officially in ROCm's support matrix, the `HSA_OVERRIDE_GFX_VERSION`
trick tells PyTorch to compile for the nearest supported arch. It works for
RDNA3 variants but YMMV for RDNA2.

## Expected performance

RX 7900 XTX (24 GB VRAM):
- Loads the full model in bfloat16 with room to spare
- One anuṣṭubh verse (~9-10 s audio) renders in ~2-4 seconds
- RTF (real-time factor) around 0.3 — faster than real-time

The model is only ~337M parameters + vocoder — small for 24 GB. You can batch
multiple verses on the same GPU without worry.

## Troubleshooting

### "HIP error: no CUDA-capable device"
PyTorch sees the GPU through HIP (AMD's CUDA shim). Check:
```bash
ls /dev/dxg                 # must exist
python -c "import torch; print(torch.cuda.is_available())"
```
If `/dev/dxg` is missing, update WSL: `wsl --update` from Windows PowerShell.

### "Unsupported architecture"
Your card's gfx arch isn't in PyTorch's compiled list. Set:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # try 11.0.0, 11.0.1, 11.0.2
```

### Slow first inference
ROCm compiles kernels on first use (like CUDA JIT). First verse may take
30-60 seconds longer. Subsequent renders are fast.

### Vocoder errors ("cuda_kernel compilation failed")
Ignored — Vāgdhenu uses BigVGAN's torch path (`use_cuda_kernel=False`), so no
custom CUDA kernels are compiled. If you see compilation warnings, they're
cosmetic.

### WSL out of memory
WSL2 has a default memory cap. Edit `C:\Users\<you>\.wslconfig`:
```ini
[wsl2]
memory=24GB
swap=0
```
Then `wsl --shutdown` and restart WSL.

## Attribution

This is a fork of [prathoshap/vagdhenu](https://github.com/prathoshap/vagdhenu)
by Prof. Prathosh at IISc Bengaluru. All credit for the model, training, and
upstream code goes to the original authors. This fork only adds an AMD/ROCm
path to the setup — it changes no model code.

License unchanged: Apache-2.0.
