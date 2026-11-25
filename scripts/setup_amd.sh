#!/usr/bin/env bash
set -e
# ============================================================
# Vāgdhenu AMD/ROCm setup — WSL2 Ubuntu 22.04/24.04 on Windows 11
# ============================================================
# Fork of the original setup.sh, patched for AMD RDNA3 GPUs
# (RX 7900 XTX / XT / 7800 XT / 7700 XT) via ROCm 6.2+.
#
# Requirements before running:
#   - Windows 11 with WSL2 + Ubuntu 22.04 or 24.04
#   - AMD RDNA3 GPU (gfx1100 for 7900 XTX/XT)
#   - Latest AMD Adrenalin driver on the Windows host
#   - WSL kernel 5.15+ with GPU passthrough (check: ls /dev/dxg)
#   - Python 3.10 (Ubuntu 24.04 ships 3.12 — we install 3.10 via deadsnakes)
#
# Usage:
#   bash scripts/setup_amd.sh
#
# The AMD RDNA3 architecture is gfx1100. Set this before running if you have
# a different card:
#   export HSA_OVERRIDE_GFX_VERSION=11.0.0   # RX 7900 XTX / XT  (default)
# ============================================================

echo "============================================================"
echo "Vāgdhenu AMD/ROCm setup starting..."
echo "============================================================"

# --- 1. Check WSL + GPU passthrough ---
if ! grep -qi microsoft /proc/version; then
  echo "WARNING: not running inside WSL — ROCm paths may differ on bare Linux."
fi
if [ ! -e /dev/dxg ]; then
  echo "ERROR: /dev/dxg not found. GPU passthrough is not available."
  echo "  - Install the latest AMD Adrenalin driver on Windows."
  echo "  - Ensure WSL kernel >= 5.15: wsl --update"
  exit 1
fi
echo "✓ /dev/dxg present (GPU passthrough enabled)"

# --- 2. Python 3.10 (deadsnakes PPA) ---
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "Installing Python 3.10..."
  sudo apt-get update -y
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -y
  sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi
echo "✓ Python 3.10 installed: $(python3.10 --version)"

# --- 3. System build deps ---
sudo apt-get install -y git curl build-essential cmake libsndfile1 ffmpeg

# --- 4. Create a Python 3.10 venv for Vāgdhenu ---
if [ ! -d .venv ]; then
  python3.10 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

# --- 5. Install PyTorch with ROCm wheels ---
# ROCm 6.2 wheels support RDNA3 (gfx1100) natively. torch 2.4.1 matches the
# original repo's pin, and the ROCm 6.2 wheels of that version are published.
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.2

# --- 6. Install Vāgdhenu requirements ---
pip install -r requirements.txt

# --- 7. Clone NVIDIA BigVGAN (torch path, no CUDA kernels compiled) ---
[ -d BigVGAN/.git ] || git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git BigVGAN
export PYTHONPATH="$PWD/BigVGAN:$PYTHONPATH"

# --- 8. Set environment variables for RDNA3 on ROCm ---
# RX 7900 XTX is gfx1100 — officially supported since ROCm 5.7, polished in 6.2+.
# If you have a different card, override before running.
if ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc 2>/dev/null; then
  {
    echo ""
    echo "# Vāgdhenu AMD/ROCm environment"
    echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0  # RX 7900 XTX/XT (gfx1100)"
    echo "export PYTORCH_ROCM_ARCH=gfx1100"
    echo "export PYTHONPATH=\"$PWD/BigVGAN:\$PYTHONPATH\""
  } >> ~/.bashrc
  echo "✓ Added ROCm env vars to ~/.bashrc"
fi
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100

# --- 9. Smoke-test: can PyTorch see the GPU? ---
echo ""
echo "============================================================"
echo "GPU smoke test:"
python -c "
import torch
print(f'torch:       {torch.__version__}')
print(f'ROCm/HIP:    {getattr(torch.version, \"hip\", None)}')
print(f'CUDA avail:  {torch.cuda.is_available()}   # True = GPU usable')
print(f'Device count:{torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  [{i}] {torch.cuda.get_device_name(i)}')
    x = torch.randn(1024, 1024, device='cuda')
    y = (x @ x).sum().item()
    print(f'Matmul OK, y={y:.2f}')
else:
    print('WARNING: GPU not visible. Check /dev/dxg and HSA_OVERRIDE_GFX_VERSION.')
"
echo "============================================================"

# --- 10. Download Vāgdhenu weights ---
python scripts/download_weights.py

echo ""
echo "✓ AMD/ROCm setup complete."
echo ""
echo "To use in a new shell:"
echo "  source .venv/bin/activate"
echo "  export HSA_OVERRIDE_GFX_VERSION=11.0.0"
echo "  export PYTHONPATH=\"\$PWD/BigVGAN:\$PYTHONPATH\""
echo ""
echo "Render a sample verse:"
echo "  python src/render.py --shard examples/sample_shard.json --results /tmp/res.json --outdir out"
