#!/bin/bash
# One-time GPU environment setup for the SeedVR2 pipeline (CUDA path, incl. Blackwell sm_120).
# For AMD ROCm see README "Path B" instead (different torch wheel + env vars).
#
# Run:
#   export SEEDVR2_NODE=$PWD/ComfyUI-SeedVR2_VideoUpscaler
#   bash setup_gpu.sh
set -e

echo "=== system deps (ffmpeg) ==="
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg)

echo "=== torch / GPU smoke test ==="
NEED=$(python - <<'PY'
try:
    import torch
    ok = torch.cuda.is_available()
    if ok:
        a = torch.randn(8,8,device="cuda"); (a@a).sum().item()   # exercise kernels (catches sm mismatch)
        print("ok")
    else:
        print("install")
except Exception:
    print("install")
PY
)
if [ "$NEED" != "ok" ]; then
  echo "=== installing CUDA 12.8 torch (Blackwell-capable) ==="
  pip install --upgrade pip
  pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
fi

echo "=== SeedVR2 node ==="
: "${SEEDVR2_NODE:?set SEEDVR2_NODE to where the node should live}"
if [ ! -d "$SEEDVR2_NODE" ]; then
  git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "$SEEDVR2_NODE"
fi
( cd "$SEEDVR2_NODE" && git checkout v2.5.23 2>/dev/null || true )
pip install -q safetensors tqdm psutil einops "omegaconf>=2.3.0" "diffusers>=0.33.1" \
  "peft>=0.17.0" "rotary_embedding_torch>=0.5.3" opencv-python-headless gguf matplotlib

echo "=== detect deps (optional, for detect_shots.py) ==="
pip install -q "scenedetect[opencv]" pillow || true

echo "=== final GPU check ==="
python - <<'PY'
import torch, torch.nn.functional as F
a = torch.randn(1024,1024,device="cuda"); print("matmul ok", float((a@a).sum()))
q = torch.randn(1,8,512,64,device="cuda"); print("sdpa ok", tuple(F.scaled_dot_product_attention(q,q,q).shape))
PY
echo "SETUP DONE. Models auto-download on first run (~3.6 GB)."
