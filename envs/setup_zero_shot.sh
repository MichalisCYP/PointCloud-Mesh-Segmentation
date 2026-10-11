#!/usr/bin/env bash
# Setup for the no-training pipeline (SAM 3 text prompts) on an NVIDIA GPU, e.g. RTX 2080 8 GB.
#
#   bash envs/setup_zero_shot.sh
#
# Needs: Linux or WSL2, NVIDIA driver >= 570 (CUDA 12.8 wheels), conda/mamba, git.
# BEFORE running: request access to the SAM 3 weights at https://huggingface.co/facebook/sam3
# and log in once with:  hf auth login   (create a read token on huggingface.co)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SAM3_DIR="${SAM3_DIR:-$REPO/../sam3}"
ENV=heritage-sam3
CONDA=$(command -v mamba || command -v conda)

if ! $CONDA env list | grep -q "^$ENV "; then
  $CONDA create -y -n $ENV python=3.12
fi
run() { $CONDA run -n $ENV --no-capture-output "$@"; }

run pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
[ -d "$SAM3_DIR" ] || git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
run pip install -e "$SAM3_DIR"
run pip install numpy scipy plyfile trimesh "laspy[lazrs]" pillow huggingface_hub

echo
echo "== SAM 3 smoke test (downloads the weights on first run) =="
run python - <<'PY'
import torch, numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
print("GPU:", torch.cuda.get_device_name(0))
proc = Sam3Processor(build_sam3_image_model())
state = proc.set_image(Image.fromarray(np.full((1008, 1008, 3), 128, np.uint8)))
proc.set_text_prompt(prompt="column", state=state)
print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB of "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
print("OK")
PY
echo
echo "Done. Now run:  ./segment.sh <file or folder>"
