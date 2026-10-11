#!/usr/bin/env bash
# One-shot setup of Pointcept + Sonata for an NVIDIA RTX 2080 (Turing, sm_75, 8 GB).
#
#   bash envs/setup_2080.sh [install_dir]      (default: ../Pointcept next to this repo)
#
# Needs: Linux (or WSL2), NVIDIA driver >= 550, conda/mamba, git, curl.
# Creates conda env "pointcept-2080" (Python 3.10, PyTorch 2.5.0, CUDA 12.4),
# skips FlashAttention (unsupported on Turing), builds pointops for sm_75,
# downloads the Sonata pretrained weights and links this repo's configs/data.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$REPO/../Pointcept}"
POINTCEPT_COMMIT=1342eda30e96cbb5fadf5374ff8eb18f6de15c71   # tested config layout
ENV=pointcept-2080
export TORCH_CUDA_ARCH_LIST="7.5"   # RTX 20xx; add e.g. "7.5 8.6" to also build for an RTX 30xx

if [ ! -d "$DEST" ]; then
  git clone https://github.com/Pointcept/Pointcept.git "$DEST"
fi
cd "$DEST"
git fetch -q origin "$POINTCEPT_COMMIT" 2>/dev/null || true
git checkout -q "$POINTCEPT_COMMIT"

# Pointcept's environment.yml minus packages that don't build/aren't needed on Turing.
sed -e "s/^name: .*/name: $ENV/" \
    -e '/flash-attention/d' \
    -e '/openai\/CLIP/d' \
    -e '/ocnn-pytorch/d' \
    environment.yml > environment-2080.yml

unset CUDA_PATH || true
CONDA=$(command -v mamba || command -v conda)
if ! $CONDA env list | grep -q "^$ENV "; then
  $CONDA env create -f environment-2080.yml
fi

# Script deps for this repo's converters
$CONDA run -n $ENV pip install -q trimesh "laspy[lazrs]" huggingface_hub

# Sonata pretrained weights (CC-BY-NC 4.0, research use)
mkdir -p weights/sonata
if [ ! -f weights/sonata/pretrain-sonata-v1m1-0-base.pth ]; then
  curl -L -o weights/sonata/pretrain-sonata-v1m1-0-base.pth \
    https://huggingface.co/facebook/sonata/resolve/main/pretrain-sonata-v1m1-0-base.pth
fi

# Link this repo's configs and data into Pointcept
ln -sfn "$REPO/pointcept_configs/heritage" configs/heritage
mkdir -p "$REPO/data/heritage"/{train,val,test}
mkdir -p data && ln -sfn "$REPO/data/heritage" data/heritage

echo
echo "== GPU smoke test =="
$CONDA run -n $ENV --no-capture-output python "$REPO/scripts/gpu_smoke_test.py" \
  --config configs/heritage/semseg-sonata-heritage-dec.py \
  --weight weights/sonata/pretrain-sonata-v1m1-0-base.pth --points 80000

echo
echo "Done. Next:  conda activate $ENV && cd $DEST   (then follow docs/PLAN.md)"
