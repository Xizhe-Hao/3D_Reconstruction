#!/usr/bin/env bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda deactivate || true
export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/lixiaoou/intern/linrui}"
export ENV_PATH="${ENV_PATH:-$PROJECT_ROOT/envs/train}"
conda activate "$ENV_PATH"
export MACA_PATH="${MACA_PATH:-/opt/maca-3.3.0}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"
export MACA_CLANG_PATH="$MACA_PATH/mxgpu_llvm/bin"
export CUDA_PATH="$MACA_PATH/tools/cu-bridge"
export CUCC_PATH="$MACA_PATH/tools/cu-bridge"
export PATH="$CUCC_PATH/tools:$CUCC_PATH/bin:$PATH"
