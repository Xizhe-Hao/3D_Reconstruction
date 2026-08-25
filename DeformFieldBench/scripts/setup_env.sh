#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs/lixiaoou/intern/linrui}"
export PHYS_RELEASE_DIR="${PHYS_RELEASE_DIR:-$PROJECT_ROOT/opensource/code}"
export PYTHONPATH="$PHYS_RELEASE_DIR:$PHYS_RELEASE_DIR/simulation:$PHYS_RELEASE_DIR/simulation/gaussian-splatting:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TORCH_HOME="${TORCH_HOME:-$PHYS_RELEASE_DIR/.torch}"
