#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_env.sh"
cd "$PHYS_RELEASE_DIR"
NUM_GPUS="${NUM_GPUS:-8}"
python -m torch.distributed.run --nproc_per_node="$NUM_GPUS" -m my_model.train \
  --config configs/my_model/train_dataset_5000_param_only.json \
  --disable_aux_losses
