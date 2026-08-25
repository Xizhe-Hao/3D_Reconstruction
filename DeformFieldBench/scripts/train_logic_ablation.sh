#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_env.sh"
cd "$PHYS_RELEASE_DIR"
CONFIG="${CONFIG:-configs/logic_model/final_logic_413_5000_new.json}"
NUM_GPUS="${NUM_GPUS:-8}"
python -m torch.distributed.run --nproc_per_node="$NUM_GPUS" -m logic_model.train --config "$CONFIG"
