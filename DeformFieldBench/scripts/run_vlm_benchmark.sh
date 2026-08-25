#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_env.sh"
cd "$PHYS_RELEASE_DIR"
python -m vlm_benchmark.run_vlm_benchmark \
  --dataset_root "${DATASET_ROOT:-auto_output/dataset_5000/train}" \
  --output_dir "${OUTPUT_DIR:-outputs/vlm_benchmark}" \
  "$@"
