#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_env.sh"
cd "$PHYS_RELEASE_DIR"
python -m eval_abalation.run_param_replay \
  --phys_dir "$PHYS_RELEASE_DIR/simulation" \
  --dataset_root "${DATASET_ROOT:-auto_output/dataset_5000/train}" \
  --records "${RECORDS:-outputs/param_eval/logic/param_metrics/sample_records.jsonl}" \
  --out_dir "${OUT_DIR:-outputs/param_ambiguity/logic}" \
  --num_samples "${REPLAY_NUM_SAMPLES:-0}" \
  --sample_mode "${REPLAY_SAMPLE_MODE:-first}" \
  --seed "${REPLAY_SEED:-0}" \
  --num_gpus "${REPLAY_NUM_GPUS:-1}"
