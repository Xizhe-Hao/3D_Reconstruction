#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_env.sh"
cd "$PHYS_RELEASE_DIR"
MODEL="${MODEL:-logic}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-outputs/param_eval/$MODEL}"
mkdir -p "$OUT_DIR"
case "$MODEL" in
  my_model)
    python -m eval_abalation.eval_my_model \
      --config configs/my_model/train_dataset_5000_param_only.json \
      --weights "${WEIGHTS:-pretrained/supervised_baseline_last.pt}" \
      --eval_split "$EVAL_SPLIT" --out_dir "$OUT_DIR" --num_samples "$NUM_SAMPLES"
    ;;
  logic)
    python -m eval_abalation.eval_logic \
      --config configs/logic_model/final_logic_413_5000_new.json \
      --checkpoint "${CHECKPOINT:-pretrained/logic_baseline_epoch_0320.pt}" \
      --eval_split "$EVAL_SPLIT" --out_dir "$OUT_DIR" --num_samples "$NUM_SAMPLES"
    ;;
  *)
    echo "MODEL must be my_model or logic" >&2
    exit 1
    ;;
esac
