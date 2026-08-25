#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_env.sh"
cd "$PHYS_RELEASE_DIR/simulation"
python modified_simulation.py \
  --ply_path "${PLY_PATH:?set PLY_PATH}" \
  --config "${CONFIG:-../configs/simulation/train_config_dataset_full.json}" \
  --output_path "${OUTPUT_PATH:-../outputs/simulation_sample}" \
  --sim_type "${SIM_TYPE:-press}" \
  --render_img \
  --output_view_stress_gaussian \
  --output_view_flow_gaussian \
  --output_view_force_mask \
  --force_mask_single_channel \
  --output_view_object_mask \
  --pack_sample_pack \
  --pack_sample_pack_include_object_mask \
  --arch4_lmdb_resize "${ARCH4_LMDB_RESIZE:-224}"
