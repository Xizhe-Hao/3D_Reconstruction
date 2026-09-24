#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mvtracker_dir="${project_dir}/submodule/mvtracker"
session_dir="${SESSION_DIR:-${project_dir}/data/test}"
output_dir="${OUTPUT_DIR:-${project_dir}/outputs/data_test}"
start_frame="${START_FRAME:-0}"
end_frame="${END_FRAME:-1414}"
target_frames="${TARGET_FRAMES:-96}"
ga_niter="${DUSTER_GA_NITER:-300}"
confidence="${DUSTER_CONFIDENCE:-20}"
query_grid="${QUERY_GRID_SIZE:-24}"
query_voxel_size="${QUERY_VOXEL_SIZE_M:-0.002}"
query_roi="${QUERY_ROI:-0.05,0.05,0.95,0.95}"
world_radius="${WORLD_RADIUS_M:-0.25}"
point_stride="${POINTCLOUD_PIXEL_STRIDE:-2}"

conda run --no-capture-output -n mvtracker python "${project_dir}/scripts/run_test_session.py" \
  --session-dir "${session_dir}" \
  --start "${start_frame}" \
  --end "${end_frame}" \
  --target-frames "${target_frames}" \
  --max-frames "${target_frames}" \
  --depth-backend duster \
  --duster-image-size 512 \
  --duster-ga-niter "${ga_niter}" \
  --duster-conf-threshold "${confidence}" \
  --query-views 0,1,2,3 \
  --query-grid-size "${query_grid}" \
  --query-voxel-size-m "${query_voxel_size}" \
  --roi "${query_roi}" \
  --world-radius-m "${world_radius}" \
  --pointcloud-pixel-stride "${point_stride}" \
  --pointcloud-radius-m 0.5 \
  --rerun-pointcloud-mode fused \
  --output-dir "${output_dir}" \
  "$@"
