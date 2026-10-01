# MVTracker adapter for `data/test`

MVTracker produces sparse 3D point trajectories over time. It does not produce
a watertight deforming mesh. All exported XYZ values are in the calibrated
world coordinate system in metres.

## Setup

```bash
cd /root/autodl-tmp/nd_project/submodule/mvtracker
bash scripts/setup_mvtracker.sh
bash scripts/setup_duster.sh
conda activate mvtracker
```

## Run with calibrated multi-view DUSt3R depth

The adapter calls MVTracker's official `estimate_depth_with_duster.py` helper. For each
sampled timestamp it matches all pairs among the four synchronized cameras, fixes the
calibrated intrinsics and camera poses, and globally aligns one metric scene. This is
the recommended backend for avoiding four independently scaled MoGe point clouds.

Start with a short smoke test because the 512 model is compute intensive:

```bash
python scripts/run_test_session.py \
  --session-dir ../../data/test \
  --start 660 --end 666 --target-frames 7 \
  --depth-backend duster \
  --duster-ga-niter 50 \
  --query-views 0,1,2,3 --query-grid-size 16
```

For the final run, omit `--duster-ga-niter` to use the official 300-iteration
default. Per-timestamp DUSt3R scenes are resumable under `duster/`; consolidated
metric depths are cached as `depths_duster_m.npz`. A full 96-timestamp run is much
slower than MoGe because it performs multi-view matching and alignment 96 times.
For a denser but noisier reconstruction, try `--duster-conf-threshold 3`; add
`--duster-no-clean-depth` only when you explicitly want to retain geometric outliers.

## Validate and prepare a clip

```bash
python scripts/test_session_adapter.py \
  --session-dir ../../data/test --start 0 --end 3 \
  --output outputs/test_session_clip.npz
```

## Run with automatic MoGe-2 depth

Select at least 7 sampled frames; 12--24 frames are recommended for a first run.
Queries are restricted to 0.18 m around the calibrated world origin by default.
Start with a short clip. The first run downloads the MoGe-2 weights.

```bash
python scripts/run_test_session.py \
  --session-dir ../../data/test \
  --start 0 --end 23 --step 1 \
  --depth-backend moge2 \
  --query-view 0 --query-grid-size 16 \
  --roi 0.2,0.2,0.8,0.8
```

The outputs are written under `outputs/data_test/frames_*`:

- `tracks_4d.npz`: dense arrays for downstream Python processing;
- `tracks_4d.csv`: `(track_id, frame, time, x, y, z, visible)` long table;
- `tracks_4d.rrd`: Rerun trajectory playback;
- `ply/`: visible tracked points at every selected timestamp.

For meaningful metric trajectories, replace monocular MoGe-2 depths with
calibrated depth maps:

```bash
python scripts/run_test_session.py \
  --session-dir ../../data/test --start 0 --end 23 \
  --depth-backend npz --depth-path /path/to/depths.npz --depth-unit m
```

The NPZ must contain `depths_m` or `depths` shaped `[V,T,H,W]` (or
`[V,T,1,H,W]`). It may include `frame_indices` when it contains a full-session
cache. Depth means camera-space Z, not Euclidean ray distance.

## Track the full duration with a target frame count

MVTracker internally uses a 12-frame rolling window. The adapter's default
96-frame ceiling is a memory safety guard, not a hard limit of the model. To
cover all 1415 source frames using exactly 96 synchronized timestamps (including
frames 0 and 1414), run:

```bash
python scripts/run_test_session.py \
  --session-dir ../../data/test \
  --start 0 --end 1414 --target-frames 96 \
  --max-frames 96 \
  --depth-backend moge2 \
  --query-view 0 --query-grid-size 16 \
  --roi 0.2,0.2,0.8,0.8
```

If `--target-frames` is supplied without `--end`, the session's final frame is
used automatically. Sampling is a rounded linear spacing, so the 1415-to-96
case alternates between 14- and 15-frame source gaps. The NPZ, CSV and Rerun
outputs retain the actual source frame index and `video_time_s` for every
sample. `--target-frames` takes precedence over `--step`.

Use a higher target only together with a matching `--max-frames`; memory and
runtime grow with the number of sampled timestamps. For fast deformation,
prefer 96 or more samples over an aggressive fixed stride.

## Progress diagnostics

Progress reporting is enabled by default. It shows four-camera decode progress,
MoGe-2 depth batches, query counts, checkpoint loading, inference CUDA memory,
and NPZ/CSV/PLY/Rerun export progress. During the predictor call a heartbeat
is printed every 15 seconds because upstream MVTracker does not expose a
per-window callback. For more frequent diagnostics use:

```bash
python scripts/run_test_session.py ... --heartbeat-seconds 2
```

Use `--no-progress` to disable stage messages and progress bars.

## Dense RGB-D point cloud plus trajectories

The adapted runner uses the same world-space RGB-D backprojection as the official demo. By default it concatenates all four views into one `world/pointcloud/fused` entity, alongside four calibrated camera frustums under `world/cameras` and current points plus history under `world/tracks`. Use `--rerun-pointcloud-mode per-view` to reproduce the official demo layout with independently toggleable `view` entities. It also writes one fused
binary PLY per timestamp to `ply_dense/`; the existing `ply/` directory remains
the sparse tracked points only.

Dense output is enabled by default. Control it with:

```bash
--pointcloud-pixel-stride 4   # 1=densest, 2=dense, 4=balanced, 8=light
--pointcloud-radius-m 0.5     # crop around calibration world origin
--pointcloud-point-radius-m 0.001
```

Use `--no-dense-ply` or `--no-dense-rerun` when storage is limited. Dense RGB-D
points do not carry persistent track IDs; only `world/tracks` and `tracks_4d.*`
represent temporal correspondences.

## Multi-view query initialization

Queries now default to all cameras (`--query-views 0,1,2,3`). Each view samples
its first-frame ROI, backprojects valid depths to the calibration world frame,
and the combined candidates are deduplicated with
`--query-voxel-size-m 0.005`. The legacy `--query-view 0` option overrides this
and reproduces single-view surface initialization. Multi-view queries cover the
union of surfaces visible from the selected cameras, but require geometrically
consistent depth; they cannot repair per-view monocular depth offsets.
