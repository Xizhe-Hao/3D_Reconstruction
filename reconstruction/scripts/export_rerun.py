"""Rebuild ``tracks_4d.rrd`` for a finished run_test_session result.

Re-decodes the same RGB frames, reuses the saved tracks and depth cache, and
writes the recording with the synchronized RGB / predicted-depth / D455 views
(see scripts/rerun_views.py). No depth model or MVTracker inference is rerun.

    python scripts/export_rerun.py --result-dir outputs/20261009/seconds_0_end_target_368_da3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_test_session import export_rerun
from scripts.test_session_adapter import load_session_clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, help="Default: session_dir recorded in run.json")
    parser.add_argument("--output", type=Path, help="Default: <result-dir>/tracks_4d.rrd")
    parser.add_argument("--view-depth-downsample", type=int, default=2,
                        help="Pixel stride for the depth image panels (1 = full resolution, larger file)")
    parser.add_argument("--pointcloud-point-radius-m", type=float, default=0.001)
    parser.add_argument("--no-dense-rerun", action="store_true")
    parser.add_argument("--no-rerun-views", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    result_dir = args.result_dir.expanduser().resolve()
    run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    session_dir = (args.session_dir or Path(run["session_dir"])).expanduser().resolve()
    frames = np.asarray(run["frames"], dtype=np.int32)

    depth_cache = result_dir / f"depths_{run['depth_backend']}_m.npz"
    with np.load(depth_cache) as data:
        depths_m = data["depths_m"].astype(np.float32)
    height, width = depths_m.shape[-2:]

    step = int(frames[1] - frames[0]) if len(frames) > 1 else 1
    clip = load_session_clip(
        session_dir, int(frames[0]), int(frames[-1]), step if run.get("target_frames") is None else 1,
        (width, height), len(frames), run.get("target_frames"), not args.no_progress,
    )
    if not np.array_equal(clip.frame_indices, frames):
        raise SystemExit("Re-decoded frame indices differ from run.json; pass the original --session-dir")

    with np.load(result_dir / "tracks_4d.npz") as data:
        tracks_m = data["trajectories_m"]
        visibility = data["visibility"].astype(bool)

    dense = run.get("dense_pointcloud") or {}
    output = (args.output or result_dir / "tracks_4d.rrd").expanduser().resolve()
    export_rerun(
        output, tracks_m, visibility, clip,
        depths_m=None if args.no_dense_rerun else depths_m,
        pointcloud_pixel_stride=int(dense.get("pixel_stride", 4)),
        pointcloud_radius_m=dense.get("radius_m", 0.5),
        pointcloud_point_radius_m=args.pointcloud_point_radius_m,
        pointcloud_mode=dense.get("rerun_mode", "fused"),
        show_progress=not args.no_progress,
        session_dir=session_dir,
        log_views=not args.no_rerun_views,
        view_depth_downsample=args.view_depth_downsample,
    )
    print(f"Wrote {output} ({output.stat().st_size / 2**20:.0f} MiB)", flush=True)


if __name__ == "__main__":
    main()
