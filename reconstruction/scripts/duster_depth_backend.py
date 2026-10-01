"""Calibrated DUSt3R depth backend for the data/test MVTracker adapter."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from scripts.test_session_adapter import SessionClip


def estimate_depths_with_duster(
    args,
    clip: SessionClip,
    run_dir: Path,
    report: Callable[[str, str, bool], None],
) -> np.ndarray:
    """Run MVTracker's official DUSt3R helper with fixed metric calibration."""
    duster_root = args.duster_root.expanduser().resolve()
    checkpoint = args.duster_checkpoint.expanduser().resolve()
    if not (duster_root / "dust3r").is_dir():
        raise FileNotFoundError(
            f"DUSt3R source not found at {duster_root}. Run scripts/setup_duster.sh."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"DUSt3R checkpoint not found: {checkpoint}. Run scripts/setup_duster.sh."
        )
    if str(duster_root) not in sys.path:
        sys.path.insert(0, str(duster_root))

    try:
        from scripts.estimate_depth_with_duster import run_duster
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Could not import MVTracker's official DUSt3R helper. "
            "Run scripts/setup_duster.sh in the mvtracker environment."
        ) from exc

    views, timestamps = clip.rgbs.shape[:2]
    intrinsics = torch.from_numpy(clip.intrinsics[:, 0]).float()
    extrinsics = torch.eye(4, dtype=torch.float32).repeat(views, 1, 1)
    extrinsics[:, :3, :] = torch.from_numpy(clip.extrinsics_w2c_m[:, 0]).float()
    raw_dir = run_dir / "duster"
    report(
        "3/8 depth",
        f"DUSt3R complete graph: {views} views x {timestamps} timestamps, "
        f"GA={args.duster_ga_niter}",
        not args.no_progress,
    )

    try:
        run_duster(
            torch.from_numpy(clip.rgbs),
            raw_dir,
            intrinsics[:, 0, 0],
            intrinsics[:, 1, 1],
            intrinsics[:, 0, 2],
            intrinsics[:, 1, 2],
            extrinsics,
            model_name_or_path=str(checkpoint),
            device=torch.device(args.device),
            image_size=args.duster_image_size,
            skip_if_output_already_exists=not args.recompute_depth,
            silent=args.no_progress,
            output_2d_matches=False,
            dump_exhaustive_data=False,
            save_ply=False,
            save_png_viz=False,
            show_debug_plots=False,
            save_rerun_viz=False,
            ga_lr=args.duster_ga_lr,
            ga_schedule="linear",
            scenegraph_type="complete",
            use_known_poses_for_pairwise_pose_init=True,
            ga_niter=args.duster_ga_niter,
            min_conf_thr=args.duster_conf_threshold,
            mask_sky=False,
            clean_depth=not args.duster_no_clean_depth,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory during DUSt3R reconstruction. Reduce "
            "--target-frames or --duster-image-size; completed frame files can resume."
        ) from exc

    depths_per_time = []
    valid_per_time = []
    for time_index in tqdm(
        range(timestamps),
        desc="Collect DUSt3R depth",
        unit="frame",
        disable=args.no_progress,
        dynamic_ncols=True,
    ):
        scene_path = raw_dir / f"3d_model__{time_index:05d}__scene.npz"
        if not scene_path.is_file():
            raise RuntimeError(f"DUSt3R did not create expected output: {scene_path}")
        with np.load(scene_path) as scene:
            depth = np.asarray(scene["depths"], dtype=np.float32)
            mask = np.asarray(scene["cleaned_mask"], dtype=bool)
        if depth.shape != mask.shape or depth.shape[0] != views:
            raise ValueError(
                f"Unexpected DUSt3R output depth={depth.shape}, mask={mask.shape}; "
                f"expected first dimension {views}"
            )
        valid = mask & np.isfinite(depth) & (depth > 0)
        depth[~valid] = 0
        depths_per_time.append(depth)
        valid_per_time.append(float(valid.mean()))

    depths = np.stack(depths_per_time, axis=1)
    (raw_dir / "manifest.json").write_text(
        json.dumps(
            {
                "backend": "duster",
                "checkpoint": str(checkpoint),
                "frame_indices": clip.frame_indices.tolist(),
                "ga_niter": args.duster_ga_niter,
                "confidence_threshold": args.duster_conf_threshold,
                "valid_fraction_per_timestamp": valid_per_time,
                "calibration": "fixed intrinsics and world-to-camera extrinsics in metres",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    gc.collect()
    torch.cuda.empty_cache()
    return depths
