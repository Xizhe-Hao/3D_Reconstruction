"""Isolated-environment adapter for the FoundationStereo depth backend."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from scripts.foundationstereo_geometry import parse_pairs
from scripts.test_session_adapter import SessionClip


FOUNDATIONSTEREO_COMMIT = "6e8806816b533e4d13ddbb95ffa907b797060a62"


def estimate_depths_with_foundationstereo(
    args, clip: SessionClip, run_dir: Path, report: Callable[[str, str, bool], None]
) -> tuple[np.ndarray, list[int]]:
    root = args.foundationstereo_root.expanduser().resolve()
    checkpoint = args.foundationstereo_checkpoint.expanduser().resolve()
    if not (root / "core" / "foundation_stereo.py").is_file():
        raise FileNotFoundError(f"FoundationStereo source not found at {root}; run scripts/setup_foundationstereo.sh")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != FOUNDATIONSTEREO_COMMIT:
        raise RuntimeError(
            f"FoundationStereo commit mismatch: expected {FOUNDATIONSTEREO_COMMIT}, got {actual_commit}"
        )
    if not checkpoint.is_file() or not (checkpoint.parent / "cfg.yaml").is_file():
        raise FileNotFoundError(f"FoundationStereo checkpoint or cfg.yaml missing near {checkpoint}; run scripts/setup_foundationstereo.sh")
    parse_pairs(args.foundationstereo_pairs)
    output_dir = run_dir / "foundationstereo"
    output_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "cache_version": 1,
        "root": str(root), "checkpoint": str(checkpoint), "commit": FOUNDATIONSTEREO_COMMIT,
        "session_dir": str(args.session_dir.expanduser().resolve()),
        "output_dir": str(output_dir), "ply_dir": str(run_dir / "ply_dense"),
        "frame_indices": clip.frame_indices.tolist(), "tracking_size": [args.width, args.height],
        "pairs": args.foundationstereo_pairs, "valid_iters": args.foundationstereo_valid_iters,
        "hiera": args.foundationstereo_hiera, "hiera_small_ratio": args.foundationstereo_hiera_small_ratio,
        "scale": args.foundationstereo_scale,
        "lr_consistency_px": args.foundationstereo_lr_consistency_px,
        "min_depth_m": args.min_depth_m, "max_depth_m": args.max_depth_m,
        "point_stride": args.pointcloud_pixel_stride, "point_radius_m": args.pointcloud_radius_m,
        "export_dense_ply": not args.no_dense_ply, "recompute": args.recompute_depth,
    }
    job_path = output_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    tracking_path = output_dir / "tracking_clip.npz"
    manifest_path = output_dir / "manifest.json"
    cache_matches = False
    if tracking_path.is_file() and manifest_path.is_file() and not args.recompute_depth:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache_matches = (previous.get("cache_version") == job["cache_version"]
                         and previous.get("commit") == job["commit"]
                         and previous.get("checkpoint") == job["checkpoint"]
                         and previous.get("frame_indices") == job["frame_indices"]
                         and previous.get("pairs_requested") == job["pairs"]
                         and previous.get("tracking_size") == job["tracking_size"]
                         and previous.get("hiera") == job["hiera"]
                         and previous.get("valid_iters") == job["valid_iters"]
                         and previous.get("hiera_small_ratio") == job["hiera_small_ratio"]
                         and previous.get("scale") == job["scale"]
                         and previous.get("lr_consistency_px") == job["lr_consistency_px"]
                         and previous.get("min_depth_m") == job["min_depth_m"]
                         and previous.get("max_depth_m") == job["max_depth_m"]
                         and previous.get("export_dense_ply") == job["export_dense_ply"]
                         and previous.get("point_stride") == job["point_stride"]
                         and previous.get("point_radius_m") == job["point_radius_m"])
    if not cache_matches:
        job["recompute"] = args.recompute_depth or manifest_path.is_file()
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        report("3/8 depth", f"running FoundationStereo in conda env {args.foundationstereo_conda_env}", not args.no_progress)
        command = ["conda", "run", "--no-capture-output", "-n", args.foundationstereo_conda_env,
                   "python", str(Path(__file__).with_name("foundationstereo_worker.py")), "--job", str(job_path)]
        try:
            subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"FoundationStereo worker failed with exit code {exc.returncode}") from exc
    with np.load(tracking_path) as data:
        rgbs = np.asarray(data["rgbs"], np.uint8)
        depths = np.asarray(data["depths_m"], np.float32)
        intrinsics = np.asarray(data["intrinsics"], np.float32)
        extrinsics = np.asarray(data["extrinsics_w2c_m"], np.float32)
    expected = (len(clip.camera_serials), len(clip.frame_indices), args.height, args.width)
    if depths.shape != expected or rgbs.shape != expected[:2] + (3, args.height, args.width):
        raise ValueError(f"FoundationStereo tracking cache has incompatible RGB={rgbs.shape}, depth={depths.shape}")
    clip.rgbs = rgbs
    clip.intrinsics = intrinsics
    clip.extrinsics_w2c_m = extrinsics
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return depths, [int(value) for value in manifest.get("dense_counts", [])]
