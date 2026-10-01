#!/usr/bin/env python3
"""FoundationStereo worker executed inside the isolated foundation_stereo environment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.foundationstereo_geometry import (  # noqa: E402
    build_rectified_pair,
    disparity_to_depth,
    downsample_depth_median,
    left_right_masks,
    parse_pairs,
    scale_intrinsics,
    unflip_reverse_disparity,
)
from scripts.test_session_adapter import load_session_metadata  # noqa: E402


def predict(model, padder_class, left: np.ndarray, right: np.ndarray, iters: int, hiera: bool, small_ratio: float) -> np.ndarray:
    tensors = [torch.from_numpy(image).cuda().float().permute(2, 0, 1)[None] for image in (left, right)]
    padder = padder_class(tensors[0].shape, divis_by=32, force_square=False)
    image0, image1 = padder.pad(*tensors)
    with torch.inference_mode(), torch.cuda.amp.autocast(True):
        if hiera:
            disparity = model.run_hierachical(image0, image1, iters=iters, test_mode=True, small_ratio=small_ratio)
        else:
            disparity = model.forward(image0, image1, iters=iters, test_mode=True)
    return padder.unpad(disparity.float()).cpu().numpy().reshape(left.shape[:2])


def save_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii")); vertices.tofile(handle)


def world_points(depth, rgb, intrinsic, extrinsic, stride, radius):
    sampled = depth[::stride, ::stride]; colors = rgb[::stride, ::stride].reshape(-1, 3)
    yy, xx = np.mgrid[:sampled.shape[0], :sampled.shape[1]]
    pixels = np.c_[xx.ravel() * stride, yy.ravel() * stride, np.ones(xx.size)]
    z = sampled.ravel(); valid = np.isfinite(z) & (z > 0)
    xyz_cam = (np.linalg.inv(intrinsic) @ pixels.T).T * z[:, None]
    world = (xyz_cam - extrinsic[:3, 3]) @ extrinsic[:3, :3]
    valid &= np.isfinite(world).all(1)
    if radius > 0: valid &= np.linalg.norm(world, axis=1) <= radius
    return world[valid].astype(np.float32), colors[valid].astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    root, checkpoint = Path(job["root"]).resolve(), Path(job["checkpoint"]).resolve()
    sys.path.insert(0, str(root))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    cfg = OmegaConf.load(checkpoint.parent / "cfg.yaml")
    if "vit_size" not in cfg: cfg["vit_size"] = "vitl"
    model = FoundationStereo(cfg)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(state["model"]); model.cuda().eval()

    cameras, _, _ = load_session_metadata(Path(job["session_dir"]))
    source_size = cameras[0].image_size
    scale = float(job["scale"])
    inference_size = (int(round(source_size[0] * scale)), int(round(source_size[1] * scale)))
    pairs = [build_rectified_pair(cameras, *pair, source_size) for pair in parse_pairs(job["pairs"])]
    rectified_by_view = {}
    for pair in pairs:
        for view in (pair.left, pair.right): rectified_by_view[view] = pair
    caps = [cv2.VideoCapture(str(camera.video_path)) for camera in cameras]
    if not all(cap.isOpened() for cap in caps): raise RuntimeError("Could not open all camera videos")

    output_dir = Path(job["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    cache_signature = json.dumps({
        key: job[key] for key in (
            "cache_version", "commit", "checkpoint", "pairs", "valid_iters", "hiera",
            "hiera_small_ratio", "scale", "lr_consistency_px",
            "min_depth_m", "max_depth_m",
        )
    }, sort_keys=True)
    ply_dir = Path(job["ply_dir"]); ply_dir.mkdir(parents=True, exist_ok=True)
    tracking_rgbs, tracking_depths, elapsed, valid_fractions, dense_counts = [], [], [], [], []
    out_w, out_h = job["tracking_size"]
    for time_index, frame_index in enumerate(job["frame_indices"]):
        frame_path = output_dir / f"frame_{time_index:05d}.npz"
        begin = time.perf_counter()
        rectified = [None] * 4
        for view, cap in enumerate(caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index)); ok, bgr = cap.read()
            if not ok: raise RuntimeError(f"Cannot decode camera {view}, frame {frame_index}")
            pair = rectified_by_view[view]; mx, my = pair.maps[view]
            image = cv2.cvtColor(cv2.remap(bgr, mx, my, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            rectified[view] = image if scale == 1 else cv2.resize(image, inference_size, interpolation=cv2.INTER_AREA)
        cache_ok = False
        if frame_path.is_file() and not job["recompute"]:
            try:
                with np.load(frame_path) as cached:
                    depths = np.asarray(cached["depths_m"], np.float32)
                    disparities = np.asarray(cached["disparities_px"], np.float32)
                    masks = np.asarray(cached["valid_masks"], bool)
                    cached_frame = int(cached["frame_index"])
                    cached_signature = str(cached["cache_signature"])
                cache_ok = (
                    cached_frame == int(frame_index)
                    and cached_signature == cache_signature
                    and depths.shape == (4, inference_size[1], inference_size[0])
                    and disparities.shape == depths.shape
                    and masks.shape == depths.shape
                )
            except (KeyError, OSError, ValueError):
                cache_ok = False
        if not cache_ok:
            depths = np.zeros((4, inference_size[1], inference_size[0]), np.float32)
            disparities = np.zeros_like(depths); masks = np.zeros_like(depths, bool)
            for pair in pairs:
                left, right = rectified[pair.left], rectified[pair.right]
                dl = predict(model, InputPadder, left, right, job["valid_iters"], job["hiera"], job["hiera_small_ratio"])
                dr = unflip_reverse_disparity(
                    predict(model, InputPadder, np.fliplr(right).copy(), np.fliplr(left).copy(),
                            job["valid_iters"], job["hiera"], job["hiera_small_ratio"])
                )
                ml, mr = left_right_masks(dl, dr, job["lr_consistency_px"])
                for view, disparity, mask in ((pair.left, dl, ml), (pair.right, dr, mr)):
                    depth = disparity_to_depth(disparity, pair.intrinsics[view][0, 0] * scale, pair.baseline_m)
                    mask &= (depth >= job["min_depth_m"]) & (depth <= job["max_depth_m"])
                    depth[~mask] = 0; disparities[view] = disparity; masks[view] = mask; depths[view] = depth
            frame_intrinsics = np.stack([rectified_by_view[v].intrinsics[v] for v in range(4)]).astype(np.float32)
            frame_intrinsics[:, :2] *= scale
            frame_extrinsics = np.stack([rectified_by_view[v].extrinsics[v] for v in range(4)]).astype(np.float32)
            np.savez_compressed(
                frame_path,
                depths_m=depths,
                disparities_px=disparities,
                valid_masks=masks,
                frame_index=np.int32(frame_index),
                cache_signature=np.asarray(cache_signature),
                rectified_intrinsics=frame_intrinsics,
                rectified_extrinsics_w2c_m=frame_extrinsics,
            )
        low_rgb = np.stack([cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA).transpose(2, 0, 1) for image in rectified])
        low_depth = np.stack([downsample_depth_median(depth, (out_w, out_h)) for depth in depths])
        tracking_rgbs.append(low_rgb); tracking_depths.append(low_depth)
        if job["export_dense_ply"]:
            pts, cols = [], []
            for view in range(4):
                pair = rectified_by_view[view]
                full_k = scale_intrinsics(pair.intrinsics[view], scale, scale)
                p, c = world_points(depths[view], rectified[view], full_k, pair.extrinsics[view], job["point_stride"], job["point_radius_m"])
                pts.append(p); cols.append(c)
            points, colors = np.concatenate(pts), np.concatenate(cols)
            save_binary_ply(ply_dir / f"scene_{int(frame_index):06d}.ply", points, colors)
            dense_counts.append(len(points))
        elapsed.append(time.perf_counter() - begin); valid_fractions.append(masks.mean(axis=(1, 2)).tolist())
        print(f"FoundationStereo {time_index + 1}/{len(job['frame_indices'])}: frame={frame_index}, valid={masks.mean():.1%}", flush=True)
    for cap in caps: cap.release()
    full_intrinsics = np.stack([rectified_by_view[v].intrinsics[v] for v in range(4)]).astype(np.float32)
    inference_intrinsics = full_intrinsics.copy()
    inference_intrinsics[:, :2] *= scale
    tracking_intrinsics = full_intrinsics.copy()
    tracking_intrinsics[:, 0, :] *= out_w / source_size[0]
    tracking_intrinsics[:, 1, :] *= out_h / source_size[1]
    extrinsics = np.stack([rectified_by_view[v].extrinsics[v] for v in range(4)]).astype(np.float32)
    t = len(job["frame_indices"])
    np.savez_compressed(output_dir / "tracking_clip.npz",
        rgbs=np.stack(tracking_rgbs, axis=1), depths_m=np.stack(tracking_depths, axis=1),
        intrinsics=np.repeat(tracking_intrinsics[:, None], t, axis=1),
        extrinsics_w2c_m=np.repeat(extrinsics[:, None], t, axis=1))
    manifest = {"backend": "foundationstereo", "cache_version": job["cache_version"],
        "checkpoint": str(checkpoint), "commit": job["commit"],
        "frame_indices": job["frame_indices"], "pairs_requested": job["pairs"],
        "pairs_rectified": [[p.left, p.right] for p in pairs], "baselines_m": [p.baseline_m for p in pairs],
        "source_size": source_size, "inference_size": inference_size, "tracking_size": job["tracking_size"], "scale": scale, "hiera": job["hiera"],
        "hiera_small_ratio": job["hiera_small_ratio"],
        "valid_iters": job["valid_iters"], "lr_consistency_px": job["lr_consistency_px"],
        "min_depth_m": job["min_depth_m"], "max_depth_m": job["max_depth_m"],
        "export_dense_ply": job["export_dense_ply"], "point_stride": job["point_stride"],
        "point_radius_m": job["point_radius_m"],
        "valid_fraction_per_timestamp_view": valid_fractions, "seconds_per_timestamp": elapsed,
        "rectified_intrinsics_full": full_intrinsics.tolist(),
        "rectified_intrinsics_inference": inference_intrinsics.tolist(),
        "rectified_intrinsics_tracking": tracking_intrinsics.tolist(),
        "rectified_extrinsics_w2c_m": extrinsics.tolist(),
        "rectification_rotations": [rectified_by_view[v].rectification_rotations[v].tolist() for v in range(4)],
        "dense_counts": dense_counts, "license": "non-commercial research only"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
