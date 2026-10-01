"""Run pretrained MVTracker on a synchronized ``data/test`` clip.

The result is a sparse 4D point trajectory rather than a dense deforming mesh.
Use ``--depth-backend duster`` for calibrated multi-view depth, ``moge2`` for a
fast monocular baseline, or ``npz`` for externally computed metric depth maps.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MVTRACKER_ROOT = PROJECT_ROOT / "submodule" / "mvtracker"
for index, path in enumerate((PROJECT_ROOT, MVTRACKER_ROOT)):
    if str(path) not in sys.path:
        sys.path.insert(index, str(path))

from scripts.test_session_adapter import SessionClip, load_session_clip
from scripts.duster_depth_backend import estimate_depths_with_duster


_RUN_STARTED = time.perf_counter()


def report(stage: str, message: str, enabled: bool = True) -> None:
    if enabled:
        elapsed = time.perf_counter() - _RUN_STARTED
        print(f"[{elapsed:8.1f}s] [{stage}] {message}", flush=True)


def cuda_memory(device: str) -> str:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "CPU mode"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"CUDA allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB"



@contextmanager
def inference_heartbeat(enabled: bool, device: str, interval_s: float):
    stop = threading.Event()

    def worker() -> None:
        while not stop.wait(interval_s):
            report("6/8 inference", f"still running; {cuda_memory(device)}", enabled)

    thread = threading.Thread(target=worker, name="mvtracker-progress", daemon=True)
    if enabled:
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if enabled:
            thread.join(timeout=1.0)


def load_or_estimate_depths(args: argparse.Namespace, clip: SessionClip, run_dir: Path) -> np.ndarray:
    V, T, _, H, W = clip.rgbs.shape
    cache_path = run_dir / f"depths_{args.depth_backend}_m.npz"
    if cache_path.is_file() and not args.recompute_depth:
        report("3/8 depth", f"loading cache {cache_path}", not args.no_progress)
        data = np.load(cache_path)
        depths = data["depths_m"]
    elif args.depth_backend == "npz":
        report("3/8 depth", f"loading NPZ {args.depth_path}", not args.no_progress)
        if args.depth_path is None:
            raise ValueError("--depth-path is required for --depth-backend npz")
        data = np.load(args.depth_path)
        key = "depths_m" if "depths_m" in data else "depths"
        depths = np.asarray(data[key], dtype=np.float32)
        if depths.ndim == 5 and depths.shape[2] == 1:
            depths = depths[:, :, 0]
        if depths.shape[1] != T and "frame_indices" in data:
            lookup = {int(frame): i for i, frame in enumerate(data["frame_indices"])}
            depths = np.stack([depths[:, lookup[int(frame)]] for frame in clip.frame_indices], axis=1)
        if args.depth_unit == "mm":
            depths = depths / 1000.0
    elif args.depth_backend == "duster":
        depths = estimate_depths_with_duster(args, clip, run_dir, report)
    elif args.depth_backend == "moge2":
        report("3/8 depth", f"loading MoGe-2 model {args.moge_model}", not args.no_progress)
        try:
            from moge.model.v2 import MoGeModel
        except ImportError as exc:
            raise RuntimeError(
                "MoGe-2 is not installed. Run scripts/setup_mvtracker.sh or use --depth-backend npz."
            ) from exc
        device = torch.device(args.device)
        model = MoGeModel.from_pretrained(args.moge_model, token=False).to(device).eval()
        report("3/8 depth", f"MoGe-2 ready; {cuda_memory(args.device)}", not args.no_progress)
        flat = torch.from_numpy(clip.rgbs.reshape(V * T, 3, H, W)).float() / 255.0
        output = []
        with torch.inference_mode():
            batch_starts = range(0, len(flat), args.depth_batch_size)
            for begin in tqdm(
                batch_starts, desc="MoGe-2 depth", unit="batch",
                disable=args.no_progress, dynamic_ncols=True,
            ):
                images = flat[begin : begin + args.depth_batch_size].to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    prediction = model.infer(images)
                depth = torch.as_tensor(prediction["depth"]).float().cpu()
                mask = prediction.get("mask")
                if mask is not None:
                    depth[~torch.as_tensor(mask).bool().cpu()] = 0
                output.append(depth)
        depths = torch.cat(output).reshape(V, T, H, W).numpy()
        del model
        torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unknown depth backend: {args.depth_backend}")

    depths = np.asarray(depths, dtype=np.float32)
    if depths.shape != (V, T, H, W):
        if depths.shape[:2] != (V, T):
            raise ValueError(f"Depth shape {depths.shape}; expected {(V, T, H, W)}")
        resized = np.empty((V, T, H, W), dtype=np.float32)
        for v in range(V):
            for t in range(T):
                resized[v, t] = cv2.resize(depths[v, t], (W, H), interpolation=cv2.INTER_NEAREST)
        depths = resized
    depths[~np.isfinite(depths)] = 0
    depths[(depths < args.min_depth_m) | (depths > args.max_depth_m)] = 0
    valid_fraction = float((depths > 0).mean())
    report("3/8 depth", f"saving cache; valid={valid_fraction:.1%}", not args.no_progress)
    np.savez_compressed(cache_path, depths_m=depths, frame_indices=clip.frame_indices)
    report("3/8 depth", f"complete: shape={depths.shape}", not args.no_progress)
    return depths


def camera_centers(extrs: np.ndarray) -> np.ndarray:
    R = extrs[:, 0, :3, :3]
    t = extrs[:, 0, :3, 3]
    return -np.einsum("vji,vj->vi", R, t)


def make_queries(
    depths: np.ndarray,
    intrs: np.ndarray,
    extrs: np.ndarray,
    views: list[int],
    grid_size: int,
    roi: tuple[float, float, float, float],
    world_radius_m: float | None,
    voxel_size_m: float,
) -> np.ndarray:
    _, _, H, W = depths.shape
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("--roi values must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    xs = np.linspace(x0 * (W - 1), x1 * (W - 1), grid_size)
    ys = np.linspace(y0 * (H - 1), y1 * (H - 1), grid_size)
    xx, yy = np.meshgrid(xs, ys)
    xi, yi = np.rint(xx).astype(int), np.rint(yy).astype(int)
    world_per_view = []
    for view in views:
        z = depths[view, 0, yi, xi]
        valid = z > 0
        pixels = np.stack(
            [xx[valid], yy[valid], np.ones(valid.sum())], axis=1
        ).astype(np.float32)
        if len(pixels) == 0:
            continue
        xyz_cam = (np.linalg.inv(intrs[view, 0]) @ pixels.T).T * z[valid, None]
        E = extrs[view, 0]
        xyz_world = (xyz_cam - E[:3, 3]) @ E[:3, :3]
        if world_radius_m is not None:
            xyz_world = xyz_world[np.linalg.norm(xyz_world, axis=1) <= world_radius_m]
        if len(xyz_world):
            world_per_view.append(xyz_world.astype(np.float32))
    if not world_per_view:
        raise RuntimeError("No query points remain after depth and --world-radius-m filtering")
    xyz_world = np.concatenate(world_per_view, axis=0)
    if voxel_size_m > 0:
        voxel = np.floor(xyz_world / voxel_size_m + 0.5).astype(np.int64)
        _, keep = np.unique(voxel, axis=0, return_index=True)
        xyz_world = xyz_world[np.sort(keep)]
    return np.c_[np.zeros(len(xyz_world), dtype=np.float32), xyz_world].astype(np.float32)


def export_csv(
    path: Path, tracks: np.ndarray, visibility: np.ndarray, clip: SessionClip, show_progress: bool = False
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "frame_index", "video_time_s", "x_m", "y_m", "z_m", "visible"])
        for t in tqdm(
            range(tracks.shape[0]), desc="Export CSV", unit="frame",
            disable=not show_progress, dynamic_ncols=True,
        ):
            for n in range(tracks.shape[1]):
                writer.writerow(
                    [n, int(clip.frame_indices[t]), float(clip.video_times_s[t]), *tracks[t, n].tolist(), int(visibility[t, n])]
                )


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            handle.write(f"{p[0]:.8g} {p[1]:.8g} {p[2]:.8g} {c[0]} {c[1]} {c[2]}\n")


def rgbd_world_points(
    clip: SessionClip, depths_m: np.ndarray, view: int, time_index: int,
    pixel_stride: int, radius_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject one RGB-D view to the calibrated world frame."""
    depth = depths_m[view, time_index, ::pixel_stride, ::pixel_stride]
    rgb = clip.rgbs[view, time_index].transpose(1, 2, 0)[::pixel_stride, ::pixel_stride]
    height, width = depth.shape
    yy, xx = np.mgrid[0:height, 0:width]
    xx = xx.astype(np.float32) * pixel_stride
    yy = yy.astype(np.float32) * pixel_stride
    pixels = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size, dtype=np.float32)], axis=1)
    z = depth.ravel()
    valid = np.isfinite(z) & (z > 0)
    camera_xyz = (np.linalg.inv(clip.intrinsics[view, time_index]) @ pixels.T).T * z[:, None]
    E = clip.extrinsics_w2c_m[view, time_index]
    world_xyz = (camera_xyz - E[:3, 3]) @ E[:3, :3]
    valid &= np.isfinite(world_xyz).all(axis=1)
    if radius_m is not None:
        valid &= np.linalg.norm(world_xyz, axis=1) <= radius_m
    return world_xyz[valid].astype(np.float32), rgb.reshape(-1, 3)[valid].astype(np.uint8)


def save_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def export_dense_ply(
    output_dir: Path, clip: SessionClip, depths_m: np.ndarray, pixel_stride: int,
    radius_m: float | None, show_progress: bool,
) -> list[int]:
    output_dir.mkdir(exist_ok=True)
    counts = []
    iterator = tqdm(
        enumerate(clip.frame_indices), total=len(clip.frame_indices), desc="Export dense PLY",
        unit="frame", disable=not show_progress, dynamic_ncols=True,
    )
    for time_index, frame_index in iterator:
        per_view = [
            rgbd_world_points(clip, depths_m, view, time_index, pixel_stride, radius_m)
            for view in range(clip.rgbs.shape[0])
        ]
        points = np.concatenate([item[0] for item in per_view], axis=0)
        colors = np.concatenate([item[1] for item in per_view], axis=0)
        save_binary_ply(output_dir / f"scene_{int(frame_index):06d}.ply", points, colors)
        counts.append(len(points))
    return counts


def export_rerun(
    path: Path, tracks: np.ndarray, visibility: np.ndarray, clip: SessionClip,
    depths_m: np.ndarray | None = None, pointcloud_pixel_stride: int = 4,
    pointcloud_radius_m: float | None = 0.5, pointcloud_point_radius_m: float = 0.001,
    pointcloud_mode: str = "fused",
    show_progress: bool = False,
) -> None:
    import rerun as rr

    rr.init("mvtracker_data_test", recording_id=path.parent.name)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for view, serial in enumerate(clip.camera_serials):
        E = clip.extrinsics_w2c_m[view, 0]
        camera_to_world_R = E[:3, :3].T
        camera_center = -camera_to_world_R @ E[:3, 3]
        camera_path = f"world/cameras/camera_{view}_{serial}"
        rr.log(
            camera_path,
            rr.Transform3D(translation=camera_center, mat3x3=camera_to_world_R),
            static=True,
        )
        rr.log(
            camera_path,
            rr.Pinhole(
                image_from_camera=clip.intrinsics[view, 0],
                width=clip.rgbs.shape[-1], height=clip.rgbs.shape[-2],
            ),
            static=True,
        )
    rng = np.random.default_rng(72)
    colors = rng.integers(40, 256, size=(tracks.shape[1], 3), dtype=np.uint8)
    for t, time_s in tqdm(
        enumerate(clip.video_times_s), total=len(clip.video_times_s), desc="Export Rerun",
        unit="frame", disable=not show_progress, dynamic_ncols=True,
    ):
        rr.set_time_sequence("frame_index", int(clip.frame_indices[t]))
        rr.set_time_seconds("video_time", float(time_s))
        if depths_m is not None:
            per_view = [
                rgbd_world_points(
                    clip, depths_m, view, t, pointcloud_pixel_stride, pointcloud_radius_m
                )
                for view in range(len(clip.camera_serials))
            ]
            if pointcloud_mode == "fused":
                points = np.concatenate([item[0] for item in per_view], axis=0)
                point_colors = np.concatenate([item[1] for item in per_view], axis=0)
                rr.log(
                    "world/pointcloud/fused",
                    rr.Points3D(points, colors=point_colors, radii=pointcloud_point_radius_m),
                )
            else:
                for view, serial in enumerate(clip.camera_serials):
                    points, point_colors = per_view[view]
                    entity = f"world/pointcloud/camera_{view}_{serial}"
                    if len(points):
                        rr.log(
                            entity,
                            rr.Points3D(points, colors=point_colors, radii=pointcloud_point_radius_m),
                        )
                    else:
                        rr.log(entity, rr.Clear(recursive=False))
        visible = visibility[t]
        rr.log("world/tracks/current", rr.Points3D(tracks[t, visible], colors=colors[visible], radii=0.002))
        strips, strip_colors = [], []
        for n in np.flatnonzero(visible):
            history_mask = visibility[: t + 1, n]
            history = tracks[: t + 1, n][history_mask]
            if len(history) >= 2:
                strips.append(history)
                strip_colors.append(colors[n])
        if strips:
            rr.log("world/tracks/history", rr.LineStrips3D(strips, colors=strip_colors, radii=0.0007))
    rr.save(path)


def parse_roi(value: str) -> tuple[float, float, float, float]:
    parts = tuple(float(x) for x in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1 in normalized image coordinates")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=PROJECT_ROOT / "data/test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--target-frames", type=int,
        help="Uniformly sample exactly N synchronized timestamps, including range endpoints",
    )
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--depth-backend", choices=["duster", "moge2", "npz"], default="duster")
    parser.add_argument("--depth-path", type=Path)
    parser.add_argument("--depth-unit", choices=["m", "mm"], default="m")
    parser.add_argument("--moge-model", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--duster-root", type=Path, default=MVTRACKER_ROOT.parent / "duster")
    parser.add_argument("--duster-checkpoint", type=Path, default=MVTRACKER_ROOT.parent / "duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    parser.add_argument("--duster-image-size", type=int, choices=[224, 512], default=512)
    parser.add_argument("--duster-ga-niter", type=int, default=300)
    parser.add_argument("--duster-ga-lr", type=float, default=0.01)
    parser.add_argument("--duster-conf-threshold", type=float, default=20.0)
    parser.add_argument("--duster-no-clean-depth", action="store_true")
    parser.add_argument("--depth-batch-size", type=int, default=4)
    parser.add_argument("--recompute-depth", action="store_true")
    parser.add_argument("--min-depth-m", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument("--query-view", type=int, choices=range(4), help="Legacy single-view override")
    parser.add_argument("--query-views", default="0,1,2,3", help="Comma-separated initialization views")
    parser.add_argument("--query-voxel-size-m", type=float, default=0.005)
    parser.add_argument("--query-grid-size", type=int, default=16)
    parser.add_argument("--roi", type=parse_roi, default=(0.2, 0.2, 0.8, 0.8))
    parser.add_argument("--world-radius-m", type=float, default=0.18)
    parser.add_argument("--pointcloud-pixel-stride", type=int, default=4)
    parser.add_argument("--pointcloud-radius-m", type=float, default=0.5)
    parser.add_argument("--pointcloud-point-radius-m", type=float, default=0.001)
    parser.add_argument("--rerun-pointcloud-mode", choices=["fused", "per-view"], default="fused")
    parser.add_argument("--no-dense-ply", action="store_true")
    parser.add_argument("--no-dense-rerun", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=MVTRACKER_ROOT / "checkpoints/mvtracker_200000_june2025.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/data_test")
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir
    show_progress = not args.no_progress
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be positive")
    if args.duster_ga_niter < 0 or args.duster_ga_lr <= 0:
        parser.error("--duster-ga-niter must be non-negative and --duster-ga-lr positive")
    if args.duster_conf_threshold <= 0:
        parser.error("--duster-conf-threshold must be positive")
    if args.pointcloud_pixel_stride <= 0:
        parser.error("--pointcloud-pixel-stride must be positive")
    if args.pointcloud_radius_m <= 0 or args.pointcloud_point_radius_m <= 0:
        parser.error("point-cloud radii must be positive")
    try:
        query_views = [args.query_view] if args.query_view is not None else [int(v) for v in args.query_views.split(",")]
    except ValueError:
        parser.error("--query-views must be comma-separated camera indices, for example 0,1,2,3")
    query_views = list(dict.fromkeys(query_views))
    if not query_views or any(view not in range(4) for view in query_views):
        parser.error("query views must contain only camera indices 0,1,2,3")
    if args.query_voxel_size_m < 0:
        parser.error("--query-voxel-size-m cannot be negative")
    report("1/8 setup", f"arguments parsed; device={args.device}", show_progress)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    sampling_name = f"target_{args.target_frames}" if args.target_frames is not None else f"step_{args.step}"
    run_name = f"frames_{args.start}_{args.end}_{sampling_name}_{args.depth_backend}"
    run_dir = (args.output_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report("2/8 decode", f"loading synchronized session {args.session_dir}", show_progress)
    clip = load_session_clip(
        args.session_dir, args.start, args.end, args.step, (args.width, args.height), args.max_frames,
        args.target_frames,
        show_progress,
    )
    report("2/8 decode", f"complete: RGB={clip.rgbs.shape}, frames={clip.frame_indices[0]}..{clip.frame_indices[-1]}", show_progress)
    if len(clip.frame_indices) < 7:
        raise ValueError("MVTracker requires at least 7 sampled frames for its 12-frame sliding window")
    report("3/8 depth", f"backend={args.depth_backend}", show_progress)
    depths_m = load_or_estimate_depths(args, clip, run_dir)
    report("4/8 queries", f"views={query_views}, grid={args.query_grid_size}x{args.query_grid_size}, roi={args.roi}", show_progress)
    queries_m = make_queries(
        depths_m, clip.intrinsics, clip.extrinsics_w2c_m, query_views, args.query_grid_size, args.roi,
        args.world_radius_m, args.query_voxel_size_m,
    )
    report("4/8 queries", f"retained {len(queries_m)} tracks after depth/world filtering", show_progress)

    # MVTracker was trained at a much larger normalized scene scale.  Scale all
    # Euclidean quantities consistently and undo the scale on the prediction.
    centers = camera_centers(clip.extrinsics_w2c_m)
    scene_scale = 6.3 / float(np.median(np.linalg.norm(centers, axis=1)))
    depths_model = depths_m * scene_scale
    extrs_model = clip.extrinsics_w2c_m.copy()
    extrs_model[..., 3] *= scene_scale
    queries_model = queries_m.copy()
    queries_model[:, 1:] *= scene_scale

    report("5/8 model", f"loading checkpoint {args.checkpoint}", show_progress)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run scripts/download_assets.py")
    import hubconf

    hubconf._WEIGHTS["mvtracker_main"] = str(checkpoint)
    predictor = hubconf.mvtracker_predictor(
        pretrained=True,
        device=args.device,
        checkpoint="mvtracker_main",
        model_kwargs={"use_flash_attention": True, "normalize_scene_in_fwd_pass": False},
        predictor_kwargs={"interp_shape": (args.height, args.width), "n_iters": args.iterations},
    )
    report("5/8 model", f"checkpoint loaded; {cuda_memory(args.device)}", show_progress)
    report("6/8 inference", "moving input tensors to device", show_progress)
    rgbs = torch.from_numpy(clip.rgbs).float()[None].to(args.device) / 255.0
    depths = torch.from_numpy(depths_model)[:, :, None][None].to(args.device)
    intrs = torch.from_numpy(clip.intrinsics)[None].to(args.device)
    extrs = torch.from_numpy(extrs_model)[None].to(args.device)
    queries = torch.from_numpy(queries_model)[None].to(args.device)
    torch.set_float32_matmul_precision("high")
    amp_dtype = (
        torch.bfloat16
        if args.device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    report(
        "6/8 inference",
        f"starting official rolling-window predictor: T={len(clip.frame_indices)}, N={len(queries_m)}; {cuda_memory(args.device)}",
        show_progress,
    )
    try:
        with inference_heartbeat(show_progress, args.device, args.heartbeat_seconds):
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=args.device.startswith("cuda")
            ):
                result = predictor(
                    rgbs=rgbs, depths=depths, intrs=intrs, extrs=extrs, query_points_3d=queries
                )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory. Reduce --target-frames or --query-grid-size, or lower the image resolution."
        ) from exc
    report("6/8 inference", f"complete; {cuda_memory(args.device)}", show_progress)
    tracks_m = result["traj_e"][0].float().cpu().numpy() / scene_scale
    visibility_score = result["vis_e_as_prob"][0].float().cpu().numpy()
    visibility = result["vis_e"][0].bool().cpu().numpy()

    report("7/8 files", "saving tracks_4d.npz", show_progress)
    np.savez_compressed(
        run_dir / "tracks_4d.npz",
        trajectories_m=tracks_m,
        visibility=visibility,
        visibility_score=visibility_score,
        query_points_m=queries_m,
        frame_indices=clip.frame_indices,
        video_times_s=clip.video_times_s,
        camera_serials=np.asarray(clip.camera_serials),
        scene_scale=np.float32(scene_scale),
    )
    export_csv(run_dir / "tracks_4d.csv", tracks_m, visibility, clip, show_progress)
    ply_dir = run_dir / "ply"
    ply_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(72)
    colors = rng.integers(40, 256, size=(tracks_m.shape[1], 3), dtype=np.uint8)
    for t, frame_index in tqdm(
        enumerate(clip.frame_indices), total=len(clip.frame_indices), desc="Export PLY",
        unit="frame", disable=not show_progress, dynamic_ncols=True,
    ):
        valid = visibility[t] & np.isfinite(tracks_m[t]).all(axis=1)
        save_ply(ply_dir / f"tracks_{int(frame_index):06d}.ply", tracks_m[t, valid], colors[valid])
    dense_counts = []
    if not args.no_dense_ply:
        report("7/8 files", f"exporting fused dense PLY with pixel stride {args.pointcloud_pixel_stride}", show_progress)
        dense_counts = export_dense_ply(
            run_dir / "ply_dense", clip, depths_m, args.pointcloud_pixel_stride,
            args.pointcloud_radius_m, show_progress,
        )
    report("8/8 rerun", "building tracks_4d.rrd", show_progress)
    export_rerun(
        run_dir / "tracks_4d.rrd", tracks_m, visibility, clip,
        depths_m=None if args.no_dense_rerun else depths_m,
        pointcloud_pixel_stride=args.pointcloud_pixel_stride,
        pointcloud_radius_m=args.pointcloud_radius_m,
        pointcloud_point_radius_m=args.pointcloud_point_radius_m,
        pointcloud_mode=args.rerun_pointcloud_mode,
        show_progress=show_progress,
    )
    metadata = {
        "session_dir": str(args.session_dir.resolve()),
        "frames": clip.frame_indices.tolist(),
        "times_s": clip.video_times_s.tolist(),
        "camera_serials": clip.camera_serials,
        "depth_backend": args.depth_backend,
        "target_frames": args.target_frames,
        "query_views": query_views,
        "query_voxel_size_m": args.query_voxel_size_m,
        "dense_pointcloud": {
            "pixel_stride": args.pointcloud_pixel_stride,
            "radius_m": args.pointcloud_radius_m,
            "ply_exported": not args.no_dense_ply,
            "rerun_logged": not args.no_dense_rerun,
            "rerun_mode": args.rerun_pointcloud_mode,
            "points_per_frame_min": min(dense_counts) if dense_counts else None,
            "points_per_frame_max": max(dense_counts) if dense_counts else None,
        },
        "coordinate_system": "calibration world frame, metres",
        "trajectory_shape": list(tracks_m.shape),
        "visible_fraction": float(visibility.mean()),
        "scene_scale_used_in_model": scene_scale,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report("done", f"all outputs complete: {run_dir}", show_progress)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Outputs: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
