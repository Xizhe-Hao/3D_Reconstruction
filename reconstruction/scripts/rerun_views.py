"""Synchronized camera views for the MVTracker Rerun recording.

Adds, on the same ``frame_index`` timeline as the 3D tracks:
  * the four RGB frames (inside each camera frustum) with projected tracks,
  * the predicted metric depth that MVTracker consumed for each RGB camera,
  * the RealSense D455 colour stream and measured depth. With
    ``calibration/d455_world.json`` both D455 cameras are placed in the world
    frame, so the measured depth also appears as a 3D point cloud next to the
    reconstruction; without it the D455 is shown in 2D only.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_D455_WORLD = REPO_ROOT / "calibration" / "d455_world.json"


def camera_entity(view: int, serial: str) -> str:
    return f"world/cameras/camera_{view}_{serial}"


D455_COLOR = "world/cameras/d455_color"
D455_DEPTH = "world/cameras/d455_depth"


@dataclass
class D455Frames:
    color_paths: dict[int, Path]
    depth_paths: dict[int, Path]
    depth_scale_m: float
    depth_range_m: tuple[float, float]
    color_intrinsics: dict | None
    depth_intrinsics: dict | None
    # World poses (metres), X_cam = R @ X_world + t; None when uncalibrated.
    color_pose: tuple[np.ndarray, np.ndarray] | None = None
    depth_pose: tuple[np.ndarray, np.ndarray] | None = None
    depth_correction: dict | None = None


def _pose_m(world: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(world[f"R_world_to_{name}"], dtype=np.float64)
    t = np.asarray(world[f"t_world_to_{name}"], dtype=np.float64).reshape(3) / 1000.0
    return R, t


def load_d455_frames(session_dir: Path | None, d455_world: Path | None = DEFAULT_D455_WORLD) -> D455Frames | None:
    """Map synchronized video frame index -> D455 colour/depth files (+ world calibration)."""
    if session_dir is None:
        return None
    session_dir = session_dir.expanduser().resolve()
    alignment = session_dir / "multimodal_video_alignment.csv"
    meta_path = session_dir / "d455" / "d455.json"
    if not alignment.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    color_paths, depth_paths = {}, {}
    with alignment.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["video_frame_index"])
            for key, target in (("d455_color_jpg", color_paths), ("d455_depth_png", depth_paths)):
                rel = (row.get(key) or "").strip()
                if rel and (session_dir / rel).is_file():
                    target[frame] = session_dir / rel
    if not color_paths and not depth_paths:
        return None
    frames = D455Frames(
        color_paths, depth_paths,
        depth_scale_m=float(meta.get("depth_scale_m", 0.001)),
        depth_range_m=(float(meta.get("depth_min_range_m", 0.3)), float(meta.get("depth_max_range_m", 1.0))),
        # The session's own intrinsics match the recorded resolutions.
        color_intrinsics=meta.get("color_intrinsics"),
        depth_intrinsics=meta.get("depth_intrinsics"),
    )
    if d455_world is not None and Path(d455_world).is_file():
        world = json.loads(Path(d455_world).read_text(encoding="utf-8"))
        if str(world.get("serial")) == str(meta.get("serial")):
            frames.color_pose = _pose_m(world, "color")
            frames.depth_pose = _pose_m(world, "depth")
            frames.depth_correction = world.get("depth_correction")
        else:
            print(f"[rerun] {d455_world} is for D455 {world.get('serial')}, session has {meta.get('serial')}; "
                  "showing the D455 in 2D only", flush=True)
    return frames


INDENTATION_SERIES = {
    # column in indentation.csv -> (label, colour)
    "ground_truth_mm": ("force gauge (ground truth)", (235, 235, 235)),
    "rgb4_mm": ("4 RGB", (255, 140, 30)),
    "rgb4_d455_mm": ("4 RGB + D455", (40, 200, 255)),
    "d455_mm": ("D455 only", (230, 80, 200)),
}


def load_indentation(result_dir: Path) -> dict | None:
    """indentation.csv / indentation_summary.json written by eval_indentation.py."""
    path = result_dir / "indentation.csv"
    if not path.is_file():
        return None
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["frame_index"])] = {
                key: float(row[key]) for key in INDENTATION_SERIES if row.get(key, "") != ""
            }
    summary_path = result_dir / "indentation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return {"rows": rows, "summary": summary}


def predicted_depth_range(depths_m: np.ndarray) -> tuple[float, float]:
    sample = depths_m[:, :: max(1, depths_m.shape[1] // 32), ::4, ::4]
    valid = sample[sample > 0]
    if valid.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(valid, [2, 98])
    return float(low), float(max(high, low + 1e-3))


def build_blueprint(camera_serials: list[str], d455: D455Frames | None, has_recon_depth: bool = True,
                    has_indentation: bool = False):
    import rerun.blueprint as rrb

    cameras = [camera_entity(v, s) for v, s in enumerate(camera_serials)]
    calibrated = d455 is not None and d455.color_pose is not None
    # Keep the 3D view readable: RGB frames and 2D tracks live in the 2D panels,
    # while the D455 measured depth stays in 3D as a point cloud when calibrated.
    contents = ["+ world/**", f"- {D455_COLOR}/image"]
    if not calibrated:
        contents += [f"- {D455_COLOR}/**", f"- {D455_DEPTH}/**"]
    for entity in cameras:
        contents += [f"- {entity}/rgb", f"- {entity}/tracks"]
    rgb_row = rrb.Horizontal(*[
        rrb.Spatial2DView(origin=entity, name=f"RGB cam {v} ({s})")
        for v, (entity, s) in enumerate(zip(cameras, camera_serials))
    ])
    depth_row = rrb.Horizontal(*[
        rrb.Spatial2DView(origin=f"views/camera_{v}_{s}", name=f"Recon depth cam {v}")
        for v, s in enumerate(camera_serials)
    ])
    rows = [rgb_row, depth_row]
    left = [rrb.Spatial3DView(origin="world", contents=contents, name="3D tracks + clouds")]
    if d455 is not None:
        d455_views = [
            rrb.Spatial2DView(origin=D455_COLOR, contents=f"{D455_COLOR}/image", name="D455 colour"),
            rrb.Spatial2DView(origin=D455_DEPTH, contents=f"{D455_DEPTH}/depth", name="D455 measured depth"),
        ]
        if calibrated and has_recon_depth:
            d455_views += [
                rrb.Spatial2DView(origin="views/d455_recon_depth", name="Recon depth in D455 view"),
                rrb.Spatial2DView(origin="views/d455_abs_error", name="|D455 - recon| (0-20 mm)"),
            ]
            left.append(rrb.TimeSeriesView(origin="metrics/d455_vs_recon", name="D455 vs recon depth (mm)"))
        rows.append(rrb.Horizontal(*d455_views))
    if has_indentation:
        left.append(rrb.TimeSeriesView(origin="metrics/indentation", name="Indentation depth (mm)"))
        left.append(rrb.TimeSeriesView(origin="metrics/indentation_error",
                                       name="Indentation error vs force gauge (mm)"))
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(*left, row_shares=[3.0] + [1.0] * (len(left) - 1)),
            rrb.Vertical(*rows),
            column_shares=[1.0, 1.4],
        ),
        collapse_panels=True,
    )


def _log_pinhole(entity: str, intr: dict, scale: float, pose, static: bool = True) -> None:
    import rerun as rr

    if pose is not None:
        R, t = pose
        rr.log(entity, rr.Transform3D(translation=-R.T @ t, mat3x3=R.T), static=static)
    K = np.array([
        [intr["fx"] * scale, 0.0, intr["ppx"] * scale],
        [0.0, intr["fy"] * scale, intr["ppy"] * scale],
        [0.0, 0.0, 1.0],
    ])
    rr.log(
        entity,
        rr.Pinhole(image_from_camera=K, width=round(intr["width"] * scale), height=round(intr["height"] * scale)),
        static=static,
    )


class CameraViewLogger:
    def __init__(
        self, clip, depths_m: np.ndarray | None, d455: D455Frames | None,
        depth_downsample: int = 2, jpeg_quality: int = 85, indentation: dict | None = None,
    ) -> None:
        self.clip = clip
        self.indentation = indentation
        self.depths_m = depths_m
        self.d455 = d455
        self.depth_downsample = max(1, int(depth_downsample))
        self.jpeg_quality = jpeg_quality
        self._pixel_rays = None
        # Rerun's depth_range is in raw image units (mm here), not metres.
        self.pred_range_mm = (
            tuple(1000.0 * v for v in predicted_depth_range(depths_m)) if depths_m is not None else None
        )

    def log_static(self) -> None:
        """D455 frustums; the RGB camera frustums are logged by the caller."""
        self._log_indentation_static()
        d455 = self.d455
        if d455 is None or d455.color_pose is None:
            return
        if d455.color_intrinsics:
            _log_pinhole(D455_COLOR, d455.color_intrinsics, 1.0, d455.color_pose)
        if d455.depth_intrinsics:
            _log_pinhole(D455_DEPTH, d455.depth_intrinsics, 1.0 / self.depth_downsample, d455.depth_pose)

    def _log_indentation_static(self) -> None:
        import rerun as rr

        if self.indentation is None:
            return
        for key, (label, colour) in INDENTATION_SERIES.items():
            name = key.removesuffix("_mm")
            rr.log(f"metrics/indentation/{name}", rr.SeriesLine(color=colour, name=label, width=2.0), static=True)
            if key != "ground_truth_mm":
                rr.log(f"metrics/indentation_error/{name}", rr.SeriesLine(color=colour, name=f"{label} - truth"),
                       static=True)
        summary = self.indentation["summary"]
        if summary.get("centre_xy_mm") and summary.get("radius_mm"):
            # The disk eval_indentation.py reads the surface in, at the sponge top.
            angle = np.linspace(0, 2 * np.pi, 97)
            cx, cy = summary["centre_xy_mm"]
            z = summary.get("surface_z_mm", 0.0)
            ring = np.stack([cx + summary["radius_mm"] * np.cos(angle), cy + summary["radius_mm"] * np.sin(angle),
                             np.full_like(angle, z)], axis=1) / 1000.0
            rr.log("world/indentation_region", rr.LineStrips3D([ring], colors=[(255, 255, 0)], radii=0.0008),
                   static=True)

    def _log_indentation(self, t: int) -> None:
        import rerun as rr

        if self.indentation is None:
            return
        row = self.indentation["rows"].get(int(self.clip.frame_indices[t]))
        truth = row.get("ground_truth_mm") if row else None
        if truth is None:
            # Outside the force log there is no ground truth, and before the first
            # press the indenter is still moving in, so the drop means nothing.
            return
        for key, value in row.items():
            name = key.removesuffix("_mm")
            rr.log(f"metrics/indentation/{name}", rr.Scalar(value))
            if key != "ground_truth_mm" and truth is not None:
                rr.log(f"metrics/indentation_error/{name}", rr.Scalar(value - truth))

    def log_frame(self, t: int, tracks_t: np.ndarray, visible_t: np.ndarray, colors: np.ndarray) -> None:
        import rerun as rr

        self._log_indentation(t)
        clip = self.clip
        H, W = clip.rgbs.shape[-2:]
        for view, serial in enumerate(clip.camera_serials):
            entity = camera_entity(view, serial)
            rgb = np.ascontiguousarray(clip.rgbs[view, t].transpose(1, 2, 0))
            rr.log(f"{entity}/rgb", rr.Image(rgb).compress(jpeg_quality=self.jpeg_quality))

            E = clip.extrinsics_w2c_m[view, t]
            cam = tracks_t[visible_t] @ E[:3, :3].T + E[:3, 3]
            uvw = cam @ clip.intrinsics[view, t].T
            with np.errstate(divide="ignore", invalid="ignore"):
                uv = uvw[:, :2] / uvw[:, 2:3]
            inside = (cam[:, 2] > 0) & np.isfinite(uv).all(axis=1)
            inside &= (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            rr.log(f"{entity}/tracks", rr.Points2D(uv[inside], colors=colors[visible_t][inside], radii=2.0))

            if self.depths_m is not None:
                step = self.depth_downsample
                depth_mm = np.clip(self.depths_m[view, t, ::step, ::step] * 1000.0, 0, 65535).astype(np.uint16)
                rr.log(
                    f"views/camera_{view}_{serial}",
                    rr.DepthImage(
                        depth_mm, meter=1000.0, colormap=rr.components.Colormap.Turbo,
                        depth_range=self.pred_range_mm,
                    ),
                )

        d455 = self.d455
        if d455 is None:
            return
        frame = int(clip.frame_indices[t])
        color_path = d455.color_paths.get(frame)
        if color_path is not None:
            rr.log(f"{D455_COLOR}/image", rr.EncodedImage(path=color_path))
        depth_path = d455.depth_paths.get(frame)
        if depth_path is None:
            return
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return
        step = self.depth_downsample
        depth_mm = raw[::step, ::step].astype(np.float32) * (d455.depth_scale_m * 1000.0)
        corr = d455.depth_correction
        if corr:  # same correction as dataset/heightmap_labels.py
            valid = depth_mm > 0
            depth_mm[valid] = (depth_mm[valid] - corr["offset_mm"]) / (1.0 + corr["scale_error"])
        d455_range_mm = tuple(1000.0 * v for v in d455.depth_range_m)
        rr.log(
            f"{D455_DEPTH}/depth",
            rr.DepthImage(
                np.clip(depth_mm, 0, 65535).astype(np.uint16), meter=1000.0,
                colormap=rr.components.Colormap.Turbo, depth_range=d455_range_mm,
            ),
        )
        if self.depths_m is None or d455.depth_pose is None or not d455.depth_intrinsics:
            return
        recon_mm = self._recon_depth_in_d455(t, depth_mm.shape)
        rr.log(
            "views/d455_recon_depth",
            rr.DepthImage(
                np.clip(recon_mm, 0, 65535).astype(np.uint16), meter=1000.0,
                colormap=rr.components.Colormap.Turbo, depth_range=d455_range_mm,
            ),
        )
        both = (depth_mm > 0) & (recon_mm > 0)
        error_mm = np.where(both, recon_mm - depth_mm, 0.0)
        # Shown as a 0-20 mm "depth" map so hovering reads |error| (in metres).
        rr.log(
            "views/d455_abs_error",
            rr.DepthImage(
                np.clip(np.abs(error_mm), 0, 65535).astype(np.uint16), meter=1000.0,
                colormap=rr.components.Colormap.Turbo, depth_range=(0.0, 20.0),
            ),
        )
        if both.sum() >= 50:
            diff = error_mm[both]
            rr.log("metrics/d455_vs_recon/median_abs_mm", rr.Scalar(float(np.median(np.abs(diff)))))
            rr.log("metrics/d455_vs_recon/p90_abs_mm", rr.Scalar(float(np.percentile(np.abs(diff), 90))))
            rr.log("metrics/d455_vs_recon/median_signed_mm", rr.Scalar(float(np.median(diff))))

    def _recon_depth_in_d455(self, t: int, shape: tuple[int, int]) -> np.ndarray:
        """Z-buffer the fused reconstructed depth (all RGB views) into the D455 depth camera, in mm."""
        clip, intr = self.clip, self.d455.depth_intrinsics
        R_d, t_d = self.d455.depth_pose
        scale = 1.0 / self.depth_downsample
        h, w = shape
        zbuf = np.full(h * w, np.inf, dtype=np.float64)
        H, W = self.depths_m.shape[-2:]
        if self._pixel_rays is None:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            self._pixel_rays = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
        for view in range(self.depths_m.shape[0]):
            z = self.depths_m[view, t].ravel().astype(np.float64)
            ok = z > 0
            cam = (np.linalg.inv(clip.intrinsics[view, t]) @ self._pixel_rays[ok].T).T * z[ok, None]
            E = clip.extrinsics_w2c_m[view, t].astype(np.float64)
            world = (cam - E[:3, 3]) @ E[:3, :3]
            d = world @ R_d.T + t_d
            front = d[:, 2] > 1e-3
            d = d[front]
            u = np.round(d[:, 0] / d[:, 2] * intr["fx"] * scale + intr["ppx"] * scale).astype(np.int64)
            v = np.round(d[:, 1] / d[:, 2] * intr["fy"] * scale + intr["ppy"] * scale).astype(np.int64)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            np.minimum.at(zbuf, v[inside] * w + u[inside], d[inside, 2] * 1000.0)
        zbuf[~np.isfinite(zbuf)] = 0.0
        return zbuf.reshape(h, w)
