#!/usr/bin/env python3
"""How accurate is a reconstruction? Compare it with things whose answer is known.

    python scripts/eval_accuracy.py --run-dir outputs/<name>/seconds_<a>_<b>_target_<N>_da3 \
        --static 0:2.5 --board 3:6 --press 8:30 --press-xy 52,-70 --ring 5,9

Reads what `run_test_session.py` wrote (`run.json`, `depths_<backend>_m.npz`),
the session's `calibration.json` and `multimodal_video_alignment.csv`, and the
session videos (for the board test only). Every time range is in video seconds,
the same unit as run_test_session's --start/--end, and must lie inside the
reconstructed clip. Each test is optional; give the segments you recorded.

Four tests, in the order they should be recorded in one capture:

static   Nothing moves. The same surface point must get the same depth in every
         frame, so the per-pixel spread over time is the reconstruction's noise
         floor. A deformation smaller than this cannot be measured.

plane    The empty platform (soft object removed) is world z = 0 by
         construction of the calibration. Reports where the reconstructed
         surface sits and how flat it is, per camera. Give --box so only the
         platform is inside it.

board    The ChArUco board lying still. Its corners are detected in the full-
         resolution videos and triangulated with the calibration (~0.16 mm),
         independently of the depth network; each camera's depth is then read
         at those corners and compared. This is a per-camera depth error in mm,
         with its sign, which also exposes a scale error (error growing with
         distance).

press    The Mark-10 pushes an indenter into the object in steps. The crosshead
         travel since contact (force_distance in the alignment table) is the
         indentation. The reconstructed surface in a ring just outside the
         indenter (--ring, mm) is compared with it frame by frame. With a
         flat-ended punch the surface at its rim moves with the punch, so the
         slope should be close to 1; a slope well below 1 means the
         reconstruction smooths the dent away.

    python scripts/eval_accuracy.py --selftest     # synthetic data, known errors

Units: millimetres throughout the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_BOARD = PROJECT_ROOT.parent / "calibration" / "board" / "board.json"


# ------------------------------------------------------------------ geometry
class Rig:
    """Calibration in mm, plus intrinsics scaled to the depth maps' size.

    run_test_session undistorts each frame with the calibrated K kept as the
    new camera matrix, then resizes; so a depth pixel (u, v) at tracking size
    is a pinhole projection through K scaled by (W/w0, H/h0), no distortion.
    """

    def __init__(self, calibration: dict, depth_size: tuple[int, int]):
        width, height = depth_size
        self.cameras = sorted(calibration["cameras"], key=lambda c: c["index"])
        self.R = [np.asarray(c["R_world_to_cam"], np.float64) for c in self.cameras]
        self.t = [np.asarray(c["t_world_to_cam"], np.float64).reshape(3) for c in self.cameras]
        self.K_full = [np.asarray(c["K"], np.float64) for c in self.cameras]
        self.dist = [np.asarray(c["dist"], np.float64).reshape(-1) for c in self.cameras]
        self.K = []
        for camera, K in zip(self.cameras, self.K_full):
            w0, h0 = camera["image_size"]
            scaled = K.copy()
            scaled[0, :] *= width / w0
            scaled[1, :] *= height / h0
            self.K.append(scaled)
        self.size = (width, height)

    def backproject(self, view: int, depth_m: np.ndarray):
        """Depth map (m) -> world points (mm) and the flat pixel indices used."""
        height, width = depth_m.shape
        v, u = np.nonzero(np.isfinite(depth_m) & (depth_m > 0))
        z = depth_m[v, u].astype(np.float64) * 1000.0
        K = self.K[view]
        x = (u - K[0, 2]) / K[0, 0] * z
        y = (v - K[1, 2]) / K[1, 1] * z
        camera = np.stack([x, y, z], axis=1)
        world = (camera - self.t[view]) @ self.R[view]
        return world, v * width + u

    def to_camera(self, view: int, world_mm: np.ndarray) -> np.ndarray:
        return world_mm @ self.R[view].T + self.t[view]

    def project(self, view: int, world_mm: np.ndarray):
        """World (mm) -> (u, v) at depth-map size and the true depth (mm)."""
        camera = self.to_camera(view, world_mm)
        K = self.K[view]
        u = K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2]
        v = K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]
        return np.stack([u, v], axis=1), camera[:, 2]


def in_box(points: np.ndarray, box) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = box
    return ((points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0)
            & (points[:, 1] <= y1) & (points[:, 2] >= z0) & (points[:, 2] <= z1))


def sample_depth(depth_m: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear depth (mm) at sub-pixel positions; NaN unless all 4 taps are valid.

    Requiring all four taps keeps a corner on a depth edge from being read as
    a blend of foreground and background.
    """
    height, width = depth_m.shape
    out = np.full(len(uv), np.nan)
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (v >= 0) & (u < width - 1) & (v < height - 1)
    u0 = np.floor(u[inside]).astype(int)
    v0 = np.floor(v[inside]).astype(int)
    du, dv = u[inside] - u0, v[inside] - v0
    taps = np.stack([depth_m[v0, u0], depth_m[v0, u0 + 1],
                     depth_m[v0 + 1, u0], depth_m[v0 + 1, u0 + 1]], axis=1)
    good = (taps > 0).all(axis=1) & np.isfinite(taps).all(axis=1)
    value = (taps[:, 0] * (1 - du) * (1 - dv) + taps[:, 1] * du * (1 - dv)
             + taps[:, 2] * (1 - du) * dv + taps[:, 3] * du * dv) * 1000.0
    filled = np.where(good, value, np.nan)
    out[np.flatnonzero(inside)] = filled
    return out


def triangulate(rig: Rig, observations: dict[int, np.ndarray]) -> np.ndarray:
    """Linear triangulation of one point from full-resolution, distorted pixels."""
    import cv2

    rows = []
    for view, pixel in observations.items():
        normalised = cv2.undistortPoints(
            np.asarray(pixel, np.float64).reshape(1, 1, 2), rig.K_full[view],
            rig.dist[view]).reshape(2)
        P = np.hstack([rig.R[view], rig.t[view].reshape(3, 1)])
        rows.append(normalised[0] * P[2] - P[0])
        rows.append(normalised[1] * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    point = vt[-1]
    return point[:3] / point[3]


def stats(values) -> dict:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "median": float(np.median(values)),
            "p90_abs": float(np.percentile(np.abs(values), 90)),
            "mean": float(values.mean()), "std": float(values.std())}


# --------------------------------------------------------------------- tests
def test_static(rig: Rig, depths: np.ndarray, frames: list[int], box) -> dict:
    """Per-pixel spread of depth over frames in which nothing moved."""
    result = {}
    for view in range(depths.shape[0]):
        stack = depths[view, frames].astype(np.float64) * 1000.0      # (T, H, W) mm
        valid = (stack > 0).all(axis=0) & np.isfinite(stack).all(axis=0)
        world, flat = rig.backproject(view, np.where(valid, depths[view, frames[0]], 0))
        keep = np.zeros(valid.size, bool)
        keep[flat[in_box(world, box)]] = True
        keep = keep.reshape(valid.shape)
        if keep.sum() < 50:
            result[view] = {"n": int(keep.sum())}
            continue
        spread = stack[:, keep].std(axis=0, ddof=1)
        step = np.abs(np.diff(stack[:, keep], axis=0)).ravel()
        result[view] = {"pixels": int(keep.sum()),
                        "temporal_std_median": float(np.median(spread)),
                        "temporal_std_p90": float(np.percentile(spread, 90)),
                        "frame_to_frame_median": float(np.median(step))}
    return result


def test_plane(rig: Rig, depths: np.ndarray, frames: list[int], box, band: float) -> dict:
    """Where the platform (world z = 0) is reconstructed, and how flat."""
    result, everything = {}, []
    for view in range(depths.shape[0]):
        points = []
        for t in frames:
            world, _ = rig.backproject(view, depths[view, t])
            world = world[in_box(world, box) & (np.abs(world[:, 2]) <= band)]
            points.append(world)
        points = np.concatenate(points) if points else np.zeros((0, 3))
        if len(points) < 50:
            result[view] = {"n": int(len(points))}
            continue
        A = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
        coef, *_ = np.linalg.lstsq(A, points[:, 2], rcond=None)
        residual = points[:, 2] - A @ coef
        tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
        result[view] = {"points": int(len(points)),
                        "z_median": float(np.median(points[:, 2])),
                        "z_p90_abs": float(np.percentile(np.abs(points[:, 2]), 90)),
                        "flatness_rms": float(np.sqrt(np.mean(residual ** 2))),
                        "tilt_deg": tilt}
        everything.append(np.median(points[:, 2]))
    if len(everything) >= 2:
        result["between_cameras_z_spread"] = float(np.max(everything) - np.min(everything))
    return result


def board_corners_world(rig: Rig, detections: list[dict[int, dict[int, np.ndarray]]],
                        min_views: int = 2) -> dict[int, np.ndarray]:
    """Triangulate each corner id in each frame, then average over frames.

    `detections[k][view][corner_id]` is a full-resolution pixel.
    """
    per_id: dict[int, list[np.ndarray]] = {}
    for frame in detections:
        ids = set()
        for corners in frame.values():
            ids |= set(corners)
        for corner in ids:
            seen = {view: corners[corner] for view, corners in frame.items() if corner in corners}
            if len(seen) >= min_views:
                per_id.setdefault(corner, []).append(triangulate(rig, seen))
    return {corner: np.median(np.stack(points), axis=0) for corner, points in per_id.items()}


def test_board(rig: Rig, depths: np.ndarray, frames: list[int],
               corners_world: dict[int, np.ndarray],
               seen_by: dict[int, set[int]]) -> dict:
    """Depth at triangulated board corners vs the corners' true depth."""
    ids = sorted(corners_world)
    world = np.stack([corners_world[i] for i in ids]) if ids else np.zeros((0, 3))
    result = {}
    for view in range(depths.shape[0]):
        errors, ranges = [], []
        mask = np.array([view in seen_by.get(i, set()) for i in ids], bool)
        if not mask.any():
            result[view] = {"n": 0}
            continue
        uv, true_depth = rig.project(view, world[mask])
        for t in frames:
            measured = sample_depth(depths[view, t], uv)
            errors.append(measured - true_depth)
            ranges.append(true_depth)
        errors = np.concatenate(errors)
        ranges = np.concatenate(ranges)
        good = np.isfinite(errors)
        entry = stats(errors)
        if good.sum() >= 5:
            entry["relative_median_pct"] = float(np.median(errors[good] / ranges[good]) * 100)
            entry["range_mm"] = [float(ranges[good].min()), float(ranges[good].max())]
        result[view] = entry
    return result


def rim_distance(xy: np.ndarray, centre_xy, rect=None, angle_deg: float = 0.0) -> np.ndarray:
    """Distance (mm) in the xy plane from the indenter's outline.

    Round indenter (rect None): distance from the axis, so --ring is a radius
    range. Rectangular indenter (rect = (length, width), long side at
    angle_deg from world +x): distance from the rectangle's edge, 0 on and
    inside it, so --ring is how far outside the edge to read the surface.
    """
    d = xy - np.asarray(centre_xy, np.float64)
    if rect is None:
        return np.hypot(d[:, 0], d[:, 1])
    a = np.radians(angle_deg)
    along = np.abs(d[:, 0] * np.cos(a) + d[:, 1] * np.sin(a)) - rect[0] / 2
    across = np.abs(-d[:, 0] * np.sin(a) + d[:, 1] * np.cos(a)) - rect[1] / 2
    return np.hypot(np.maximum(along, 0), np.maximum(across, 0))


def test_press(rig: Rig, depths: np.ndarray, frames: list[int], centre_xy, ring,
               travel_mm: np.ndarray, load: np.ndarray, contact_load: float, box,
               rect=None, angle_deg: float = 0.0, above_mm: float = 2.0) -> dict:
    """Reconstructed dent at the indenter rim vs crosshead travel since contact."""
    r_in, r_out = ring
    views = depths.shape[0]
    loaded = load >= contact_load
    if not loaded.any() or loaded.all():
        return {"error": f"no frame crosses the contact load {contact_load:g}; "
                         "set --contact-load or check the force import"}
    band = [[None] * views for _ in frames]
    for k, t in enumerate(frames):
        for view in range(views):
            world, _ = rig.backproject(view, depths[view, t])
            world = world[in_box(world, box)]
            radius = rim_distance(world[:, :2], centre_xy, rect, angle_deg)
            band[k][view] = world[(radius >= r_in) & (radius <= r_out), 2]
    # A press only lowers the surface. Seen obliquely, the indenter's side
    # faces fall inside the band and stand far above it (on 2026-10-09 most
    # band points at the deepest hold were the indenter), so anything more
    # than above_mm over the unloaded surface before the first contact is
    # the indenter, its rod or a hand, and is dropped.
    first = int(np.argmax(loaded))
    ceiling = np.full(views, np.inf)
    for view in range(views):
        rest = [np.median(band[k][view]) for k in range(first) if band[k][view].size >= 10]
        if rest:
            ceiling[view] = np.median(rest) + above_mm
    # Height per camera, not pooled: cameras disagree by a constant offset (the
    # board test measures it), and a pooled median drifts as the mix of
    # cameras seeing the ring changes during the press.
    heights = np.full((len(frames), views), np.nan)
    for k in range(len(frames)):
        for view in range(views):
            values = band[k][view]
            values = values[values <= ceiling[view]]
            if values.size >= 10:
                heights[k, view] = np.median(values)
    # Each press is measured from its own start: a soft object does not fully
    # recover between presses (on 2026-10-08 the contact height crept down by
    # 0.9 mm over four presses), so both the zero of the travel and the
    # unloaded surface height are taken from the unloaded frames just before
    # that press.
    sign = 1.0
    indentation = np.zeros(len(frames))
    dent_by_view = np.full_like(heights, np.nan)
    presses = 0
    last_unloaded = None
    rest: list[int] = []
    k = 0
    while k < len(frames):
        if not loaded[k]:
            rest.append(k)
            last_unloaded = k
            k += 1
            continue
        end = k
        while end < len(frames) and loaded[end]:
            end += 1
        if last_unloaded is not None:
            zero = travel_mm[last_unloaded]
            with np.errstate(all="ignore"):
                reference = np.nanmedian(heights[rest[-3:]], axis=0)
            run = np.arange(k, end)
            indentation[run] = travel_mm[run] - zero
            dent_by_view[run] = reference[None, :] - heights[run]
            presses += 1
        rest = []
        k = end
    # The stand may count travel up or down; pressing means indentation grows
    # while the load grows.
    if np.nansum(indentation[loaded] * load[loaded]) < 0:
        sign = -1.0
    indentation = np.maximum(sign * indentation, 0.0)
    for k in np.flatnonzero(~loaded):
        dent_by_view[k] = 0.0
    contact = int(np.argmax(loaded))
    with np.errstate(all="ignore"):
        dent = np.nanmedian(dent_by_view, axis=1)
    dent[~loaded] = np.nan          # only loaded frames are evidence
    good = np.isfinite(dent)
    if good.sum() < 3:
        return {"error": "too few frames with surface points in the ring; widen --ring "
                         "or check --press-xy"}
    slope = float(np.sum(dent[good] * indentation[good]) /
                  max(np.sum(indentation[good] ** 2), 1e-9))
    residual = dent[good] - slope * indentation[good]
    error = dent[good] - indentation[good]
    per_view = {}
    for view in range(views):
        ok = np.isfinite(dent_by_view[:, view])
        if ok.sum() >= 3:
            per_view[view] = float(np.sum(dent_by_view[ok, view] * indentation[ok]) /
                                   max(np.sum(indentation[ok] ** 2), 1e-9))
    return {"frames": len(frames), "presses": presses,
            "contact_clip_index": int(frames[contact]),
            "max_indentation": float(indentation.max()),
            "slope_dent_per_travel": slope,
            "slope_per_camera": per_view,
            "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "error_vs_travel": stats(error),
            "per_frame": [{"clip_index": int(f), "travel": round(float(i), 3),
                           "dent": None if not np.isfinite(d) else round(float(d), 3)}
                          for f, i, d in zip(frames, indentation, dent)]}


def locate_dent(rig: Rig, depths: np.ndarray, reference: list[int], loaded: int,
                box, cell: float = 1.0, with_angle: bool = False):
    """Where the surface went down the most: the indenter's axis, roughly.

    Fused height maps (median z per 1 mm cell over all cameras) before contact
    and at peak load are differenced. The indenter itself occludes the bottom
    of the dent and stands above the surface, so the deepest cells form a ring
    around it; the centroid of the cells within half the maximum depression is
    the axis. With `with_angle`, also returns the direction of their principal
    axis (degrees from world +x, in [0, 180)): a rectangular indenter's long side.
    """
    x0, x1, y0, y1 = box[:4]
    nx, ny = int((x1 - x0) / cell), int((y1 - y0) / cell)

    def height_map(frames):
        points = []
        for t in frames:
            for view in range(depths.shape[0]):
                world, _ = rig.backproject(view, depths[view, t])
                points.append(world[in_box(world, box)])
        points = np.concatenate(points)
        ix = ((points[:, 0] - x0) / cell).astype(int).clip(0, nx - 1)
        iy = ((points[:, 1] - y0) / cell).astype(int).clip(0, ny - 1)
        flat = iy * nx + ix
        order = np.argsort(flat)
        flat, z = flat[order], points[order, 2]
        cells, start = np.unique(flat, return_index=True)
        grid = np.full(nx * ny, np.nan)
        for cell_id, segment in zip(cells, np.split(z, start[1:])):
            if segment.size >= 3:
                grid[cell_id] = np.median(segment)
        return grid.reshape(ny, nx)

    with np.errstate(all="ignore"):
        dent = height_map(reference) - height_map([loaded])
    if not np.isfinite(dent).any():
        return None
    deepest = np.nanmax(dent)
    if deepest <= 0:
        return None
    rows, cols = np.nonzero(np.nan_to_num(dent, nan=-np.inf) >= 0.5 * deepest)
    centre = [float(x0 + (cols.mean() + 0.5) * cell), float(y0 + (rows.mean() + 0.5) * cell)]
    if not with_angle:
        return centre
    if rows.size < 3:
        return centre, 0.0
    _, vectors = np.linalg.eigh(np.cov(np.stack([cols, rows]).astype(np.float64)))
    major = vectors[:, -1]
    return centre, float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)


# ------------------------------------------------------------------ inputs
def time_range(text: str) -> tuple[float, float]:
    a, b = (float(v) for v in text.split(":"))
    if b <= a:
        raise argparse.ArgumentTypeError(f"empty range {text}")
    return a, b


def floats(count: int):
    def parse(text: str):
        values = [float(v) for v in text.split(",")]
        if len(values) != count:
            raise argparse.ArgumentTypeError(f"expected {count} comma-separated numbers")
        return values
    return parse


def frames_in(times: np.ndarray, window) -> list[int]:
    picked = np.flatnonzero((times >= window[0]) & (times <= window[1])).tolist()
    if not picked:
        raise SystemExit(f"no reconstructed frame lies in {window[0]:g}-{window[1]:g} s; "
                         f"the clip covers {times.min():g}-{times.max():g} s")
    return picked


def load_alignment(session: Path) -> dict[int, dict]:
    path = session / "multimodal_video_alignment.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {int(row["video_frame_index"]): row for row in csv.DictReader(handle)}


def detect_board(session: Path, video_frames: list[int], board_json: Path,
                 camera_count: int) -> list[dict[int, dict[int, np.ndarray]]]:
    import cv2

    spec = json.loads(board_json.read_text(encoding="utf-8"))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    board = cv2.aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]),
                                   float(spec["square_mm"]), float(spec["marker_mm"]), dictionary)
    board.setLegacyPattern(bool(spec.get("legacy_pattern", False)))
    detector = cv2.aruco.CharucoDetector(board)
    export = json.loads((session / "multimodal_video_export.json").read_text(encoding="utf-8"))
    out = [dict() for _ in video_frames]
    for view in range(camera_count):
        capture = cv2.VideoCapture(str(session / export["videos"][view]))
        try:
            for k, frame_index in enumerate(video_frames):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, image = capture.read()
                if not ok:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                corners, ids, _, _ = detector.detectBoard(gray)
                if ids is not None and len(ids) >= 6:
                    out[k][view] = {int(i): c for i, c in
                                    zip(np.asarray(ids).ravel(), np.asarray(corners).reshape(-1, 2))}
        finally:
            capture.release()
    return out


# ------------------------------------------------------------------ report
def print_report(report: dict) -> None:
    def row(name, entry, keys):
        if "n" in entry and len(entry) == 1:
            print(f"    cam{name}: not enough data ({entry['n']})")
            return
        print(f"    cam{name}: " + "  ".join(
            f"{label} {entry[key]:+.3f}" if isinstance(entry.get(key), float) else
            f"{label} {entry.get(key)}" for label, key in keys))

    if "static" in report:
        print("\nstatic -- noise floor (mm): depth spread over frames, nothing moving")
        for view, entry in report["static"].items():
            row(view, entry, [("std median", "temporal_std_median"),
                              ("p90", "temporal_std_p90"),
                              ("frame-to-frame", "frame_to_frame_median"),
                              ("pixels", "pixels")])
    if "plane" in report:
        print("\nplane -- empty platform, should be z = 0 (mm)")
        for view, entry in report["plane"].items():
            if view == "between_cameras_z_spread":
                print(f"    cameras disagree on the platform height by {entry:.3f} mm")
                continue
            row(view, entry, [("z median", "z_median"), ("|z| p90", "z_p90_abs"),
                              ("flatness rms", "flatness_rms"), ("tilt deg", "tilt_deg")])
    if "board" in report:
        board = report["board"]
        print(f"\nboard -- depth error at {board['corners']} triangulated corners "
              "(mm, + = reconstructed too far)")
        for view, entry in board["per_camera"].items():
            row(view, entry, [("median", "median"), ("|err| p90", "p90_abs"),
                              ("std", "std"), ("rel %", "relative_median_pct"), ("n", "n")])
    if "press" in report:
        press = report["press"]
        print("\npress -- dent at the indenter rim vs crosshead travel since contact (mm)")
        if "error" in press:
            print(f"    {press['error']}")
        else:
            print(f"    max indentation {press['max_indentation']:.3f}   slope dent/travel "
                  f"{press['slope_dent_per_travel']:.3f}   residual rms "
                  f"{press['residual_rms']:.3f}")
            error = press["error_vs_travel"]
            print(f"    dent - travel: median {error['median']:+.3f}  |err| p90 "
                  f"{error['p90_abs']:.3f}")
            print("    slope per camera: " + "  ".join(
                f"cam{k} {v:.3f}" for k, v in press["slope_per_camera"].items()))


# ----------------------------------------------------------------- selftest
def selftest() -> int:
    """Synthetic rig with known errors; every test must recover them."""
    import cv2  # noqa: F401  (needed by triangulate)

    calibration_path = PROJECT_ROOT.parent / "calibration" / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    rig = Rig(calibration, (512, 384))
    rng = np.random.default_rng(0)
    V, T, H, W = 4, 12, 384, 512
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    bias = [0.0, 0.5, -0.3, 1.0]       # mm, per camera, planted
    noise = 0.2                        # mm
    centre = np.array([50.0, -60.0])
    dent_depth = np.linspace(0, 4, T)  # mm, the "travel"

    def plane_depth(view, height_fn):
        """Depth map of the surface z = height(x, y), by fixed-point iteration."""
        K, R, t = rig.K[view], rig.R[view], rig.t[view]
        rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        origin = -R.T @ t
        direction = rays @ R                      # world direction per unit camera z
        z_surface = np.zeros_like(u)
        for _ in range(6):
            s = (z_surface - origin[2]) / direction[..., 2]
            x = origin[0] + s * direction[..., 0]
            y = origin[1] + s * direction[..., 1]
            z_surface = height_fn(x, y)
        depth = (z_surface - origin[2]) / direction[..., 2]
        return depth

    depths = np.zeros((V, T, H, W), np.float32)
    for view in range(V):
        for t in range(T):
            d = dent_depth[t]

            def height(x, y, d=d):
                r = np.hypot(x - centre[0], y - centre[1])
                return -d * np.exp(-(r / 12.0) ** 2)
            depth = plane_depth(view, height)
            depth = depth + bias[view] + rng.normal(0, noise, depth.shape)
            depths[view, t] = np.where(depth > 0, depth / 1000.0, 0)

    failures = []

    def expect(ok, message):
        print(f"  [{'ok  ' if ok else 'FAIL'}] {message}")
        if not ok:
            failures.append(message)

    box = (-150, 250, -250, 150, -30, 30)
    noisy = depths[:, :1].repeat(6, axis=1) + rng.normal(0, noise / 1000.0, (V, 6, H, W)).astype(np.float32)
    static = test_static(rig, noisy, list(range(6)), box)
    got = static[0]["temporal_std_median"]
    expect(abs(got - noise) < 0.05, f"static noise {got:.3f} mm (planted {noise})")

    flat = depths[:, :1].copy()
    plane = test_plane(rig, flat, [0], (centre[0] + 25, 250, -250, 150, -30, 30), 10.0)
    for view in range(V):
        entry = plane[view]
        # A depth bias b along an oblique ray shifts the surface by b*cos(angle)
        # in z, towards the camera (up) when b < 0.
        expect(entry["points"] > 100 and abs(entry["z_median"]) < 1.2,
               f"plane cam{view}: z {entry['z_median']:+.3f} mm, flatness "
               f"{entry['flatness_rms']:.3f} mm")

    # Board: corners on the platform at z = 0, observed in full-res pixels with
    # 0.1 px noise; triangulated and compared with the biased depth maps.
    corner_world = np.array([[x, y, 0.0] for x in np.arange(0, 100, 9.0)
                             for y in np.arange(-130, -10, 9.0)])
    detections = [dict() for _ in range(3)]
    import cv2
    for view in range(V):
        rvec, _ = cv2.Rodrigues(rig.R[view])
        pixels, _ = cv2.projectPoints(corner_world, rvec, rig.t[view], rig.K_full[view], rig.dist[view])
        pixels = pixels.reshape(-1, 2)
        w0, h0 = rig.cameras[view]["image_size"]
        for k in range(3):
            noisy_px = pixels + rng.normal(0, 0.1, pixels.shape)
            inside = (noisy_px[:, 0] > 0) & (noisy_px[:, 1] > 0) & (noisy_px[:, 0] < w0) & (noisy_px[:, 1] < h0)
            detections[k][view] = {i: noisy_px[i] for i in np.flatnonzero(inside)}
    corners = board_corners_world(rig, detections)
    seen_by = {}
    for frame in detections:
        for view, found in frame.items():
            for corner in found:
                seen_by.setdefault(corner, set()).add(view)
    tri_error = np.median([np.linalg.norm(corners[i] - corner_world[i]) for i in corners])
    expect(tri_error < 0.1, f"triangulated corners within {tri_error:.3f} mm of truth")
    board = test_board(rig, depths[:, :1], [0], corners, seen_by)
    for view in range(V):
        got = board[view].get("median", np.nan)
        expect(abs(got - bias[view]) < 0.1, f"board cam{view}: depth error {got:+.3f} mm "
               f"(planted {bias[view]:+.1f})")

    found = locate_dent(rig, depths, [0], T - 1, (-150, 250, -250, 150, -30, 30))
    miss = float(np.hypot(*(np.asarray(found) - centre))) if found else np.inf
    expect(miss < 1.5, f"dent located {miss:.2f} mm from the true indenter axis")

    load = np.where(dent_depth > 0, 1 + dent_depth, 0.0)
    travel = 100.0 - dent_depth          # crosshead counting down while pressing
    press = test_press(rig, depths, list(range(T)), centre, (0.0, 3.0), travel, load, 0.5,
                       (-150, 250, -250, 150, -30, 30))
    expect("error" not in press and abs(press["slope_dent_per_travel"] - 0.97) < 0.06,
           f"press slope {press.get('slope_dent_per_travel', float('nan')):.3f} "
           "(surface inside r=3 mm of a 12 mm Gaussian dent: ~0.97)")

    # Rectangular punch, 35 x 10 mm at 30 deg: flat-bottomed dent, falling off
    # outside the edge as exp(-(edge distance / 6 mm)^2).
    rect, rect_angle = (35.0, 10.0), 30.0
    rect_depths = np.zeros_like(depths)
    for view in range(V):
        for t in range(T):
            def height(x, y, d=dent_depth[t]):
                xy = np.stack([x.ravel(), y.ravel()], axis=1)
                edge = rim_distance(xy, centre, rect, rect_angle).reshape(x.shape)
                return -d * np.exp(-(edge / 6.0) ** 2)
            depth = plane_depth(view, height) + rng.normal(0, noise, (H, W))
            rect_depths[view, t] = np.where(depth > 0, depth / 1000.0, 0)
    found = locate_dent(rig, rect_depths, [0], T - 1, box, with_angle=True)
    turn = abs((found[1] - rect_angle + 90) % 180 - 90) if found else np.inf
    expect(turn < 5, f"rectangular dent long side at {found[1] if found else np.nan:.1f} deg "
           f"(planted {rect_angle:g})")
    press = test_press(rig, rect_depths, list(range(T)), centre, (0.5, 1.5), travel, load, 0.5,
                       box, rect, rect_angle)
    expect("error" not in press and abs(press["slope_dent_per_travel"] - 0.97) < 0.04,
           f"rectangular press slope {press.get('slope_dent_per_travel', float('nan')):.3f} "
           "(band 0.5-1.5 mm outside the edge: ~0.97)")
    round_press = test_press(rig, rect_depths, list(range(T)), centre, (18.5, 21.5), travel,
                             load, 0.5, box)
    expect("error" not in round_press and round_press["slope_dent_per_travel"] < 0.8,
           f"a round ring outside the half-diagonal reads slope "
           f"{round_press.get('slope_dent_per_travel', float('nan')):.3f}: biased low, as expected")
    print(f"\n{'SELFTEST PASSED' if not failures else f'{len(failures)} FAILED'}")
    return 1 if failures else 0


# ---------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, help="one run_test_session output folder")
    parser.add_argument("--session-dir", type=Path,
                        help="capture session; default: the one in run.json")
    parser.add_argument("--calibration", type=Path,
                        help="default: <session>/calibration.json")
    parser.add_argument("--box", type=floats(6), default=[-200, 200, -200, 200, -30, 200],
                        metavar="X0,X1,Y0,Y1,Z0,Z1",
                        help="world region (mm) the tests look at; keep the background out")
    parser.add_argument("--static", type=time_range, metavar="A:B")
    parser.add_argument("--plane", type=time_range, metavar="A:B")
    parser.add_argument("--plane-band", type=float, default=15.0,
                        help="only points within this many mm of z = 0 count as platform")
    parser.add_argument("--board", type=time_range, metavar="A:B")
    parser.add_argument("--board-json", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--press", type=time_range, metavar="A:B")
    parser.add_argument("--press-xy", type=floats(2), metavar="X,Y",
                        help="world position (mm) of the indenter axis; default: found "
                             "from where the surface went down most at peak load")
    parser.add_argument("--ring", type=floats(2), default=[5.0, 9.0], metavar="R_IN,R_OUT",
                        help="annulus (mm) around the indenter axis to read the surface in; "
                             "start just outside the indenter's radius. With --press-rect: "
                             "band this far outside the rectangle's edge, e.g. 1,4")
    parser.add_argument("--press-rect", type=floats(2), metavar="LENGTH,WIDTH",
                        help="rectangular indenter face (mm); --ring is then measured from its edge")
    parser.add_argument("--press-angle", type=float, metavar="DEG",
                        help="long side of --press-rect, degrees from world +x; default: "
                             "the principal axis of the dent at peak load")
    parser.add_argument("--above-mm", type=float, default=2.0,
                        help="press: drop band points this far above the unloaded surface "
                             "(the indenter seen side-on)")
    parser.add_argument("--distance-unit", choices=["mm", "in"], default="mm",
                        help="unit of force_distance in the Mark-10 export")
    parser.add_argument("--contact-load", type=float, default=None,
                        help="load above the unloaded level that counts as contact; "
                             "default 5%% of the peak")
    parser.add_argument("--report", type=Path, help="also write the numbers as JSON")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if args.run_dir is None:
        parser.error("--run-dir is required")
    if not any([args.static, args.plane, args.board, args.press]):
        parser.error("give at least one of --static / --plane / --board / --press")

    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    session = args.session_dir or Path(run["session_dir"])
    calibration = json.loads((args.calibration or session / "calibration.json")
                             .read_text(encoding="utf-8"))
    depth_files = sorted(args.run_dir.glob(f"depths_{run['depth_backend']}_m.npz"))
    if not depth_files:
        raise SystemExit(f"no depths_{run['depth_backend']}_m.npz in {args.run_dir}")
    data = np.load(depth_files[0])
    depths = data["depths_m"]
    frame_indices = data["frame_indices"].astype(int)
    times = np.asarray(run["times_s"], np.float64)
    rig = Rig(calibration, (depths.shape[3], depths.shape[2]))
    box = args.box

    report = {"run_dir": str(args.run_dir), "backend": run["depth_backend"],
              "depth_size": [int(depths.shape[3]), int(depths.shape[2])], "box_mm": box}
    print(f"run      {args.run_dir}  ({run['depth_backend']}, {depths.shape[1]} frames, "
          f"{depths.shape[3]}x{depths.shape[2]})")
    print(f"session  {session}")

    if args.static:
        report["static"] = test_static(rig, depths, frames_in(times, args.static), box)
    if args.plane:
        report["plane"] = test_plane(rig, depths, frames_in(times, args.plane), box,
                                     args.plane_band)
    if args.board:
        picked = frames_in(times, args.board)
        detections = detect_board(session, [int(frame_indices[k]) for k in picked],
                                  args.board_json, depths.shape[0])
        corners = board_corners_world(rig, detections)
        seen_by: dict[int, set[int]] = {}
        for frame in detections:
            for view, found in frame.items():
                for corner in found:
                    seen_by.setdefault(corner, set()).add(view)
        report["board"] = {"corners": len(corners),
                           "per_camera": test_board(rig, depths, picked, corners, seen_by)}
    if args.press:
        picked = frames_in(times, args.press)
        alignment = load_alignment(session)
        rows = [alignment.get(int(frame_indices[k]), {}) for k in picked]

        def column(name):
            values = []
            for row in rows:
                try:
                    values.append(float(row.get(name, "")))
                except ValueError:
                    values.append(np.nan)
            return np.asarray(values)
        travel = column("force_distance") * (25.4 if args.distance_unit == "in" else 1.0)
        raw_load = column("force_load")
        # The gauge is not always zeroed (2026-10-09 read -7.5 N unloaded), so the
        # unloaded level is estimated -- the median of the lowest 30 % of the
        # readings -- and removed before anything is compared with a threshold.
        finite = raw_load[np.isfinite(raw_load)]
        if finite.size:
            low = finite[finite <= np.percentile(finite, 30)]
            zero = float(np.median(low))
            load = raw_load - zero
            if np.nanmax(-load) > np.nanmax(load):   # compression read as negative
                load = -load
            if abs(zero) > 0.5:
                print(f"force gauge was not zeroed: unloaded reading {zero:+.2f}; removed")
        else:
            load = raw_load
        if not np.isfinite(travel).any():
            raise SystemExit("no force_distance in the alignment table for these frames; "
                             "run align_force_curve_only.py and re-export first")
        contact = args.contact_load if args.contact_load is not None else 0.05 * np.nanmax(load)
        keep = np.isfinite(travel) & np.isfinite(load)
        centre, angle = args.press_xy, args.press_angle
        if centre is None or (args.press_rect and angle is None):
            unloaded = [picked[i] for i in np.flatnonzero(keep & (load < contact))]
            peak = picked[int(np.nanargmax(np.where(keep, load, -np.inf)))]
            found = locate_dent(rig, depths, unloaded, peak, box, with_angle=True) \
                if unloaded else None
            if found is None:
                raise SystemExit("could not find the dent; give --press-xy X,Y (mm)"
                                 + (" and --press-angle" if args.press_rect else ""))
            if centre is None:
                centre = found[0]
                print(f"indenter axis located at ({centre[0]:.1f}, {centre[1]:.1f}) mm "
                      "-- pass --press-xy to override")
            if args.press_rect and angle is None:
                angle = found[1]
                print(f"indenter long side at {angle:.1f} deg from world +x "
                      "-- pass --press-angle to override")
        report["press"] = test_press(rig, depths, [picked[i] for i in np.flatnonzero(keep)],
                                     centre, args.ring, travel[keep], load[keep],
                                     contact, box, args.press_rect, angle or 0.0,
                                     args.above_mm)
        report["press"]["centre_xy"] = centre
        if args.press_rect:
            report["press"]["rect_mm"] = args.press_rect
            report["press"]["angle_deg"] = angle
    print_report(report)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
