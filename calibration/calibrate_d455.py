#!/usr/bin/env python3
"""Put the D455 into the calibrated world frame of the Blackfly rig.

    python calibrate_d455.py <capture session> --board board_d455/board.json

RUN WITH THE CALIBRATION ENVIRONMENT (.venv-calib). The session is an
ordinary capture (capture_3blackfly_sensor_force.py) recorded while
record_d455.py ran, then attached with align_d455.py so that
`<session>/d455/` exists. Record it with `record_d455.py --color-size
1280x720`: at the D455's 55-60 cm the colour image needs the resolution.

What to film: the A4 board in board_d455/ (25 mm squares -- the 9 mm board
is too small for the D455 to decode at its minimum range), lying or propped
STILL in 5-10 different places and tilts over the platform, ~2 s each, so
that the D455 and at least two Blackflys see it every time. Partial views are
fine for the Blackflys.

How it works: in every frame set where the board is still, each corner seen
by two or more Blackflys is triangulated with calibration.json -- that is its
position in the world frame, to ~0.2 mm. The same corners detected in the
D455 colour image give 2D-3D pairs, and one PnP over all poses gives the
D455's pose in the world. No D455 depth is used for that, so the depth can
then be checked independently: the board corners' true depth in the D455
depth frame against what the D455 measured there.

Output: <session>/d455/d455_world.json, and with --install also
calibration/d455_world.json (what downstream tools read):

    R_world_to_color, t_world_to_color (mm)   colour camera
    R_world_to_depth, t_world_to_depth (mm)   depth frame (left IR imager)
    color/depth intrinsics, depth_scale_m, residuals, depth check
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

from charuco_board import BoardSpec, make_detector

HERE = Path(__file__).resolve().parent
GENICAM_RG_TO_GRAY = cv.COLOR_BayerBG2GRAY       # see session_io: BayerRG8 is OpenCV BG


# ------------------------------------------------------------------ geometry
def load_rig(path: Path):
    calibration = json.loads(path.read_text(encoding="utf-8"))
    cams = sorted(calibration["cameras"], key=lambda c: c["index"])
    return [{"serial": c["serial"], "K": np.asarray(c["K"], float),
             "dist": np.asarray(c["dist"], float).reshape(-1),
             "R": np.asarray(c["R_world_to_cam"], float),
             "t": np.asarray(c["t_world_to_cam"], float).reshape(3)} for c in cams]


def triangulate(rig, observations: dict[int, np.ndarray]):
    """DLT from distorted full-resolution pixels; returns (point, rms_px)."""
    rows = []
    for view, pixel in observations.items():
        cam = rig[view]
        n = cv.undistortPoints(np.asarray(pixel, float).reshape(1, 1, 2),
                               cam["K"], cam["dist"]).reshape(2)
        P = np.hstack([cam["R"], cam["t"].reshape(3, 1)])
        rows += [n[0] * P[2] - P[0], n[1] * P[2] - P[1]]
    _, _, vt = np.linalg.svd(np.asarray(rows))
    point = vt[-1][:3] / vt[-1][3]
    errors = []
    for view, pixel in observations.items():
        cam = rig[view]
        rvec, _ = cv.Rodrigues(cam["R"])
        projected, _ = cv.projectPoints(point.reshape(1, 3), rvec, cam["t"], cam["K"], cam["dist"])
        errors.append(np.linalg.norm(projected.reshape(2) - np.asarray(pixel, float)))
    return point, float(np.sqrt(np.mean(np.square(errors))))


def realsense_K_dist(intrinsics: dict):
    K = np.array([[intrinsics["fx"], 0, intrinsics["ppx"]],
                  [0, intrinsics["fy"], intrinsics["ppy"]], [0, 0, 1.0]])
    coeffs = np.asarray(intrinsics.get("coeffs", [0] * 5), float)
    model = intrinsics.get("model", "")
    # librealsense "brown_conrady" uses OpenCV's k1 k2 p1 p2 k3 order. The
    # "inverse" models describe undistortion instead; their coefficients are
    # typically ~0 on D400 colour, so treat them as none rather than wrongly.
    dist = coeffs if "inverse" not in model.lower() else np.zeros(5)
    return K, dist


def depth_to_color(meta: dict):
    e = meta["depth_to_color"]
    R = np.asarray(e["rotation_col_major"], float).reshape(3, 3).T
    t = np.asarray(e["translation_m"], float) * 1000.0
    return R, t


def solve_pose(world: np.ndarray, pixels: np.ndarray, K, dist, reproj_px=4.0):
    """World (mm) <-> colour pixels -> R, t world->colour, inlier mask, rms."""
    ok, rvec, tvec, inliers = cv.solvePnPRansac(
        world.astype(np.float64), pixels.astype(np.float64), K, dist,
        reprojectionError=reproj_px, iterationsCount=2000, flags=cv.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers) < 12:
        raise SystemExit("PnP failed: too few consistent board corners in the D455 view")
    keep = np.zeros(len(world), bool)
    keep[inliers.ravel()] = True
    rvec, tvec = cv.solvePnPRefineLM(world[keep], pixels[keep], K, dist, rvec, tvec)
    projected, _ = cv.projectPoints(world[keep], rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels[keep]) ** 2, axis=1))))
    R, _ = cv.Rodrigues(rvec)
    return R, tvec.reshape(3), keep, rms


def sample_depth_mm(depth_raw: np.ndarray, scale_m: float, uv: np.ndarray) -> np.ndarray:
    h, w = depth_raw.shape
    out = np.full(len(uv), np.nan)
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
    u0, v0 = np.floor(u[inside]).astype(int), np.floor(v[inside]).astype(int)
    du, dv = u[inside] - u0, v[inside] - v0
    taps = np.stack([depth_raw[v0, u0], depth_raw[v0, u0 + 1],
                     depth_raw[v0 + 1, u0], depth_raw[v0 + 1, u0 + 1]], 1).astype(float)
    good = (taps > 0).all(1)
    value = (taps[:, 0] * (1 - du) * (1 - dv) + taps[:, 1] * du * (1 - dv)
             + taps[:, 2] * (1 - du) * dv + taps[:, 3] * du * dv) * scale_m * 1000.0
    out[np.flatnonzero(inside)] = np.where(good, value, np.nan)
    return out


def depth_check(points_world: list[np.ndarray], depths: list[np.ndarray], R_wd, t_wd,
                K_depth, scale_m) -> dict:
    """D455 depth at board corners vs their true depth in the depth frame."""
    errors, ranges = [], []
    for world, depth in zip(points_world, depths):
        camera = world @ R_wd.T + t_wd
        uv = (camera[:, :2] / camera[:, 2:3]) @ K_depth[:2, :2].T + K_depth[:2, 2]
        measured = sample_depth_mm(depth, scale_m, uv)
        errors.append(measured - camera[:, 2])
        ranges.append(camera[:, 2])
    errors, ranges = np.concatenate(errors), np.concatenate(ranges)
    good = np.isfinite(errors)
    if good.sum() < 5:
        return {"n": int(good.sum())}
    e, r = errors[good], ranges[good]
    out = {"n": int(good.sum()), "median_mm": float(np.median(e)),
           "p90_abs_mm": float(np.percentile(np.abs(e), 90)), "std_mm": float(e.std()),
           "range_mm": [float(r.min()), float(r.max())]}
    # error = a * range + b, i.e. measured = (1 + a) * true + b. On 2026-10-09
    # this was a = -1.0 %, b = +1.5 mm; removing it left 0.76 mm scatter.
    if np.ptp(r) > 30:
        a, b = np.polyfit(r, e, 1)
        residual = e - (a * r + b)
        out["linear_fit"] = {"scale_error": float(a), "offset_mm": float(b),
                             "residual_std_mm": float(residual.std()),
                             "residual_p90_abs_mm": float(np.percentile(np.abs(residual), 90))}
    return out


# ------------------------------------------------------------------ session io
def blackfly_frames(session: Path) -> dict[int, dict[int, Path]]:
    """capture_sequence_index -> camera index -> image path (BMPs)."""
    out: dict[int, dict[int, Path]] = {}
    folders = sorted(p for p in session.glob("camera_*_*") if p.is_dir())
    for view, folder in enumerate(folders):
        with (folder / "frames.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = session / row["image_path"].replace("\\", "/")
                if row.get("complete", "1") == "1" and path.exists():
                    out.setdefault(int(row["capture_sequence_index"]), {})[view] = path
    return out


def detect(detector, gray, upscale: int = 1) -> dict[int, np.ndarray]:
    """ChArUco corners by id. `upscale` > 1 helps the D455's small, soft colour
    image: on 2026-10-09 the 18 mm board gave 61 corners at 1x and 65 at 2x."""
    if upscale > 1:
        gray = cv.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv.INTER_CUBIC)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return {}
    # Pixel centres scale as (x + 0.5) * s - 0.5, so map back the same way.
    pts = (np.asarray(corners, np.float64).reshape(-1, 2) + 0.5) / upscale - 0.5
    return {int(i): c for i, c in zip(np.asarray(ids).ravel(), pts)}


# ------------------------------------------------------------------ main flow
def collect(session: Path, rig, spec: BoardSpec, stride: int, still_mm: float,
            max_tri_rms: float):
    detector = make_detector(spec.build())
    d455_dir = session / "d455"
    meta = json.loads((d455_dir / "d455.json").read_text(encoding="utf-8"))
    with (d455_dir / "frames.csv").open(newline="", encoding="utf-8") as handle:
        d455_rows = {int(r["capture_sequence_index"]): r for r in csv.DictReader(handle)}
    images = blackfly_frames(session)
    sequences = sorted(set(images) & set(d455_rows))[::max(1, stride)]
    if not sequences:
        raise SystemExit("no frame set has both Blackfly BMPs and an extracted D455 frame; "
                         "run align_d455.py first, and keep the BMPs")

    per_frame = []
    for seq in sequences:
        observations: dict[int, dict[int, np.ndarray]] = {}
        for view, path in images[seq].items():
            raw = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            gray = cv.cvtColor(raw, GENICAM_RG_TO_GRAY) if raw.ndim == 2 else \
                cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
            for corner, pixel in detect(detector, gray).items():
                observations.setdefault(corner, {})[view] = pixel
        world = {}
        for corner, seen in observations.items():
            if len(seen) >= 2:
                point, rms = triangulate(rig, seen)
                if rms <= max_tri_rms:
                    world[corner] = point
        color = cv.imread(str(d455_dir / "color" / f"{seq:09d}.jpg"), cv.IMREAD_GRAYSCALE)
        d455_pixels = detect(detector, color, upscale=2) if color is not None else {}
        per_frame.append({"seq": seq, "world": world, "d455": d455_pixels})

    # Keep frames whose board did not move relative to the previous sampled set:
    # the D455 and the Blackflys are not exposed at the same instant.
    kept = []
    for previous, frame in zip([None] + per_frame[:-1], per_frame):
        shared = set(frame["world"]) & set(previous["world"]) if previous else set()
        if len(shared) >= 6:
            moved = np.median([np.linalg.norm(frame["world"][c] - previous["world"][c])
                               for c in shared])
            if moved <= still_mm and len(set(frame["world"]) & set(frame["d455"])) >= 6:
                kept.append(frame)
    return meta, per_frame, kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path, nargs="?")
    parser.add_argument("--board", type=Path, default=HERE / "board_18mm" / "board.json",
                        help="board_18mm: the 8x11 board in the lab (18 mm squares); "
                             "board_d455: the printable A4 alternative (25 mm)")
    parser.add_argument("--calibration", type=Path, default=HERE / "calibration.json")
    parser.add_argument("--stride", type=int, default=2, help="use every Nth frame set")
    parser.add_argument("--still-mm", type=float, default=0.5,
                        help="board motion between sampled frame sets to still count as still")
    parser.add_argument("--max-tri-rms", type=float, default=2.0,
                        help="drop triangulated corners with a worse re-projection (px)")
    parser.add_argument("--install", action="store_true",
                        help="also write calibration/d455_world.json")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest(args)
    if args.session is None:
        parser.error("session is required")

    session = args.session.expanduser().resolve()
    rig = load_rig(args.calibration)
    spec = BoardSpec.load(args.board)
    meta, frames, kept = collect(session, rig, spec, args.stride, args.still_mm,
                                 args.max_tri_rms)
    print(f"{len(frames)} frame sets examined, {len(kept)} with the board still and "
          "seen by the D455 and >= 2 Blackflys")
    if len(kept) < 3:
        raise SystemExit("need the board still in at least 3 frame sets; hold each pose ~2 s")

    world_pts, pixels, per_pose = [], [], []
    for frame in kept:
        shared = sorted(set(frame["world"]) & set(frame["d455"]))
        world_pts.append(np.stack([frame["world"][c] for c in shared]))
        pixels.append(np.stack([frame["d455"][c] for c in shared]))
        per_pose.append((frame["seq"], shared))
    world_all, pixels_all = np.concatenate(world_pts), np.concatenate(pixels)
    K_c, dist_c = realsense_K_dist(meta["color_intrinsics"])
    R_wc, t_wc, keep, rms = solve_pose(world_all, pixels_all, K_c, dist_c)
    R_dc, t_dc = depth_to_color(meta)
    R_wd = R_dc.T @ R_wc
    t_wd = R_dc.T @ (t_wc - t_dc)
    centre = -R_wc.T @ t_wc
    spread = np.ptp(world_all, axis=0)
    print(f"PnP on {int(keep.sum())}/{len(keep)} corners from {len(kept)} poses: "
          f"re-projection RMS {rms:.2f} px")
    print(f"D455 colour camera at ({centre[0]:.1f}, {centre[1]:.1f}, {centre[2]:.1f}) mm "
          f"in the world; board corners spread {spread[0]:.0f} x {spread[1]:.0f} x "
          f"{spread[2]:.0f} mm")
    if spread[2] < 10:
        print("WARNING: every pose lies in one plane; tilt or raise the board in some "
              "poses so the pose is well determined in depth.")

    K_d, _ = realsense_K_dist(meta["depth_intrinsics"])
    depths = [cv.imread(str(session / "d455" / "depth" / f"{f['seq']:09d}.png"),
                        cv.IMREAD_UNCHANGED) for f in kept]
    check = depth_check([np.stack(list(f["world"].values())) for f in kept], depths,
                        R_wd, t_wd, K_d, float(meta["depth_scale_m"]))
    if check.get("n", 0) >= 5:
        print(f"D455 depth at board corners vs triangulated truth: median "
              f"{check['median_mm']:+.2f} mm, |err| p90 {check['p90_abs_mm']:.2f} mm, "
              f"std {check['std_mm']:.2f} mm over {check['n']} corners at "
              f"{check['range_mm'][0]:.0f}-{check['range_mm'][1]:.0f} mm range")
        fit = check.get("linear_fit")
        if fit:
            print(f"  as measured = (1 {fit['scale_error']:+.4f}) x true {fit['offset_mm']:+.2f} mm; "
                  f"with that removed: std {fit['residual_std_mm']:.2f} mm, |err| p90 "
                  f"{fit['residual_p90_abs_mm']:.2f} mm (stored as depth_correction)")
    else:
        print("D455 depth check: too few corners with valid depth")

    result = {
        "schema": "d455-world/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "units": "millimetre",
        "pose_convention": "X_cam = R_world_to_cam @ X_world + t_world_to_cam",
        "calibration": str(args.calibration.resolve()),
        "session": str(session), "board": spec.__dict__,
        "serial": meta.get("serial"), "firmware": meta.get("firmware"),
        "R_world_to_color": R_wc.tolist(), "t_world_to_color": t_wc.tolist(),
        "R_world_to_depth": R_wd.tolist(), "t_world_to_depth": t_wd.tolist(),
        "color_position_world": centre.tolist(),
        "color_intrinsics": meta["color_intrinsics"],
        "depth_intrinsics": meta["depth_intrinsics"],
        "depth_scale_m": meta["depth_scale_m"],
        "depth_to_color": meta["depth_to_color"],
        "pnp": {"poses": len(kept), "corners": int(keep.sum()), "rms_px": rms,
                "frame_sets": [seq for seq, _ in per_pose]},
        "depth_check": check,
        # true_mm = (measured_mm - offset_mm) / (1 + scale_error), fitted on the
        # board's printed surface; a scattering surface such as foam can differ.
        "depth_correction": ({"scale_error": check["linear_fit"]["scale_error"],
                              "offset_mm": check["linear_fit"]["offset_mm"]}
                             if "linear_fit" in check else None),
    }
    out = session / "d455" / "d455_world.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    if args.install:
        (HERE / "d455_world.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"installed {HERE / 'd455_world.json'}")
    return 0


# ------------------------------------------------------------------ selftest
def selftest(args) -> int:
    """Known D455 pose, synthetic detections from the real rig calibration."""
    rng = np.random.default_rng(1)
    rig = load_rig(args.calibration)
    spec = BoardSpec.load(args.board)
    board = spec.object_points().astype(float)
    # D455 ~650 mm up, tilted 20 deg, looking at the platform.
    tilt = np.radians(12)
    R_wc_true = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]) @ \
        cv.Rodrigues(np.array([tilt, 0, 0]))[0]
    centre_true = np.array([70.0, -20.0, 620.0])
    t_wc_true = -R_wc_true @ centre_true
    meta = {"color_intrinsics": {"fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0,
                                 "model": "brown_conrady", "coeffs": [0, 0, 0, 0, 0],
                                 "width": 1280, "height": 720},
            "depth_intrinsics": {"fx": 425.0, "fy": 425.0, "ppx": 424.0, "ppy": 240.0,
                                 "model": "brown_conrady", "coeffs": [0, 0, 0, 0, 0],
                                 "width": 848, "height": 480},
            "depth_to_color": {"rotation_col_major": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                               "translation_m": [-0.059, 0.0, 0.0]},
            "depth_scale_m": 0.001}
    K_c, dist_c = realsense_K_dist(meta["color_intrinsics"])
    world_pts, pixels, depth_frames, poses_world = [], [], [], []
    R_dc, t_dc = depth_to_color(meta)
    R_wd_true = R_dc.T @ R_wc_true
    t_wd_true = R_dc.T @ (t_wc_true - t_dc)
    K_d, _ = realsense_K_dist(meta["depth_intrinsics"])
    bias = 1.5                                      # mm, planted D455 depth bias
    for k in range(6):
        R_b = cv.Rodrigues(rng.normal(0, 0.25, 3))[0]
        t_b = np.array([rng.uniform(-20, 80), rng.uniform(-150, 0), rng.uniform(0, 40)])
        corners = board @ R_b.T + t_b
        seen = {}
        for view, cam in enumerate(rig):
            rvec, _ = cv.Rodrigues(cam["R"])
            px, _ = cv.projectPoints(corners, rvec, cam["t"], cam["K"], cam["dist"])
            seen[view] = px.reshape(-1, 2) + rng.normal(0, 0.15, (len(corners), 2))
        tri = np.stack([triangulate(rig, {v: seen[v][i] for v in seen})[0]
                        for i in range(len(corners))])
        rvec, _ = cv.Rodrigues(R_wc_true)
        px, _ = cv.projectPoints(corners, rvec, t_wc_true, K_c, dist_c)
        world_pts.append(tri)
        pixels.append(px.reshape(-1, 2) + rng.normal(0, 0.3, (len(corners), 2)))
        # depth image of the board plane, in the depth frame, plus a bias
        depth = np.zeros((480, 848), np.uint16)
        cam_pts = corners @ R_wd_true.T + t_wd_true
        normal_c = R_wd_true @ (R_b @ np.array([0, 0, 1.0]))
        d0 = normal_c @ cam_pts[0]
        vv, uu = np.mgrid[0:480, 0:848]
        rays = np.stack([(uu - K_d[0, 2]) / K_d[0, 0], (vv - K_d[1, 2]) / K_d[1, 1],
                         np.ones_like(uu, float)], -1)
        z = d0 / (rays @ normal_c)
        depth[:] = np.clip(np.where(z > 0, z + bias, 0), 0, 65535).astype(np.uint16)
        depth_frames.append(depth)
        poses_world.append(tri)
    R, t, keep, rms = solve_pose(np.concatenate(world_pts), np.concatenate(pixels), K_c, dist_c)
    angle = np.degrees(np.arccos(np.clip((np.trace(R @ R_wc_true.T) - 1) / 2, -1, 1)))
    shift = np.linalg.norm(-R.T @ t - centre_true)
    R_wd = R_dc.T @ R
    t_wd = R_dc.T @ (t - t_dc)
    check = depth_check(poses_world, depth_frames, R_wd, t_wd, K_d, 0.001)
    ok = angle < 0.3 and shift < 3.0 and abs(check["median_mm"] - bias) < 0.6
    print(f"  pose error {angle:.3f} deg, position error {shift:.2f} mm (PnP rms {rms:.2f} px)")
    print(f"  depth check median {check['median_mm']:+.2f} mm (planted {bias:+.1f}; uint16 "
          "rounding adds up to 0.5)")
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
