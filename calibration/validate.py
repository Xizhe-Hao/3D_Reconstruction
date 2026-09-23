#!/usr/bin/env python3
"""Independent accuracy check of a solved calibration.

    python validate.py calibration.json detections.npz --plots report

A low re-projection error is NOT proof of a good calibration: it says the model
fits the data it was fitted to, and it says nothing about metric scale or about
the depth direction, which is exactly where a ring of cameras with too little
angular separation goes wrong. So this script works on the frames `calibrate.py`
withheld, triangulates the ChArUco corners in 3D, and compares the result
against the board's known geometry -- an error in millimetres, not pixels.

Checks performed
----------------
* re-projection of triangulated points into every camera (px)
* rigid-fit residual of a triangulated board against the printed board (mm)
* inter-corner distance error, which is independent of the fit and therefore
  the honest statement of metric accuracy (mm)
* symmetric epipolar distance for every camera pair (px)
* triangulation parallax, i.e. whether the rig geometry can resolve depth at all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geometry import kabsch, triangulate, undistort_to_pixels

THRESHOLDS = {
    "intrinsics_rms_px": 0.30,
    "multiview_rms_px": 0.50,
    "holdout_reprojection_rms_px": 0.50,
    "board_residual_rms_mm": 0.20,
    "distance_error_p95_mm": 0.30,
    "epipolar_rms_px": 1.00,
    "median_parallax_deg": 15.0,   # this one is a lower bound
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a multi-view calibration against held-out frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("calibration", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("--frames", default="holdout", choices=["holdout", "all", "calibration"],
                        help="which frames to evaluate on; 'holdout' is the honest one")
    parser.add_argument("--min-cameras", type=int, default=2,
                        help="corners seen by fewer cameras cannot be triangulated")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--plots", type=Path, default=None, help="directory for diagnostic plots")
    parser.add_argument("--report", type=Path, default=None, help="write metrics as JSON")
    return parser.parse_args()


def verdict(name: str, value: float, limit: float, higher_is_better: bool = False) -> str:
    if not np.isfinite(value):
        return "n/a  "
    good = value >= limit if higher_is_better else value <= limit
    if good:
        return "PASS "
    slack = 2.0
    marginal = value <= limit * slack if not higher_is_better else value >= limit / slack
    return "WARN " if marginal else "FAIL "


def main() -> int:
    args = parse_args()
    calib = json.loads(args.calibration.read_text(encoding="utf-8"))
    data = np.load(args.detections)

    corners = data["corners"]
    counts = data["counts"]
    frame_indices = list(int(v) for v in data["frame_indices"])
    object_points = data["object_points"].astype(np.float64)
    num_cams, num_frames, num_pts, _ = corners.shape

    cameras = calib["cameras"]
    if len(cameras) != num_cams:
        raise SystemExit("calibration and detections disagree on the number of cameras")
    detection_meta = json.loads(
        args.detections.with_suffix(".json").read_text(encoding="utf-8"))
    for ci, camera in enumerate(cameras):
        if camera["serial"] != detection_meta["serials"][ci]:
            raise SystemExit(
                f"camera {ci} serial mismatch: calibration says {camera['serial']}, "
                f"detections say {detection_meta['serials'][ci]}. Calibration files are "
                "keyed by serial for exactly this reason."
            )

    Ks = [np.asarray(c["K"], np.float64) for c in cameras]
    dists = [np.asarray(c["dist"], np.float64) for c in cameras]
    Rs = [np.asarray(c["R_world_to_cam"], np.float64) for c in cameras]
    ts = [np.asarray(c["t_world_to_cam"], np.float64) for c in cameras]
    centers = [np.asarray(c["position_world"], np.float64) for c in cameras]

    wanted = {"holdout": set(calib["source"]["frames_holdout"]),
              "calibration": set(calib["source"]["frames_calibration"])}
    if args.frames == "all":
        selected = list(range(num_frames))
    else:
        selected = [i for i, index in enumerate(frame_indices) if index in wanted[args.frames]]
    if not selected:
        raise SystemExit(f"no {args.frames} frames available")
    if len(selected) > args.max_frames:
        picks = np.linspace(0, len(selected) - 1, args.max_frames).round().astype(int)
        selected = [selected[i] for i in np.unique(picks)]
    print(f"evaluating on {len(selected)} {args.frames} frames")

    # Undistort every observation once.
    undistorted = np.full_like(corners, -1.0, dtype=np.float64)
    for ci in range(num_cams):
        for fi in selected:
            valid = corners[ci, fi, :, 0] >= 0
            if valid.any():
                undistorted[ci, fi, valid] = undistort_to_pixels(
                    corners[ci, fi, valid], Ks[ci], dists[ci])

    reprojection_errors = [[] for _ in range(num_cams)]
    board_residuals = []
    distance_errors = []
    parallaxes = []
    triangulated_total = 0

    for fi in selected:
        points3d = {}
        for pid in range(num_pts):
            views = [ci for ci in range(num_cams) if corners[ci, fi, pid, 0] >= 0]
            if len(views) < args.min_cameras:
                continue
            observations = [(Ks[ci], Rs[ci], ts[ci], undistorted[ci, fi, pid]) for ci in views]
            X = triangulate(observations)
            if not np.all(np.isfinite(X)):
                continue
            points3d[pid] = X
            triangulated_total += 1

            rays = []
            for ci in views:
                direction = X - centers[ci]
                norm = np.linalg.norm(direction)
                if norm > 1e-9:
                    rays.append(direction / norm)
            if len(rays) >= 2:
                angles = [np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
                          for i, a in enumerate(rays) for b in rays[i + 1:]]
                parallaxes.append(max(angles))

            for ci in views:
                cam = Rs[ci] @ X + ts[ci]
                if cam[2] <= 1e-9:
                    continue
                projected = np.array([
                    Ks[ci][0, 0] * cam[0] / cam[2] + Ks[ci][0, 1] * cam[1] / cam[2] + Ks[ci][0, 2],
                    Ks[ci][1, 1] * cam[1] / cam[2] + Ks[ci][1, 2],
                ])
                reprojection_errors[ci].append(
                    float(np.linalg.norm(projected - undistorted[ci, fi, pid])))

        if len(points3d) >= 6:
            ids = sorted(points3d)
            cloud = np.array([points3d[i] for i in ids])
            model = object_points[ids]
            _, _, residuals = kabsch(cloud, model)
            board_residuals.extend(residuals.tolist())

            # Scale-truthful check: distances between corners cannot be fitted away.
            index = np.triu_indices(len(ids), k=1)
            measured = np.linalg.norm(cloud[index[0]] - cloud[index[1]], axis=1)
            expected = np.linalg.norm(model[index[0]] - model[index[1]], axis=1)
            distance_errors.extend(np.abs(measured - expected).tolist())

    if not board_residuals:
        raise SystemExit(
            "no frame had 6+ corners visible from 2+ cameras -- cannot validate. "
            "Capture frames where the board is shared between views."
        )

    # ------------------------------------------------------------- epipolar
    epipolar = {}
    for i in range(num_cams):
        for j in range(i + 1, num_cams):
            R_rel = Rs[j] @ Rs[i].T
            t_rel = ts[j] - R_rel @ ts[i]
            skew = np.array([[0, -t_rel[2], t_rel[1]],
                             [t_rel[2], 0, -t_rel[0]],
                             [-t_rel[1], t_rel[0], 0]])
            F = np.linalg.inv(Ks[j]).T @ (skew @ R_rel) @ np.linalg.inv(Ks[i])
            residuals = []
            for fi in selected:
                shared = np.flatnonzero((corners[i, fi, :, 0] >= 0) & (corners[j, fi, :, 0] >= 0))
                if not shared.size:
                    continue
                x1 = np.hstack([undistorted[i, fi, shared], np.ones((shared.size, 1))])
                x2 = np.hstack([undistorted[j, fi, shared], np.ones((shared.size, 1))])
                line2 = x1 @ F.T
                line1 = x2 @ F
                numerator = np.abs(np.sum(x2 * line2, axis=1))
                denom2 = np.sqrt(line2[:, 0] ** 2 + line2[:, 1] ** 2)
                denom1 = np.sqrt(line1[:, 0] ** 2 + line1[:, 1] ** 2)
                residuals.extend((0.5 * (numerator / denom2 + numerator / denom1)).tolist())
            if residuals:
                epipolar[f"{i}-{j}"] = float(np.sqrt(np.mean(np.square(residuals))))

    # ------------------------------------------------------------- summarise
    def rms(values):
        return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")

    per_camera_reprojection = [rms(v) for v in reprojection_errors]
    metrics = {
        "frames_evaluated": len(selected),
        "points_triangulated": triangulated_total,
        "holdout_reprojection_rms_px": rms([e for v in reprojection_errors for e in v]),
        "per_camera_reprojection_rms_px": per_camera_reprojection,
        "board_residual_rms_mm": rms(board_residuals),
        "board_residual_p95_mm": float(np.percentile(np.abs(board_residuals), 95)),
        "distance_error_mean_mm": float(np.mean(distance_errors)),
        "distance_error_p95_mm": float(np.percentile(distance_errors, 95)),
        "epipolar_rms_px": epipolar,
        "epipolar_rms_px_worst": float(max(epipolar.values())) if epipolar else float("nan"),
        "median_parallax_deg": float(np.median(parallaxes)) if parallaxes else float("nan"),
        "intrinsics_rms_px_worst": float(max(c["intrinsics_rms_px"] for c in cameras)),
        "multiview_rms_px": float(calib["metrics"]["multiview_rms_px"]),
    }

    print()
    print("geometry")
    for i in range(num_cams):
        for j in range(i + 1, num_cams):
            baseline = float(np.linalg.norm(centers[i] - centers[j]))
            axis_i = Rs[i].T @ np.array([0.0, 0.0, 1.0])
            axis_j = Rs[j].T @ np.array([0.0, 0.0, 1.0])
            angle = np.degrees(np.arccos(np.clip(np.dot(axis_i, axis_j), -1.0, 1.0)))
            print(f"  cam {i}-{j}: baseline {baseline:8.1f} mm   optical axes {angle:5.1f} deg"
                  f"   epipolar rms {epipolar.get(f'{i}-{j}', float('nan')):.3f} px")

    print()
    print("accuracy")
    rows = [
        ("worst per-camera intrinsics RMS", metrics["intrinsics_rms_px_worst"],
         THRESHOLDS["intrinsics_rms_px"], "px", False),
        ("multi-view solve RMS", metrics["multiview_rms_px"],
         THRESHOLDS["multiview_rms_px"], "px", False),
        (f"{args.frames} re-projection RMS", metrics["holdout_reprojection_rms_px"],
         THRESHOLDS["holdout_reprojection_rms_px"], "px", False),
        ("board rigid-fit residual RMS", metrics["board_residual_rms_mm"],
         THRESHOLDS["board_residual_rms_mm"], "mm", False),
        ("inter-corner distance error P95", metrics["distance_error_p95_mm"],
         THRESHOLDS["distance_error_p95_mm"], "mm", False),
        ("worst pair epipolar RMS", metrics["epipolar_rms_px_worst"],
         THRESHOLDS["epipolar_rms_px"], "px", False),
        ("median triangulation parallax", metrics["median_parallax_deg"],
         THRESHOLDS["median_parallax_deg"], "deg", True),
    ]
    failures = 0
    for label, value, limit, unit, higher in rows:
        state = verdict(label, value, limit, higher)
        failures += state.strip() == "FAIL"
        comparison = ">=" if higher else "<="
        print(f"  [{state.strip():4}] {label:<34} {value:8.4f} {unit:<4} "
              f"(want {comparison} {limit:g})")

    print()
    if metrics["distance_error_p95_mm"] > THRESHOLDS["distance_error_p95_mm"]:
        print("  The metric error is what matters for shape ground truth. If it is high")
        print("  while the pixel errors look fine, suspect the printed square size in")
        print("  board.json or a board that is not flat.")
    if metrics["median_parallax_deg"] < THRESHOLDS["median_parallax_deg"]:
        print("  Low parallax: the cameras are too close to each other in angle for the")
        print("  depth direction to be well constrained. Spread them 30-60 deg apart.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"  metrics -> {args.report}")

    if args.plots:
        make_plots(args.plots, corners, counts, cameras, reprojection_errors,
                   distance_errors, parallaxes)
        print(f"  plots   -> {args.plots}")

    return 1 if failures else 0


def make_plots(out_dir: Path, corners, counts, cameras, reprojection_errors,
               distance_errors, parallaxes) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    num_cams = len(cameras)

    # Corner coverage: thin coverage at the image border means the distortion
    # coefficients are extrapolating there.
    fig, axes = plt.subplots(1, num_cams, figsize=(4 * num_cams, 3.4), squeeze=False)
    for ci, camera in enumerate(cameras):
        width, height = camera["image_size"]
        points = corners[ci][corners[ci][:, :, 0] >= 0]
        axis = axes[0][ci]
        if len(points):
            axis.hist2d(points[:, 0], points[:, 1], bins=(32, 24),
                        range=[[0, width], [0, height]], cmap="viridis")
        axis.set_title(f"cam {ci} ({camera['serial']})\ncorner coverage")
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ci in range(num_cams):
        if reprojection_errors[ci]:
            axes[0].hist(reprojection_errors[ci], bins=60, histtype="step",
                         label=f"cam {ci}")
    axes[0].set_xlabel("re-projection error (px)")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    axes[1].hist(distance_errors, bins=60, color="tab:orange")
    axes[1].set_xlabel("inter-corner distance error (mm)")
    if parallaxes:
        axes[2].hist(parallaxes, bins=60, color="tab:green")
    axes[2].set_xlabel("triangulation parallax (deg)")
    fig.tight_layout()
    fig.savefig(out_dir / "errors.png", dpi=120)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 5))
    axis = fig.add_subplot(projection="3d")
    for camera in cameras:
        position = np.asarray(camera["position_world"], float)
        direction = np.asarray(camera["optical_axis_world"], float)
        length = 0.2 * max(1.0, float(np.linalg.norm(position)))
        axis.scatter(*position, s=40)
        axis.quiver(*position, *(direction * length), color="gray")
        axis.text(*position, f"  {camera['index']}:{camera['serial']}", fontsize=8)
    axis.scatter(0, 0, 0, marker="x", color="red", s=60)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")
    axis.set_title("camera layout in the world frame (red = origin)")
    fig.tight_layout()
    fig.savefig(out_dir / "layout.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
