#!/usr/bin/env python3
"""Solve intrinsics + extrinsics for the camera rig from detected corners.

    python calibrate.py detections.npz --out calibration.json --world-frames 0:120

Why intrinsics are computed here instead of inside `calibrateMultiview`
---------------------------------------------------------------------
`cv::calibrateMultiview` accepts partially observed patterns by asking you to
fill the unobserved corners with ``(-1, -1)``. Its *final* Levenberg-Marquardt
stage is robustified (a point whose error exceeds ~10 px gets weight ~0), so
those placeholders are harmlessly ignored there. Its *initialisation* stages
are not: `calibrateCamera`, `solvePnP` and `registerCameras` are plain
least-squares and receive the very same arrays, placeholders included (see
`modules/calib/src/multiview_calibration.cpp`). A single ``(-1, -1)`` sits
about a thousand pixels away from the truth and therefore dominates a
least-squares fit.

This is not theoretical. Feeding this pipeline's own synthetic rig with
placeholder-padded partial views makes the solve diverge to an RMS of ~2e14 px;
the same data with placeholder-free cells solves to 0.17 px.

So this script:

1. calibrates each camera's intrinsics itself, feeding `calibrateCamera` only
   the corners that were actually detected (variable-length views), and
2. hands those intrinsics to `calibrateMultiview` with ``CALIB_USE_INTRINSIC_GUESS``
   -- which is free, because the multi-view LM never refines intrinsics anyway --
   and enables a camera/frame cell only when every corner it passes was really
   observed, so no placeholder ever reaches the initialisation.

By default step 2 uses cells that saw the complete board. If your views never
share a whole board, ``--common-corners N`` instead solves on the N corners the
views do share; that shrinks the pattern but keeps every observation genuine.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

import bundle
from charuco_board import BoardSpec
from geometry import (as_rotation_matrix, average_rigid, camera_center, compose,
                      invert)

MM_PER_METRE = 1000.0

DIST_MODELS = {
    "standard": 0,                       # k1 k2 p1 p2 k3
    "rational": cv.CALIB_RATIONAL_MODEL,  # + k4 k5 k6
    "thin_prism": cv.CALIB_RATIONAL_MODEL | cv.CALIB_THIN_PRISM_MODEL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-view calibration from ChArUco detections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("detections", type=Path, help="detections.npz from detect_corners.py")
    parser.add_argument("--out", type=Path, default=Path("calibration.json"))

    group = parser.add_argument_group("intrinsics")
    group.add_argument("--dist-model", default="standard", choices=sorted(DIST_MODELS),
                       help="'standard' is right for machine-vision lenses; "
                            "'rational' only pays off on very wide optics")
    group.add_argument("--min-corners-intrinsics", type=int, default=12)
    group.add_argument("--max-intrinsic-views", type=int, default=250,
                       help="cap per camera; more views mostly cost time")
    group.add_argument("--reject-view-rms", type=float, default=1.0,
                       help="drop views worse than this many px and refit once")

    group = parser.add_argument_group("multi-view")
    group.add_argument("--common-corners", type=int, default=0, metavar="N",
                       help="fall-back for rigs whose views never share the whole "
                            "board: solve on the N most widely seen corners instead, "
                            "keeping only cells that see all N. 0 = use the complete "
                            "board. Never fills unobserved corners with placeholders.")
    group.add_argument("--max-frames", type=int, default=400,
                       help="cap on frames handed to the global optimiser")
    group.add_argument("--holdout-frac", type=float, default=0.2,
                       help="fraction of frames withheld for validate.py")
    group.add_argument("--holdout-block", type=int, default=10,
                       help="hold out contiguous blocks of this many frames; at 24 fps "
                            "a random split would leak near-duplicate frames")
    group.add_argument("--no-refine", action="store_true",
                       help="skip the final bundle adjustment. The detectionMask "
                            "calibrateMultiview takes is indexed by (camera, frame), "
                            "so a view counts only if that camera measured the WHOLE "
                            "pattern in it -- on a real session that discards most of "
                            "what was detected. The refinement re-solves the extrinsics "
                            "and the per-frame board poses against every corner that "
                            "was actually seen.")
    group.add_argument("--stereo-init", action="store_true",
                       help="initialise pairs with stereoCalibrate instead of registerCameras")

    group = parser.add_argument_group("world frame")
    group.add_argument("--world-frames", default=None, metavar="A:B",
                       help="capture_sequence_index range in which the board lay flat "
                            "on the tactile array / object platform; those frames "
                            "define the world origin. Without this the world frame is "
                            "camera 0.")
    group.add_argument("--world-z-offset", type=float, default=0.0, metavar="MM",
                       help="shift the world origin along final +z, e.g. to subtract "
                            "the thickness of the board substrate")
    group.add_argument("--world-flip-z", action="store_true",
                       help="rotate the world frame 180 deg about x so +z points from "
                            "the board towards the cameras (height above the array)")
    return parser.parse_args()


# --------------------------------------------------------------------- helpers
def uniform_subset(indices: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    picks = np.linspace(0, len(indices) - 1, limit).round().astype(int)
    return indices[np.unique(picks)]


def block_holdout(num_frames: int, fraction: float, block: int) -> np.ndarray:
    """Mark contiguous blocks of frames as held out, spread over the session."""
    holdout = np.zeros(num_frames, bool)
    if fraction <= 0 or num_frames < 2 * block:
        return holdout
    num_blocks = max(1, int(round(num_frames * fraction / block)))
    starts = np.linspace(0, num_frames - block, num_blocks).round().astype(int)
    for start in np.unique(starts):
        holdout[start:start + block] = True
    return holdout


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    n = len(adjacency)
    seen = [False] * n
    groups = []
    for start in range(n):
        if seen[start]:
            continue
        stack, group = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            group.append(node)
            for other in range(n):
                if not seen[other] and adjacency[node, other]:
                    seen[other] = True
                    stack.append(other)
        groups.append(sorted(group))
    return groups


def choose_common_corners(detected: np.ndarray, counts: np.ndarray, wanted: int) -> np.ndarray:
    """Pick the N corner ids that every camera can actually contribute.

    This is the exact alternative to OpenCV's ``(-1, -1)`` placeholder
    convention. Instead of padding partial views with fake points, we shrink the
    *pattern* to the part of the board the views genuinely share, so every value
    the solver sees is a real measurement. The cost is a smaller effective
    board, hence a slightly weaker per-frame pose -- a bounded trade-off rather
    than a silent corruption.

    Corners are ranked by their detection rate in the camera that sees them
    LEAST often, not by their total count. Ranking by the total is dominated by
    whichever cameras see the most of the board, and it happily picks a region
    the weakest camera never sees -- which then drops out of the solve entirely.
    Optimising the bottleneck instead gave, on this rig, 45% more usable frames
    for the worst camera, seven times the co-visibility on its thinnest pair,
    and a *larger* board patch at the same N.
    """
    candidate = counts >= wanted
    if candidate.sum() == 0:
        raise SystemExit(f"no view detects {wanted} corners; lower --common-corners")

    num_cameras = detected.shape[0]
    rate = np.zeros((num_cameras, detected.shape[2]))
    for camera in range(num_cameras):
        rows = detected[camera][counts[camera] >= wanted]
        if len(rows):
            rate[camera] = rows.mean(axis=0)
    score = rate.min(axis=0)

    order = np.argsort(-score, kind="stable")
    chosen = np.sort(order[:wanted])
    if score[chosen].min() <= 0:
        # No corner is seen by every camera; fall back to overall frequency so
        # the run still produces something, and say so.
        print("   NOTE: no corner is visible to all cameras at this size; "
              "falling back to overall frequency")
        frequency = detected[candidate].sum(axis=0)
        chosen = np.sort(np.argsort(-frequency, kind="stable")[:wanted])
        if frequency[chosen].min() == 0:
            raise SystemExit("not enough distinct corners were ever detected")
    return chosen


def calibrate_intrinsics(object_points, corners, counts, image_size, args):
    """Per-camera intrinsics using only the corners that were really detected."""
    flags = DIST_MODELS[args.dist_model]
    views = np.flatnonzero(counts >= args.min_corners_intrinsics)
    if views.size < 8:
        raise SystemExit(
            f"only {views.size} usable views for intrinsics -- need at least 8; "
            "shoot more frames with the board filling the frame"
        )
    views = uniform_subset(views, args.max_intrinsic_views)

    def fit(selection):
        obj, img = [], []
        for fi in selection:
            valid = corners[fi, :, 0] >= 0
            obj.append(np.ascontiguousarray(object_points[valid], np.float32))
            img.append(np.ascontiguousarray(corners[fi, valid], np.float32).reshape(-1, 1, 2))
        rms, K, dist, _, _, _, _, per_view = cv.calibrateCameraExtended(
            obj, img, tuple(int(v) for v in image_size), None, None, flags=flags)
        return rms, K, dist, np.asarray(per_view).reshape(-1)

    rms, K, dist, per_view = fit(views)
    rejected = 0
    if args.reject_view_rms > 0 and views.size > 12:
        threshold = max(args.reject_view_rms, 3.0 * float(np.median(per_view)))
        keep = per_view <= threshold
        if keep.sum() >= 8 and keep.sum() < views.size:
            rejected = int((~keep).sum())
            views = views[keep]
            rms, K, dist, per_view = fit(views)

    return {
        "K": K, "dist": dist, "rms": float(rms),
        "views": int(views.size), "rejected": rejected, "flags": flags,
    }


def board_pose_in_camera(object_points, image_points, K, dist):
    """solvePnP on the detected subset only, refined with LM."""
    valid = image_points[:, 0] >= 0
    if valid.sum() < 8:
        return None
    obj = np.ascontiguousarray(object_points[valid], np.float64)
    img = np.ascontiguousarray(image_points[valid], np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv.solvePnP(obj, img, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rvec, tvec = cv.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    return as_rotation_matrix(rvec), np.asarray(tvec, np.float64).reshape(3)


# ------------------------------------------------------------------------ main
def main() -> int:
    args = parse_args()
    data = np.load(args.detections)
    meta = json.loads(args.detections.with_suffix(".json").read_text(encoding="utf-8"))
    spec = BoardSpec(**{k: v for k, v in meta["board"].items()
                        if k in BoardSpec.__annotations__})

    corners = data["corners"]           # (C, F, P, 2)
    counts = data["counts"]             # (C, F)
    frame_indices = data["frame_indices"]
    image_sizes = data["image_sizes"]
    object_points = data["object_points"].astype(np.float32)
    num_cams, num_frames, num_pts, _ = corners.shape
    serials = meta["serials"]

    print(f"detections: {args.detections}  ({num_cams} cameras, {num_frames} frames, "
          f"{num_pts} corners)")
    print(f"board     : {spec.describe()}")

    # -------------------------------------------------------- 1. intrinsics
    print()
    print("1) intrinsics, per camera, from detected corners only")
    intrinsics = []
    for ci in range(num_cams):
        result = calibrate_intrinsics(object_points, corners[ci], counts[ci],
                                      image_sizes[ci], args)
        intrinsics.append(result)
        K = result["K"]
        print(f"   cam {ci} ({serials[ci]}): rms {result['rms']:.4f} px  "
              f"views {result['views']}"
              + (f" (rejected {result['rejected']})" if result["rejected"] else "")
              + f"  f=({K[0,0]:.1f}, {K[1,1]:.1f})  c=({K[0,2]:.1f}, {K[1,2]:.1f})")
        if result["rms"] > 0.5:
            print("        WARNING: > 0.5 px. Check focus, motion blur, and that the")
            print("        board is rigid and fills the frame in some views.")

    # ------------------------------------------------- 2. multi-view solving
    # Whatever selection we make, the invariant is the same: every corner handed
    # to `calibrateMultiview` for an enabled camera/frame cell was really
    # observed. No (-1, -1) placeholder ever reaches it.
    detected = corners[:, :, :, 0] >= 0            # (C, F, P)
    print()
    if args.common_corners > 0:
        point_ids = choose_common_corners(detected, counts, args.common_corners)
        enabled = detected[:, :, point_ids].all(axis=2)
        extent = object_points[point_ids]
        span = extent.max(axis=0) - extent.min(axis=0)
        print(f"2) multi-view on a common sub-pattern of {point_ids.size} corners "
              f"spanning {span[0]:.0f}x{span[1]:.0f} mm of the board")
        print("   (a cell counts when it sees ALL of them, so the observations stay exact)")
    else:
        point_ids = np.arange(num_pts)
        enabled = counts == num_pts
        print(f"2) multi-view on the complete {num_pts}-corner board")
    subset_points = object_points[point_ids]
    num_subset = point_ids.size

    usable_frames = np.flatnonzero(enabled.sum(axis=0) >= 2)
    if usable_frames.size == 0:
        raise SystemExit(
            "no frame is seen by two cameras at this threshold. Try "
            "--common-corners 20 to solve on the part of the board that the "
            "views actually share."
        )

    holdout_mask = block_holdout(usable_frames.size, args.holdout_frac, args.holdout_block)
    calib_frames = usable_frames[~holdout_mask]
    holdout_frames = usable_frames[holdout_mask]
    calib_frames = uniform_subset(calib_frames, args.max_frames)
    print(f"   frames: {usable_frames.size} usable -> {calib_frames.size} for the solve, "
          f"{holdout_frames.size} held out for validation")

    mask = enabled[:, calib_frames].astype(np.uint8)
    for ci in range(num_cams):
        if mask[ci].sum() < 10:
            raise SystemExit(
                f"camera {ci} ({serials[ci]}) contributes only {mask[ci].sum()} frames. "
                "Move the board so this view sees the whole board more often, or use "
                "--common-corners to solve on the shared part of the board."
            )
    overlap = (mask[:, None, :] & mask[None, :, :]).sum(axis=2)
    np.fill_diagonal(overlap, 0)
    groups = connected_components(overlap > 0)
    if len(groups) > 1:
        raise SystemExit(
            "the camera graph is not connected: " + " | ".join(str(g) for g in groups)
            + "\nEvery camera must share frames with at least one other camera."
        )

    image_points = []
    for ci in range(num_cams):
        per_frame = []
        for fi in calib_frames:
            if enabled[ci, fi]:
                per_frame.append(np.ascontiguousarray(corners[ci, fi, point_ids], np.float32))
            else:
                # Disabled cells are skipped by every stage of the solver, so the
                # contents are irrelevant; the shape still has to match.
                per_frame.append(np.full((num_subset, 2), -1.0, np.float32))
        image_points.append(per_frame)

    # `calibrateMultiview` normalises the object points by dividing them by the
    # SQUARED maximum pairwise distance (getScaleOfObjPoints returns NORM_L2SQR
    # but is used as a length). The normalised points therefore span 1/D, so a
    # board given in millimetres shrinks by ~1e-2 and trips the internal
    # collinearity guard with "Pattern points are collinear!". Feeding metres
    # keeps that guard happy; translations come back in the same unit, so we
    # scale them straight back to millimetres afterwards.
    obj_per_frame = [(subset_points / MM_PER_METRE).astype(np.float32) for _ in calib_frames]
    flags = cv.CALIB_USE_INTRINSIC_GUESS | (cv.CALIB_STEREO_REGISTRATION if args.stereo_init else 0)

    rms, Ks, dists, Rs, Ts, init_pairs, rvecs0, tvecs0, per_frame_errors = \
        cv.calibrateMultiviewExtended(
            objPoints=obj_per_frame,
            imagePoints=image_points,
            imageSize=[tuple(int(v) for v in size) for size in image_sizes],
            detectionMask=mask,
            models=np.full(num_cams, cv.CALIB_MODEL_PINHOLE, np.uint8),
            Ks=[np.asarray(item["K"], np.float64) for item in intrinsics],
            distortions=[np.asarray(item["dist"], np.float64) for item in intrinsics],
            Rs=None, Ts=None,
            flagsForIntrinsics=np.array([item["flags"] for item in intrinsics], np.int32),
            flags=flags,
        )

    per_frame_errors = np.asarray(per_frame_errors)
    per_camera_rms = [float(np.sqrt(np.mean(row[row >= 0] ** 2))) if np.any(row >= 0) else float("nan")
                      for row in per_frame_errors]
    print(f"   overall RMS: {rms:.4f} px")
    for ci in range(num_cams):
        print(f"   cam {ci} ({serials[ci]}): {per_camera_rms[ci]:.4f} px "
              f"over {int((per_frame_errors[ci] >= 0).sum())} frames")
    print(f"   initialisation pairs: "
          f"{[tuple(int(v) for v in pair) for pair in np.asarray(init_pairs).reshape(-1, 2)]}")
    if rms > 0.5:
        print("   WARNING: > 0.5 px overall. Do not use this calibration before")
        print("   validate.py confirms the metric error.")

    # camera-0-relative poses, converted back from metres to millimetres
    poses = []
    for ci in range(num_cams):
        R = as_rotation_matrix(np.asarray(Rs[ci]))
        t = np.asarray(Ts[ci], np.float64).reshape(3) * MM_PER_METRE
        poses.append((R, t))

    refine_info = None
    if not args.no_refine:
        print()
        print("2b) bundle adjustment over every detected corner")
        Rs_ref, Ts_ref, refine_info = bundle.refine(
            corners, object_points,
            [item["K"] for item in intrinsics],
            [item["dist"] for item in intrinsics],
            [pose[0] for pose in poses], [pose[1] for pose in poses])
        poses = [(as_rotation_matrix(np.asarray(R)), np.asarray(t, np.float64).reshape(3))
                 for R, t in zip(Rs_ref, Ts_ref)]

    # ----------------------------------------------------- 3. world frame
    print()
    world_info = {"origin": "camera_0", "frames": [], "z_offset_mm": args.world_z_offset,
                  "flip_z": bool(args.world_flip_z)}
    R_w, t_w = np.eye(3), np.zeros(3)

    if args.world_frames:
        lo, hi = (int(v) for v in args.world_frames.split(":"))
        selected = np.flatnonzero((frame_indices >= lo) & (frame_indices < hi))
        rotations, translations, used = [], [], []
        for fi in selected:
            for ci in range(num_cams):
                if counts[ci, fi] < 12:
                    continue
                pose = board_pose_in_camera(object_points, corners[ci, fi],
                                            Ks[ci], dists[ci])
                if pose is None:
                    continue
                R_bc, t_bc = pose
                R_c0, t_c0 = invert(*poses[ci])          # camera ci -> camera 0
                R_b0, t_b0 = compose(R_bc, t_bc, R_c0, t_c0)
                rotations.append(R_b0)
                translations.append(t_b0)
                used.append(int(frame_indices[fi]))
        if not rotations:
            raise SystemExit(
                f"--world-frames {args.world_frames} contains no frame where the board "
                "was detected well enough to define the world origin"
            )
        R_w, t_w, spread = average_rigid(rotations, translations)
        world_info.update({"origin": "board", "frames": sorted(set(used)), "spread": spread})
        print(f"3) world frame from {len(rotations)} board observations in "
              f"{len(set(used))} frames")
        print(f"   spread: {spread['max_rotation_deg']:.3f} deg, "
              f"{spread['max_translation_mm']:.3f} mm max deviation")
        if spread["max_rotation_deg"] > 0.5 or spread["max_translation_mm"] > 1.0:
            print("   WARNING: the 'static' board moved, or one camera disagrees.")
            print("   Re-shoot the world-frame segment with the board truly still.")
    else:
        print("3) world frame = camera 0 (no --world-frames given).")
        print("   For tactile-array-referenced ground truth, re-run with a range of")
        print("   frames in which the board lay flat on the sensor platform.")

    if args.world_flip_z:
        flip = np.diag([1.0, -1.0, -1.0])
        R_w, t_w = R_w @ flip, t_w
    if args.world_z_offset:
        t_w = t_w + R_w @ np.array([0.0, 0.0, float(args.world_z_offset)])

    # ------------------------------------------------------------ 4. output
    cameras = []
    for ci in range(num_cams):
        R_c, t_c = poses[ci]
        R_wc, t_wc = compose(R_w, t_w, R_c, t_c)   # world -> camera ci
        K = np.asarray(Ks[ci], np.float64)
        P = K @ np.hstack([R_wc, t_wc.reshape(3, 1)])
        cameras.append({
            "index": ci,
            "serial": serials[ci],
            "image_size": [int(v) for v in image_sizes[ci]],
            "K": K.tolist(),
            "dist": np.asarray(dists[ci], np.float64).reshape(-1).tolist(),
            "dist_model": args.dist_model,
            "R_world_to_cam": R_wc.tolist(),
            "t_world_to_cam": t_wc.tolist(),
            "P": P.tolist(),
            "position_world": camera_center(R_wc, t_wc).tolist(),
            "optical_axis_world": (R_wc.T @ np.array([0.0, 0.0, 1.0])).tolist(),
            "intrinsics_rms_px": intrinsics[ci]["rms"],
            "intrinsics_views": intrinsics[ci]["views"],
            "multiview_rms_px": per_camera_rms[ci],
        })

    print()
    print("   camera positions in the world frame (mm):")
    for camera in cameras:
        x, y, z = camera["position_world"]
        print(f"     cam {camera['index']} ({camera['serial']}): "
              f"({x:9.2f}, {y:9.2f}, {z:9.2f})")

    payload = {
        "schema": "multiview-calibration/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "opencv_version": cv.__version__,
        "units": "millimetre",
        "pose_convention": "X_cam = R_world_to_cam @ X_world + t_world_to_cam",
        "board": meta["board"],
        "source": {
            "detections": str(args.detections),
            "session": meta.get("session"),
            "bayer": meta.get("bayer"),
            "frames_calibration": [int(frame_indices[fi]) for fi in calib_frames],
            "frames_holdout": [int(frame_indices[fi]) for fi in holdout_frames],
            "corner_ids_used": [int(v) for v in point_ids],
            "common_corners": int(args.common_corners),
            "stereo_init": bool(args.stereo_init),
            "bundle_refinement": refine_info,
        },
        "world": world_info,
        "cameras": cameras,
        "metrics": {
            # After refinement this is the RMS of the model actually shipped,
            # measured over every detected corner rather than over the subset
            # calibrateMultiview was able to accept. Reporting the pre-refinement
            # number would grade a calibration nobody is using.
            "multiview_rms_px": float(refine_info["rms_after"]) if refine_info
                                else float(rms),
            "opencv_multiview_rms_px": float(rms),
            "per_camera_rms_px": per_camera_rms,
            "initialization_pairs": np.asarray(init_pairs).reshape(-1, 2).tolist(),
            "frames_used": int(calib_frames.size),
            "overlap_matrix": overlap.tolist(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"wrote {args.out}")
    print(f"next : python validate.py {args.out} {args.detections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
