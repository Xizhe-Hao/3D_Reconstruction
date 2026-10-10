#!/usr/bin/env python3
"""Has any camera moved since calibration.json was solved?

RUN WITH THE **CAPTURE** ENVIRONMENT (needs PySpin):

    cd "sensor-data-collection\\Python files"
    .\\.venv\\Scripts\\Activate.ps1
    python ..\\..\\calibration\\check_camera_moved.py

Two independent checks, both against the calibration session itself:

1. **Scene check (no board needed).** Each camera's live view is compared with
   the frame it recorded during calibration (the first world-frame image of
   the session named in calibration.json). ORB features on the static rig are
   matched and RANSAC keeps the dominant motion; the median shift is how far
   the camera has moved, in pixels. Anything that changed inside the frame (a
   new sensor, a hand, the board) is outvoted by the background.

2. **Board check (optional, stronger).** Lay the ChArUco board flat on the
   tactile array, as for the world segment, so every camera sees it. Each
   camera's board pose (solvePnP with the calibrated K and distortion) is
   carried into every other camera through the *calibrated* extrinsics and
   compared with what that camera detected. A camera that moved shows up as a
   whole row and column of large errors. The same matrix is computed on the
   calibration session's own world frame, so the numbers are judged against
   what this calibration achieved on the day, not against a guessed threshold.
   It also reports where the board surface now sits in the world frame: a
   different sensor stack moves it in z.

Verdict per camera, from the scene check:

    < 1 px    unchanged
    1-3 px    small shift; the board check decides
    > 3 px    moved -- recalibrate (at ~0.1 mm/px this is the size of the
              deformation being measured)

Live and reference frames are saved, with a red/cyan overlay per camera, under
--output, so the verdict can be checked by eye: a camera that has not moved
shows a grey background with colour fringes only on things that changed.

Cameras are left ready for a triggered capture on exit.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2 as cv
except ImportError:
    sys.exit("OpenCV missing in the capture environment:\n"
             "  python -m pip install opencv-python==4.9.0.80")

from charuco_board import BoardSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
UNCHANGED_PX, MOVED_PX = 1.0, 3.0
MIN_BOARD_CORNERS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, default=HERE / "calibration.json")
    parser.add_argument("--board", type=Path, default=HERE / "board" / "board.json")
    parser.add_argument("--session", type=Path,
                        help="calibration session folder; default: the one "
                             "recorded in calibration.json")
    parser.add_argument("--reference-frame", type=int,
                        help="capture index of the reference image; default: "
                             "the first world-frame index")
    parser.add_argument("--live", type=Path,
                        help="folder of saved live images (camera_<serial>.png) "
                             "to check instead of grabbing from the cameras")
    parser.add_argument("--frames", type=int, default=5,
                        help="live frames per camera; the median image is used")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--trigger-hz", type=float, default=24.2)
    parser.add_argument("--output", type=Path,
                        default=HERE / "camera_check")
    return parser.parse_args()


# ---------------------------------------------------------------- calibration

def load_calibration(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cameras = {}
    for cam in data["cameras"]:
        cameras[cam["serial"]] = {
            "index": cam["index"],
            "K": np.asarray(cam["K"], np.float64),
            "dist": np.asarray(cam["dist"], np.float64).reshape(-1),
            "R": np.asarray(cam["R_world_to_cam"], np.float64),
            "t": np.asarray(cam["t_world_to_cam"], np.float64).reshape(3),
            "size": tuple(cam["image_size"]),
        }
    return data, cameras


def bayer_gray(raw: np.ndarray) -> np.ndarray:
    # GenICam BayerRG8 is OpenCV's BG (see session_io.GENICAM_TO_OPENCV_BAYER).
    if raw.ndim == 2:
        return cv.cvtColor(raw, cv.COLOR_BayerBG2GRAY)
    return cv.cvtColor(raw, cv.COLOR_BGR2GRAY)


def load_reference(session: Path, serial: str, index: int) -> np.ndarray | None:
    folders = sorted(session.glob(f"camera_*_{serial}"))
    if not folders:
        return None
    files = sorted(folders[0].glob(f"capture_{index:09d}_*.*"))
    if not files:
        return None
    raw = cv.imread(str(files[0]), cv.IMREAD_UNCHANGED)
    return None if raw is None else bayer_gray(raw)


# ---------------------------------------------------------------- live frames

def grab_live(serials: list[str], args) -> dict[str, np.ndarray]:
    try:
        import PySpin
    except ImportError:
        sys.exit("PySpin not found -- activate the capture environment (.venv), "
                 "or pass --live <folder>.")
    from tune_exposure import prepare_for_capture, throttle

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    handles = []
    for i in range(cam_list.GetSize()):
        camera = cam_list[i]
        camera.Init()
        node = PySpin.CStringPtr(
            camera.GetTLDeviceNodeMap().GetNode("DeviceSerialNumber"))
        handles.append((node.GetValue() if PySpin.IsReadable(node) else "?", camera))
    found = {s for s, _ in handles}
    missing = [s for s in serials if s not in found]
    if missing:
        print(f"  WARNING: calibrated camera(s) not connected: {missing}")

    frames: dict[str, list[np.ndarray]] = {s: [] for s, _ in handles}
    try:
        for _, camera in handles:
            throttle(camera, args.fps)
            camera.BeginAcquisition()
        for attempt in range(args.frames + 3):
            for serial, camera in handles:
                try:
                    image = camera.GetNextImage(args.timeout_ms)
                except PySpin.SpinnakerException:
                    continue
                try:
                    if not image.IsIncomplete() and attempt >= 2:  # skip warm-up
                        frames[serial].append(bayer_gray(image.GetNDArray().copy()))
                finally:
                    image.Release()
    finally:
        for _, camera in handles:
            try:
                camera.EndAcquisition()
            except PySpin.SpinnakerException:
                pass
        for _, camera in handles:
            try:
                prepare_for_capture(camera, args.trigger_hz)
            except PySpin.SpinnakerException:
                pass
        while handles:
            _, camera = handles.pop()
            try:
                camera.DeInit()
            except PySpin.SpinnakerException:
                pass
            del camera
        cam_list.Clear()
        system.ReleaseInstance()

    return {
        s: np.median(np.stack(f), axis=0).astype(np.uint8)
        for s, f in frames.items() if f
    }


# ---------------------------------------------------------------- board

def make_detector(spec: BoardSpec):
    """OpenCV 4.x API, as in pose_helper.py."""
    dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, spec.dictionary))
    board = cv.aruco.CharucoBoard((spec.squares_x, spec.squares_y),
                                  float(spec.square_mm), float(spec.marker_mm),
                                  dictionary)
    board.setLegacyPattern(bool(spec.legacy_pattern))
    params = cv.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
    charuco_params = cv.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    detector = cv.aruco.CharucoDetector(board, charuco_params, params,
                                        cv.aruco.RefineParameters())
    object_points = np.asarray(board.getChessboardCorners(), np.float64).reshape(-1, 3)
    return detector, object_points


def detect(detector, gray):
    """Returns (corner_ids, corner_px, marker_corners)."""
    corners, ids, marker_corners, _ = detector.detectBoard(gray)
    if ids is None:
        return np.zeros(0, int), np.zeros((0, 2)), marker_corners
    return (np.asarray(ids).reshape(-1), np.asarray(corners, np.float64).reshape(-1, 2),
            marker_corners)


def board_mask(shape, marker_corners) -> np.ndarray:
    """255 everywhere except a dilated hull around the detected board."""
    mask = np.full(shape, 255, np.uint8)
    if marker_corners is None or len(marker_corners) == 0:
        return mask
    points = np.concatenate([np.asarray(c).reshape(-1, 2) for c in marker_corners])
    hull = cv.convexHull(points.astype(np.float32)).astype(np.int32)
    cv.fillConvexPoly(mask, hull, 0)
    return cv.erode(mask, np.ones((61, 61), np.uint8))


# ---------------------------------------------------------------- scene check

def scene_shift(reference, live, mask_ref, mask_live):
    """Median pixel shift of the static scene; (px, inliers) or (None, n)."""
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    a, b = clahe.apply(reference), clahe.apply(live)
    orb = cv.ORB_create(8000)
    ka, da = orb.detectAndCompute(a, mask_ref)
    kb, db = orb.detectAndCompute(b, mask_live)
    if da is None or db is None or len(da) < 30 or len(db) < 30:
        return None, 0
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(da, db), key=lambda m: m.distance)[:1500]
    if len(matches) < 30:
        return None, len(matches)
    src = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, inliers = cv.findHomography(src, dst, cv.RANSAC, 3.0)
    if inliers is None:
        return None, 0
    keep = inliers.ravel().astype(bool)
    if keep.sum() < 20:
        return None, int(keep.sum())
    shift = np.linalg.norm(dst[keep].reshape(-1, 2) - src[keep].reshape(-1, 2), axis=1)
    return float(np.median(shift)), int(keep.sum())


def tile_shift(reference, live, mask, tile=256, min_response=0.25):
    """Fallback when the new scene hides most features from ORB.

    Phase-correlates overlapping tiles and keeps the confident ones; on a rig
    that has not moved every static tile reads ~0 px. Returns (median px,
    confident tiles) or (None, n).
    """
    window = cv.createHanningWindow((tile, tile), cv.CV_32F)
    shifts = []
    height, width = reference.shape
    for y in range(0, height - tile + 1, tile // 2):
        for x in range(0, width - tile + 1, tile // 2):
            if mask[y:y + tile, x:x + tile].mean() < 250:
                continue
            a = reference[y:y + tile, x:x + tile].astype(np.float32)
            b = live[y:y + tile, x:x + tile].astype(np.float32)
            if a.std() < 8 or b.std() < 8:
                continue
            a = (a - a.mean()) / a.std()
            b = (b - b.mean()) / b.std()
            (dx, dy), response = cv.phaseCorrelate(a, b, window)
            if response >= min_response:
                shifts.append(np.hypot(dx, dy))
    if len(shifts) < 4:
        return None, len(shifts)
    return float(np.median(shifts)), len(shifts)


def overlay(reference, live):
    """Reference in red, live in cyan: grey where they agree."""
    return cv.merge([live, live, reference])


# ---------------------------------------------------------------- board check

def board_poses(images, detector, object_points, cams):
    """serial -> (R_board_to_cam, t, ids, px) for cameras that see the board."""
    poses = {}
    for serial, gray in images.items():
        if serial not in cams:
            continue
        ids, px, _ = detect(detector, gray)
        if len(ids) < MIN_BOARD_CORNERS:
            continue
        cam = cams[serial]
        ok, rvec, tvec = cv.solvePnP(object_points[ids], px, cam["K"], cam["dist"],
                                     flags=cv.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        R, _ = cv.Rodrigues(rvec)
        poses[serial] = (R, tvec.reshape(3), ids, px)
    return poses


def cross_errors(poses, object_points, cams):
    """err[i][j]: RMS px of camera i's board pose carried into camera j."""
    serials = sorted(poses, key=lambda s: cams[s]["index"])
    err = {}
    for si in serials:
        R_bi, t_bi, _, _ = poses[si]
        ci = cams[si]
        for sj in serials:
            if si == sj:
                continue
            cj = cams[sj]
            R_ij = cj["R"] @ ci["R"].T
            t_ij = cj["t"] - R_ij @ ci["t"]
            _, _, ids_j, px_j = poses[sj]
            points_i = object_points[ids_j] @ R_bi.T + t_bi
            points_j = points_i @ R_ij.T + t_ij
            projected, _ = cv.projectPoints(points_j, np.zeros(3), np.zeros(3),
                                            cj["K"], cj["dist"])
            residual = projected.reshape(-1, 2) - px_j
            err[(si, sj)] = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    return serials, err


def board_in_world(poses, object_points, cams):
    """Per camera: board corners in world -> (mean z mm, tilt deg, centre xy)."""
    out = {}
    for serial, (R_b, t_b, _, _) in poses.items():
        cam = cams[serial]
        points_cam = object_points @ R_b.T + t_b
        points_w = (points_cam - cam["t"]) @ cam["R"]   # R^T (X - t), row-wise
        normal = (cam["R"].T @ R_b)[:, 2]
        tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
        out[serial] = (float(points_w[:, 2].mean()), tilt, points_w[:, :2].mean(axis=0))
    return out


def print_matrix(title, serials, err, cams):
    print(f"\n  {title}  (row = source camera, column = camera checked; RMS px)")
    print("          " + "".join(f"{'cam' + str(cams[s]['index']):>9}" for s in serials))
    for si in serials:
        cells = [
            f"{'-':>9}" if si == sj else f"{err[(si, sj)]:9.2f}" for sj in serials
        ]
        print(f"    cam{cams[si]['index']}  " + "".join(cells))


def per_camera(serials, err):
    """Median error over every pair a camera takes part in."""
    return {
        s: float(np.median([v for (a, b), v in err.items() if s in (a, b)]))
        for s in serials
    }


# ---------------------------------------------------------------- main

def verdict(px):
    if px is None:
        return "?? no match"
    if px < UNCHANGED_PX:
        return "unchanged"
    if px < MOVED_PX:
        return "small shift"
    return "MOVED"


def main() -> int:
    args = parse_args()
    calib, cams = load_calibration(args.calibration)
    session = args.session or Path(calib["source"]["session"])
    world_frames = calib.get("world", {}).get("frames") or [0]
    ref_index = args.reference_frame if args.reference_frame is not None else world_frames[0]
    serials = sorted(cams, key=lambda s: cams[s]["index"])
    print(f"calibration: {args.calibration}  (solved {calib.get('created_utc', '?')})")
    print(f"reference:   {session}  frame {ref_index}")

    references = {}
    for s in serials:
        image = load_reference(session, s, ref_index)
        if image is None:
            print(f"  cam{cams[s]['index']} ({s}): reference image not found")
        else:
            references[s] = image
    if not references:
        print("No reference images; pass --session <calibration session folder>.")
        return 1

    if args.live:
        live = {}
        for s in serials:
            path = args.live / f"camera_{s}.png"
            if path.exists():
                live[s] = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        print(f"live:        {args.live}")
    else:
        print(f"live:        grabbing {args.frames} frame(s) per camera ...")
        live = grab_live(serials, args)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output / stamp
    out.mkdir(parents=True, exist_ok=True)

    spec = BoardSpec.load(args.board)
    detector, object_points = make_detector(spec)

    # 1. scene check
    print("\n1) Scene check against the calibration-day image (board region masked)")
    scene = {}
    for s in serials:
        if s not in references or s not in live:
            print(f"    cam{cams[s]['index']} ({s}): no {'live' if s in references else 'reference'} image")
            continue
        ref, now = references[s], live[s]
        if ref.shape != now.shape:
            print(f"    cam{cams[s]['index']} ({s}): image size changed "
                  f"{ref.shape[::-1]} -> {now.shape[::-1]}; ROI or binning differs")
            continue
        _, _, ref_markers = detect(detector, ref)
        _, _, live_markers = detect(detector, now)
        mask_ref = board_mask(ref.shape, ref_markers)
        mask_live = board_mask(now.shape, live_markers)
        px, support = scene_shift(ref, now, mask_ref, mask_live)
        method = "matched features"
        if px is None or support < 200:
            # Few ORB matches: the new scene hides most of the old one, and a
            # homography on a handful of matches is not trustworthy.
            px, support = tile_shift(ref, now, np.minimum(mask_ref, mask_live))
            method = "confident background tiles"
        scene[s] = px
        shown = f"{px:6.2f} px" if px is not None else "    --   "
        print(f"    cam{cams[s]['index']} ({s}): {shown}  ({support} {method})"
              f"  -> {verdict(px)}")
        cv.imwrite(str(out / f"camera_{s}.png"), now)
        cv.imwrite(str(out / f"reference_{s}.png"), ref)
        cv.imwrite(str(out / f"overlay_{s}.jpg"), overlay(ref, now))

    # 2. board check
    print("\n2) Board check (board flat on the tactile array, seen by >= 2 cameras)")
    now_poses = board_poses(live, detector, object_points, cams)
    if len(now_poses) < 2:
        seen = ", ".join(f"cam{cams[s]['index']}" for s in now_poses) or "none"
        print(f"    skipped: board fully detected by {seen}. Lay the board flat on "
              "the sensor so every camera sees it, then re-run for the stronger check.")
    else:
        ref_poses = board_poses(references, detector, object_points, cams)
        ref_serials, ref_err = cross_errors(ref_poses, object_points, cams)
        now_serials, now_err = cross_errors(now_poses, object_points, cams)
        if len(ref_poses) >= 2:
            print_matrix("calibration day", ref_serials, ref_err, cams)
        print_matrix("now", now_serials, now_err, cams)
        ref_cam = per_camera(ref_serials, ref_err) if len(ref_poses) >= 2 else {}
        now_cam = per_camera(now_serials, now_err)
        print("\n    per camera (median over its pairs):")
        for s in now_serials:
            then = ref_cam.get(s)
            ratio = now_cam[s] / then if then else None
            flag = "MOVED?" if ratio is not None and ratio > 3 and now_cam[s] > 2 else ""
            then_text = f"{then:6.2f}" if then is not None else "   -- "
            print(f"      cam{cams[s]['index']}: then {then_text} px   now "
                  f"{now_cam[s]:6.2f} px   {flag}")

        then_world = board_in_world(ref_poses, object_points, cams)
        now_world = board_in_world(now_poses, object_points, cams)
        print("\n    board surface in the world frame (z = height above the array):")
        for s in now_serials:
            z, tilt, xy = now_world[s]
            line = (f"      cam{cams[s]['index']}: z {z:+7.2f} mm  tilt {tilt:5.2f} deg  "
                    f"centre ({xy[0]:+.1f}, {xy[1]:+.1f}) mm")
            if s in then_world:
                line += f"   [calibration day z {then_world[s][0]:+.2f} mm]"
            print(line)
        z_now = np.median([v[0] for v in now_world.values()])
        if then_world:
            z_then = np.median([v[0] for v in then_world.values()])
            print(f"    surface height changed by {z_now - z_then:+.2f} mm "
                  "(a different sensor stack; fix with --world-z-offset or a new "
                  "world segment if this matters)")

    print(f"\nimages and red/cyan overlays: {out}")
    moved = [s for s, px in scene.items() if px is not None and px >= MOVED_PX]
    unknown = [s for s in serials if scene.get(s) is None]
    if moved:
        print("VERDICT: camera(s) moved -> recalibrate: "
              + ", ".join(f"cam{cams[s]['index']}" for s in moved))
    elif unknown:
        print("VERDICT: inconclusive for "
              + ", ".join(f"cam{cams[s]['index']}" for s in unknown)
              + "; check the overlays")
    else:
        print("VERDICT: no camera moved beyond "
              f"{MOVED_PX:g} px; the extrinsics still describe this rig.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
