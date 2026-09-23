#!/usr/bin/env python3
"""Live feedback while you hold the calibration board, one camera at a time.

RUN WITH THE **CAPTURE** ENVIRONMENT (needs PySpin):

    cd "sensor-data-collection\\Python files"
    .\\.venv\\Scripts\\Activate.ps1
    python ..\\..\\calibration\\pose_helper.py --board ..\\..\\calibration\\board\\board.json

Intrinsics are only well determined when a camera sees the board close to
square-on. Judging that by eye from behind the rig is hopeless -- a board that
looks "tilted a bit" to you is still 70 degrees edge-on to a low-mounted camera.
So this prints a live number per camera:

    facing = 0.00  board seen edge-on, useless for intrinsics
    facing = 1.00  board perfectly square-on to that camera

Rotate the board until the camera you are aiming at reads **above 0.55**, hold
it still for a second, then move on. The bar shows the live value, `best` is the
highest you have reached, and `held` counts the distinct good poses banked so
far. Aim for 6-8 held poses per camera, at different tilts and distances.

The value comes from the homography between the board and the image, so it needs
no camera calibration -- it works before anything has been solved.

Cameras are left ready for a triggered capture when you quit with Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2 as cv
except ImportError:
    sys.exit("OpenCV missing in the capture environment:\n"
             "  python -m pip install opencv-python==4.9.0.80")
try:
    import PySpin
except ImportError:
    sys.exit("PySpin not found -- activate the capture environment (.venv).")

from charuco_board import BoardSpec  # noqa: E402
from tune_exposure import prepare_for_capture, throttle  # noqa: E402

GOOD = 0.55


def make_detector(board):
    """Local copy: charuco_board.make_detector targets the OpenCV 5 API."""
    detector_params = cv.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
    charuco_params = cv.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    return cv.aruco.CharucoDetector(board, charuco_params, detector_params,
                                    cv.aruco.RefineParameters())


def build_board(spec: BoardSpec):
    dictionary = cv.aruco.getPredefinedDictionary(
        getattr(cv.aruco, spec.dictionary))
    board = cv.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        float(spec.square_mm), float(spec.marker_mm), dictionary)
    board.setLegacyPattern(bool(spec.legacy_pattern))
    return board


def facing_score(object_points, image_points):
    if len(object_points) < 8:
        return None
    homography, _ = cv.findHomography(
        np.ascontiguousarray(object_points[:, :2], np.float64),
        np.ascontiguousarray(image_points, np.float64), 0)
    if homography is None:
        return None
    singular = np.linalg.svd(homography[:2, :2], compute_uv=False)
    if singular[0] < 1e-9:
        return None
    return float(singular[1] / singular[0])


def bar(value: float, width: int = 12) -> str:
    filled = int(round(np.clip(value, 0.0, 1.0) * width))
    return "#" * filled + "-" * (width - filled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--hold-seconds", type=float, default=0.8,
                        help="how long a good pose must be held before it counts")
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--trigger-hz", type=float, default=24.2)
    args = parser.parse_args()

    spec = BoardSpec.load(args.board)
    board = build_board(spec)
    object_points = np.asarray(board.getChessboardCorners(), np.float32).reshape(-1, 3)
    print(f"board: {spec.describe()}")

    system = PySpin.System.GetInstance()
    cameras = system.GetCameras()
    if cameras.GetSize() == 0:
        print("no cameras found (is SpinView open?)")
        cameras.Clear()
        system.ReleaseInstance()
        return 1

    handles = []
    for index in range(cameras.GetSize()):
        camera = cameras[index]
        camera.Init()
        node = PySpin.CStringPtr(
            camera.GetTLDeviceNodeMap().GetNode("DeviceSerialNumber"))
        handles.append((node.GetValue() if PySpin.IsReadable(node) else "?", camera))
    handles.sort(key=lambda item: item[0])

    detectors = {serial: make_detector(board) for serial, _ in handles}
    best = {serial: 0.0 for serial, _ in handles}
    held = {serial: 0 for serial, _ in handles}
    good_since = {serial: None for serial, _ in handles}
    banked = {serial: -1.0 for serial, _ in handles}

    print()
    print("Hold the board so it FACES one camera; rotate until that camera goes")
    print(f"above {GOOD:.2f}, hold it still, then move to the next camera.")
    print("Target: 6-8 held poses each, at different tilts and distances.")
    print("Ctrl+C when done.")
    print()

    try:
        for _, camera in handles:
            throttle(camera, args.fps)
            camera.BeginAcquisition()
        while True:
            line = []
            for serial, camera in handles:
                score = None
                try:
                    image = camera.GetNextImage(args.timeout_ms)
                except PySpin.SpinnakerException:
                    image = None
                if image is not None:
                    try:
                        if not image.IsIncomplete():
                            raw = image.GetNDArray()
                            gray = cv.cvtColor(raw, cv.COLOR_BayerBG2GRAY) \
                                if raw.ndim == 2 else cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
                            corners, ids, _, _ = detectors[serial].detectBoard(gray)
                            if ids is not None and len(ids) >= 8:
                                score = facing_score(
                                    object_points[np.asarray(ids).reshape(-1)],
                                    np.asarray(corners).reshape(-1, 2))
                    finally:
                        image.Release()

                now = time.monotonic()
                if score is None:
                    good_since[serial] = None
                    line.append(f"{serial[-4:]} ....  ")
                    continue
                best[serial] = max(best[serial], score)
                if score > GOOD:
                    if good_since[serial] is None:
                        good_since[serial] = now
                    elif (now - good_since[serial] >= args.hold_seconds
                          and abs(score - banked[serial]) > 0.06):
                        held[serial] += 1
                        banked[serial] = score
                        good_since[serial] = now
                else:
                    good_since[serial] = None
                mark = "OK" if score > GOOD else "  "
                line.append(f"{serial[-4:]} {score:.2f}{mark}")
            status = "  |  ".join(line)
            totals = " ".join(f"{s[-4:]}:{held[s]}" for s, _ in handles)
            sys.stdout.write("\r" + status + f"   held[{totals}]   ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print()
    finally:
        for _, camera in handles:
            try:
                camera.EndAcquisition()
            except PySpin.SpinnakerException:
                pass
        print()
        print(f"{'camera':>12} {'best facing':>12} {'poses held':>11}")
        for serial, _ in handles:
            flag = "" if held[serial] >= 6 else "   <- need more"
            print(f"{serial:>12} {best[serial]:12.2f} {held[serial]:11d}{flag}")
        for _, camera in handles:
            prepare_for_capture(camera, args.trigger_hz)
        print()
        print("cameras left ready for a triggered capture")
        while handles:
            _, camera = handles.pop()
            try:
                camera.DeInit()
            except PySpin.SpinnakerException:
                pass
            del camera
        cameras.Clear()
        system.ReleaseInstance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
