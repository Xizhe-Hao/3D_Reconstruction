#!/usr/bin/env python3
"""Live focus readout per camera -- turn the ring until the number bottoms out.

RUN WITH THE **CAPTURE** ENVIRONMENT (needs PySpin):

    cd "sensor-data-collection\\Python files"
    .\\.venv\\Scripts\\Activate.ps1
    python ..\\..\\calibration\\focus_check.py

Why this exists
---------------
A defocused camera does not announce itself. It still detects *some* ChArUco
corners, so the capture GUI happily banks poses, the session finishes, and the
failure only surfaces two steps later as a co-visibility matrix full of zeros --
because the blurred camera never saw enough of the board to share a frame with
anyone. One session was lost exactly that way: three cameras at ~2 px edge
width, one at 6.8 px, and no possible calibration because the camera graph was
disconnected.

The metric is the 10-90% rise width of a black/white edge, in pixels:

    < 2.2 px   sharp    -- what a focused Blackfly gives on a printed target
    2.2-3.2    soft     -- usable, but the corner refiner is losing accuracy
    > 3.2 px   DEFOCUSED-- the refiner is fitting a ramp, not an edge

`scene` measures the sharpest region anywhere in the frame, so it reads the
static rig at its true working distance and tells you whether the LENS is
focused. `board` measures the printed target itself. The two together separate
the two failure modes that look identical in a preview window:

    scene sharp, board blurry -> lens is fine, you are holding the board
                                 outside the depth of field
    scene blurry too          -> the lens itself is misfocused; turn the ring

Focus each camera on the tactile array (where the object will be), not on the
board. Then hold the board at that same distance.

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

SHARP = 2.2
SOFT = 3.2
TILE = 64
MIN_CONTRAST = 60.0   # a tile flatter than this carries no edge to measure


def edge_width(patch_range: np.ndarray, grad_max: np.ndarray) -> np.ndarray:
    """10-90% rise width of a step edge, in pixels.

    For a step of height `dI` blurred by any symmetric kernel, the peak
    gradient is dI / w where w is the effective width, so 0.8 * dI / grad
    recovers the 10-90% width without assuming the blur is Gaussian. This is
    resolution-independent and comparable across cameras, unlike the usual
    Laplacian-variance "focus score", which is dominated by how much contrast
    happens to be in the frame.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.8 * patch_range / grad_max


def scene_sharpness(gray: np.ndarray) -> float | None:
    """Edge width of the sharpest region anywhere in the frame."""
    h, w = gray.shape
    rows, cols = h // TILE, w // TILE
    if rows == 0 or cols == 0:
        return None
    g = gray[:rows * TILE, :cols * TILE].astype(np.float32)

    gx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.hypot(gx, gy)

    def tiles(a):
        return a.reshape(rows, TILE, cols, TILE).transpose(0, 2, 1, 3) \
                .reshape(rows * cols, TILE * TILE)

    flat, gflat = tiles(g), tiles(grad)
    lo = np.percentile(flat, 2, axis=1)
    hi = np.percentile(flat, 98, axis=1)
    span = hi - lo
    keep = (span >= MIN_CONTRAST) & (gflat.max(axis=1) > 1e-6)
    if not keep.any():
        return None
    widths = edge_width(span[keep], gflat.max(axis=1)[keep])
    # 5th percentile = the best-focused part of the scene, robust to one
    # freak tile of sensor noise.
    return float(np.percentile(widths, 5))


def board_sharpness(gray: np.ndarray, points: np.ndarray) -> float | None:
    """Edge width measured at the detected ChArUco corners themselves."""
    h, w = gray.shape
    widths = []
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        if not (12 <= x < w - 12 and 12 <= y < h - 12):
            continue
        patch = gray[y - 10:y + 11, x - 10:x + 11].astype(np.float32)
        lo, hi = np.percentile(patch, 5), np.percentile(patch, 95)
        if hi - lo < 40:
            continue
        gx = cv.Sobel(patch, cv.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv.Sobel(patch, cv.CV_32F, 0, 1, ksize=3) / 8.0
        gmax = float(np.hypot(gx, gy).max())
        if gmax > 1e-6:
            widths.append(float(edge_width(hi - lo, gmax)))
    return float(np.median(widths)) if len(widths) >= 8 else None


def build_board(spec: BoardSpec):
    dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, spec.dictionary))
    board = cv.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        float(spec.square_mm), float(spec.marker_mm), dictionary)
    board.setLegacyPattern(bool(spec.legacy_pattern))
    return board


def make_detector(board):
    """Local copy: charuco_board.make_detector targets the OpenCV 5 API."""
    detector_params = cv.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
    charuco_params = cv.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    return cv.aruco.CharucoDetector(board, charuco_params, detector_params,
                                    cv.aruco.RefineParameters())


def verdict(width: float | None) -> str:
    if width is None:
        return "  ?  "
    if width < SHARP:
        return "SHARP"
    if width < SOFT:
        return "soft "
    return "BLUR!"


def bar(width: float | None, span: int = 14) -> str:
    """Shorter bar = sharper. Saturates at 6 px so a wild reading stays legible."""
    if width is None:
        return "-" * span
    filled = int(round(np.clip((width - 1.0) / 5.0, 0.0, 1.0) * span))
    return "#" * filled + "." * (span - filled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", type=Path, default=None,
                        help="board.json; adds a second reading measured on the "
                             "printed target, which separates a misfocused lens "
                             "from a board held outside the depth of field")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--trigger-hz", type=float, default=24.2)
    parser.add_argument("--bayer", default="RG",
                        help="Bayer pattern of the raw frames")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    detector = None
    if args.board is not None:
        spec = BoardSpec.load(args.board)
        detector = make_detector(build_board(spec))
        print(f"board: {spec.describe()}")

    bayer_code = getattr(cv, f"COLOR_Bayer{args.bayer}2GRAY")

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

    best_scene = {serial: float("inf") for serial, _ in handles}
    best_board = {serial: float("inf") for serial, _ in handles}

    print()
    print(f"{len(handles)} camera(s): " + ", ".join(s for s, _ in handles))
    print()
    print("Point each camera at the TACTILE ARRAY and turn its focus ring until")
    print(f"`scene` bottoms out -- below {SHARP:g} px is sharp, above {SOFT:g} px is")
    print("defocused. `best` remembers the lowest value reached, so you can tell")
    print("when you have turned past the optimum. Ctrl+C when done.")
    print()

    try:
        for _, camera in handles:
            throttle(camera, args.fps)
            camera.BeginAcquisition()
        while True:
            line = []
            for serial, camera in handles:
                gray = None
                try:
                    image = camera.GetNextImage(args.timeout_ms)
                except PySpin.SpinnakerException:
                    image = None
                if image is not None:
                    try:
                        if not image.IsIncomplete():
                            raw = image.GetNDArray()
                            gray = (cv.cvtColor(raw, bayer_code) if raw.ndim == 2
                                    else cv.cvtColor(raw, cv.COLOR_BGR2GRAY))
                    finally:
                        image.Release()

                if gray is None:
                    line.append(f"{serial[-4:]} ......")
                    continue

                scene = scene_sharpness(gray)
                if scene is not None:
                    best_scene[serial] = min(best_scene[serial], scene)

                cell = f"{serial[-4:]} {bar(scene)} "
                cell += f"{scene:4.2f}" if scene is not None else " -- "
                cell += f" {verdict(scene)}"

                if detector is not None:
                    corners, ids, _, _ = detector.detectBoard(gray)
                    board_w = None
                    if ids is not None and len(ids) >= 8:
                        board_w = board_sharpness(
                            gray, np.asarray(corners).reshape(-1, 2))
                    if board_w is not None:
                        best_board[serial] = min(best_board[serial], board_w)
                        cell += f"  board {board_w:4.2f}"
                    else:
                        cell += "  board  -- "
                line.append(cell)

            sys.stdout.write("\r" + "  |  ".join(line) + "  ")
            sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        for _, camera in handles:
            try:
                camera.EndAcquisition()
            except PySpin.SpinnakerException:
                pass

        print()
        header = f"{'camera':>12} {'best scene':>11} {'verdict':>9}"
        if detector is not None:
            header += f" {'best board':>11}"
        print(header)
        failed = []
        for serial, _ in handles:
            scene = best_scene[serial]
            row = f"{serial:>12} "
            row += f"{scene:10.2f} " if np.isfinite(scene) else f"{'--':>10} "
            row += f"{verdict(scene if np.isfinite(scene) else None):>9}"
            if detector is not None:
                board_w = best_board[serial]
                row += f"{board_w:11.2f}" if np.isfinite(board_w) else f"{'--':>11}"
            if np.isfinite(scene) and scene >= SOFT:
                row += "   <- refocus this lens"
                failed.append(serial)
            print(row)

        # Restoring the capture state matters: `throttle` caps the free-run rate,
        # and a camera left capped drops every hardware trigger that lands inside
        # its enforced idle period -- recording zero images while reporting success.
        for _, camera in handles:
            try:
                prepare_for_capture(camera, args.trigger_hz)
            except PySpin.SpinnakerException:
                pass
            camera.DeInit()
        del camera, handles
        cameras.Clear()
        system.ReleaseInstance()

        if failed:
            print()
            print(f"{len(failed)} camera(s) still defocused: {', '.join(failed)}")
            print("Do NOT capture yet -- a blurred camera detects too little of the")
            print("board to share a frame with its neighbours, which disconnects the")
            print("camera graph and makes the extrinsics unsolvable.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
