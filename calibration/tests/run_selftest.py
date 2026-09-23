#!/usr/bin/env python3
"""End-to-end self-test of the calibration pipeline against known ground truth.

    python tests/run_selftest.py --work-dir <scratch>

Renders a synthetic four-camera rig, runs the real `detect_corners.py`,
`calibrate.py` and `validate.py` on it, and checks the recovered intrinsics,
extrinsics and world frame against the values used to render. Exits non-zero if
anything drifts outside tolerance, so it can be run after any edit to the
scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from geometry import rotation_angle_deg  # noqa: E402
from session_io import GENICAM_TO_OPENCV_BAYER  # noqa: E402

PYTHON = sys.executable
# Tolerances are set from the precision this synthetic rig can actually reach
# (960x720 sensor, ~0.2 px corner noise, ~120 board poses per camera), not from
# wishful thinking. Any real defect -- a sign flip, a swapped Bayer pattern, a
# unit error, a wrong world transform -- misses these by orders of magnitude.
# Focal length and camera distance are coupled: a 0.25 % focal error shows up as
# a ~0.25 % range error, which is why `position_mm` is looser than it looks.
TOLERANCES = {
    "focal_relative": 0.005,        # 0.5 %
    "principal_point_px": 5.0,
    "k1_absolute": 0.012,
    "position_mm": 2.0,
    "rotation_deg": 0.20,
    "distance_p95_mm": 0.15,
}

results: list[tuple[bool, str]] = []


def check(passed: bool, message: str) -> None:
    results.append((bool(passed), message))
    print(f"  [{'ok  ' if passed else 'FAIL'}] {message}")


def run(script: str, *arguments: str) -> None:
    command = [PYTHON, str(ROOT / script), *[str(a) for a in arguments]]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(f"{script} failed with exit code {completed.returncode}")


def test_bayer_mapping() -> None:
    """GenICam names a Bayer pattern by its first row, OpenCV by its second.

    Build a mosaic in the layout Spinnaker calls BayerRG8 and confirm that the
    OpenCV code this repository maps it to actually recovers the colours. Using
    OpenCV's identically-named constant instead would swap red and blue.
    """
    print("bayer naming")
    rng = np.random.RandomState(0)
    height, width = 64, 64
    bgr = np.zeros((height, width, 3), np.uint8)
    bgr[:, :, 0] = 40      # blue
    bgr[:, :, 1] = 120     # green
    bgr[:, :, 2] = 220     # red
    bgr = np.clip(bgr.astype(int) + rng.randint(-3, 4, bgr.shape), 0, 255).astype(np.uint8)

    raw = np.empty((height, width), np.uint8)
    raw[0::2, 0::2] = bgr[0::2, 0::2, 2]   # R
    raw[0::2, 1::2] = bgr[0::2, 1::2, 1]   # G
    raw[1::2, 0::2] = bgr[1::2, 0::2, 1]   # G
    raw[1::2, 1::2] = bgr[1::2, 1::2, 0]   # B

    code = GENICAM_TO_OPENCV_BAYER["BayerRG"]
    check(code == "BG", f"GenICam BayerRG maps to OpenCV Bayer{code}")

    correct = cv.cvtColor(raw, getattr(cv, f"COLOR_Bayer{code}2BGR"))
    wrong = cv.cvtColor(raw, cv.COLOR_BayerRG2BGR)
    inner = (slice(4, -4), slice(4, -4))
    correct_error = float(np.abs(correct[inner].astype(int) - bgr[inner].astype(int)).mean())
    wrong_error = float(np.abs(wrong[inner].astype(int) - bgr[inner].astype(int)).mean())
    check(correct_error < 5.0, f"mapped code recovers colours (mean error {correct_error:.1f})")
    check(wrong_error > 40.0,
          f"naively reusing the GenICam name would swap R/B (mean error {wrong_error:.1f})")


def compare_to_truth(calibration: dict, truth: dict, flip: bool) -> None:
    flip_matrix = np.diag([1.0, -1.0, -1.0]) if flip else np.eye(3)
    truth_by_serial = {camera["serial"]: camera for camera in truth["cameras"]}

    for camera in calibration["cameras"]:
        gt = truth_by_serial[camera["serial"]]
        name = f"cam {camera['index']} ({camera['serial']})"

        K = np.asarray(camera["K"])
        K_gt = np.asarray(gt["K"])
        focal_error = max(abs(K[0, 0] / K_gt[0, 0] - 1.0), abs(K[1, 1] / K_gt[1, 1] - 1.0))
        centre_error = float(np.hypot(K[0, 2] - K_gt[0, 2], K[1, 2] - K_gt[1, 2]))
        k1_error = abs(camera["dist"][0] - gt["dist"][0])
        check(focal_error < TOLERANCES["focal_relative"],
              f"{name} focal length within {focal_error * 100:.3f} %")
        check(centre_error < TOLERANCES["principal_point_px"],
              f"{name} principal point within {centre_error:.2f} px")
        check(k1_error < TOLERANCES["k1_absolute"], f"{name} k1 within {k1_error:.4f}")

        R = np.asarray(camera["R_world_to_cam"])
        R_expected = np.asarray(gt["R_world_to_cam"]) @ flip_matrix
        angle = rotation_angle_deg(R, R_expected)
        position = np.asarray(camera["position_world"])
        position_expected = flip_matrix @ np.asarray(gt["position_world"])
        offset = float(np.linalg.norm(position - position_expected))
        check(angle < TOLERANCES["rotation_deg"], f"{name} orientation within {angle:.4f} deg")
        check(offset < TOLERANCES["position_mm"], f"{name} position within {offset:.3f} mm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--wave-frames", type=int, default=120)
    parser.add_argument("--world-frames", type=int, default=10)
    parser.add_argument("--keep", action="store_true", help="keep the rendered session")
    args = parser.parse_args()

    work: Path = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    board_dir = work / "board"
    session = work / "session"
    detections = work / "detections.npz"

    test_bayer_mapping()

    print()
    print("rendering synthetic rig")
    run("make_board.py", "--out-dir", board_dir)
    if session.exists() and not args.keep:
        import shutil
        shutil.rmtree(session)
    run("tests/make_synthetic_session.py", "--out", session,
        "--board", board_dir / "board.json", "--cameras", args.cameras,
        "--wave-frames", args.wave_frames, "--world-frames", args.world_frames)

    print("detecting corners")
    run("detect_corners.py", session, "--board", board_dir / "board.json", "--out", detections)

    truth = json.loads((session / "ground_truth.json").read_text(encoding="utf-8"))
    world_range = f"{truth['world_frames'][0]}:{truth['world_frames'][1]}"

    print()
    print("solving (world frame = board, as solved)")
    plain = work / "calibration_plain.json"
    run("calibrate.py", detections, "--out", plain, "--world-frames", world_range)
    compare_to_truth(json.loads(plain.read_text(encoding="utf-8")), truth, flip=False)

    print()
    print("solving (--world-flip-z, the frame the rig actually wants)")
    flipped = work / "calibration.json"
    run("calibrate.py", detections, "--out", flipped, "--world-frames", world_range,
        "--world-flip-z")
    calibration = json.loads(flipped.read_text(encoding="utf-8"))
    compare_to_truth(calibration, truth, flip=True)
    check(all(camera["position_world"][2] > 0 for camera in calibration["cameras"]),
          "--world-flip-z puts every camera above the board plane")

    print()
    print("solving (--common-corners 24, the placeholder-free partial-view path)")
    common = work / "calibration_common.json"
    run("calibrate.py", detections, "--out", common, "--world-frames", world_range,
        "--world-flip-z", "--common-corners", 24)
    common_calibration = json.loads(common.read_text(encoding="utf-8"))
    compare_to_truth(common_calibration, truth, flip=True)
    check(common_calibration["metrics"]["multiview_rms_px"] < 1.0,
          f"sub-pattern solve stays converged "
          f"({common_calibration['metrics']['multiview_rms_px']:.3f} px)")

    print()
    print("validating on held-out frames")
    report = work / "validation.json"
    completed = subprocess.run(
        [PYTHON, str(ROOT / "validate.py"), str(flipped), str(detections),
         "--report", str(report)],
        cwd=ROOT, capture_output=True, text=True)
    print(completed.stdout[completed.stdout.find("accuracy"):] if "accuracy" in completed.stdout
          else completed.stdout)
    if completed.returncode not in (0, 1):
        print(completed.stderr, file=sys.stderr)
        raise SystemExit("validate.py crashed")
    check(completed.returncode == 0, "validate.py reports no FAIL")
    metrics = json.loads(report.read_text(encoding="utf-8"))
    check(metrics["distance_error_p95_mm"] < TOLERANCES["distance_p95_mm"],
          f"held-out metric error P95 {metrics['distance_error_p95_mm']:.4f} mm")

    print()
    failed = [message for ok, message in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        for message in failed:
            print(f"  FAILED: {message}")
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
