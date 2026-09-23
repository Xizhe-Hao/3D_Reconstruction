"""Grade a guided capture session the moment it finishes.

Used by `capture_gui.py`, which runs in the CAPTURE environment. The solve
itself cannot: `calibrateMultiview` needs OpenCV 5 and numpy 2, while PySpin
pins numpy 1.26. So the three solver scripts are run as subprocesses under the
calibration interpreter and their output is summarised here.

Two things are reported, and the first matters more than the second.

**Did the rig hold still?** Extrinsics are a claim about where the cameras are.
If one moves during the half hour the session takes, no calibration of it
exists: every capture step fits itself and none of them fit each other. Nothing
downstream detects this -- the corners still detect, the board is still flat,
the intrinsics are still right, and the residual is merely large. Measuring it
is easy and costs nothing, because the session already contains the evidence:
the static scene behind the board must not move between the first frame and the
last.

**Then, how good is the calibration?** `validate.py` already owns the thresholds
and prints a verdict per row, so this parses that output rather than restating
the numbers -- there is exactly one definition of "good enough" in the tree.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import cv2 as cv
except ImportError:  # pragma: no cover - the caller has already failed by now
    cv = None

VERDICT_LINE = re.compile(r"^\s*\[(PASS|WARN|FAIL)\]\s+(.+?)\s\s+(\S+)\s+(\S+)?\s*\(want")
DRIFT_SOLID, DRIFT_BAD = 1.0, 3.0


# --------------------------------------------------------------- rig stability
def measure_drift(session_dir: Path, bayer: str = "RG") -> dict[str, float | None]:
    """Per-camera displacement of the STATIC scene, first saved frame to last.

    The board is masked out of both frames: it is the most feature-dense thing
    in view and it moves on purpose, so left in it can out-vote the background
    and RANSAC will happily report the board's motion instead of the camera's.
    """
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    orb = cv.ORB_create(6000)
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    code = getattr(cv, f"COLOR_Bayer{bayer}2GRAY")
    out: dict[str, float | None] = {}

    for position, serial in enumerate(meta["camera_serials"]):
        folder = session_dir / f"camera_{position}_{serial}"
        frames = sorted(folder.glob("*.bmp"))
        if len(frames) < 4:
            out[serial] = None
            continue
        def load(path):
            raw = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            if raw is None:
                return None
            gray = (cv.cvtColor(raw, code) if raw.ndim == 2
                    else cv.cvtColor(raw, cv.COLOR_BGR2GRAY))
            return gray, _board_mask(gray, orb)

        reference = load(frames[1])
        if reference is None:
            out[serial] = None
            continue
        # Several late frames, not one. The mask is an intensity heuristic, and a
        # single frame where it misses the board reports the BOARD moving instead
        # of the camera. A median across the second half is immune to that.
        late = frames[len(frames) // 2:]
        values = []
        for path in late[::max(1, len(late) // 8)]:
            current = load(path)
            if current is None:
                continue
            value = _displacement(reference, current, orb, matcher)
            if value is not None:
                values.append(value)
        out[serial] = float(np.median(values)) if values else None
    return out


def _board_mask(gray, orb):
    """Mask out the bright printed target, keeping the rig behind it."""
    # The board is a large, near-white, high-contrast region; a plain intensity
    # threshold finds it without needing the detector, and over-masking costs
    # nothing here because the rest of the frame carries plenty of features.
    _, hot = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
    hot = cv.dilate(hot, np.ones((41, 41), np.uint8))
    contours, _ = cv.findContours(hot, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.full(gray.shape, 255, np.uint8)
    for contour in contours:
        if cv.contourArea(contour) > 0.02 * gray.size:
            x, y, w, h = cv.boundingRect(contour)
            mask[y:y + h, x:x + w] = 0
    return mask if mask.mean() > 60 else None      # never mask the whole frame


def _displacement(first, second, orb, matcher):
    ka, da = orb.detectAndCompute(first[0], first[1])
    kb, db = orb.detectAndCompute(second[0], second[1])
    if da is None or db is None or len(da) < 30 or len(db) < 30:
        return None
    matches = sorted(matcher.match(da, db), key=lambda m: m.distance)[:800]
    if len(matches) < 30:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, inliers = cv.findHomography(src, dst, cv.RANSAC, 3.0)
    if inliers is None or inliers.sum() < 20:
        return None
    keep = inliers.ravel().astype(bool)
    shift = np.linalg.norm(dst[keep].reshape(-1, 2) - src[keep].reshape(-1, 2), axis=1)
    return float(np.median(shift))


# ------------------------------------------------------------------- the solve
def find_calib_python(explicit: Path | None) -> Path | None:
    if explicit:
        return explicit if explicit.exists() else None
    root = Path(__file__).resolve().parent.parent
    for candidate in (root / ".venv-calib" / "Scripts" / "python.exe",
                      root / ".venv-calib" / "bin" / "python"):
        if candidate.exists():
            return candidate
    return None


def _run(argv, label, quiet=False):
    if not quiet:
        print(f"  {label:<16} ", end="", flush=True)
    done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    # validate.py exits non-zero when a check FAILs, which is a result, not a
    # crash -- its output is exactly what we came for.
    ok = done.returncode == 0 or label == "validate"
    if not quiet:
        print("done" if ok else "FAILED")
    if not ok:
        for line in ((done.stdout or "").strip().splitlines()[-4:]
                     + (done.stderr or "").strip().splitlines()[-4:]):
            print(f"      {line}")
        return None
    return done.stdout or ""


def solve_and_grade(session: Path, board: Path, calib_python: Path,
                    world_frames: str, extra_calibrate: list[str] | None = None):
    """Run detect -> calibrate -> validate and return validate's stdout."""
    here = Path(__file__).resolve().parent
    detections = session / "detections.npz"
    calibration = session / "calibration.json"

    if _run([str(calib_python), str(here / "detect_corners.py"), str(session),
             "--board", str(board), "--out", str(detections)], "detect corners") is None:
        return None
    base = [str(calib_python), str(here / "calibrate.py"), str(detections),
            "--out", str(calibration), "--world-frames", world_frames,
            "--world-flip-z"] + (extra_calibrate or [])
    if _run(base, "calibrate") is None:
        # The strict solve needs frames in which a camera measured the WHOLE
        # board. When too few exist, calibrate.py says so and names the escape
        # hatch; take it rather than reporting nothing, but say that we did --
        # a sub-pattern gives a weaker per-frame pose than the full board.
        print("  calibrate        retrying on the shared sub-pattern "
              "(--common-corners 40)")
        if _run(base + ["--common-corners", "40"], "calibrate", quiet=True) is None:
            print("  calibrate        FAILED even on the sub-pattern")
            return None
        print("  calibrate        done (sub-pattern -- expect weaker extrinsics)")
    return _run([str(calib_python), str(here / "validate.py"), str(calibration),
                 str(detections), "--report", str(session / "metrics.json")], "validate")


# ---------------------------------------------------------------------- report
def report(session: Path, board: Path, calib_python: Path | None,
           steps: list[tuple[str, int, int]]) -> int:
    """Print the whole verdict. Returns a process exit code."""
    bar = "=" * 72
    print()
    print(bar)
    print("SESSION REPORT")
    print(bar)

    incomplete = [(k, c, n) for k, c, n in steps if c < n]
    total = sum(c for _, c, _ in steps)
    print(f"\ncapture: {total} frame sets, "
          f"{len(steps) - len(incomplete)}/{len(steps)} steps complete")
    for key, got, want in incomplete:
        print(f"  step '{key}' stopped at {got}/{want}")

    print("\nrig stability -- how far the STATIC scene moved during the session")
    print("  (a camera that moves mid-session cannot be calibrated at all)")
    drift = measure_drift(session)
    moved = []
    for serial, value in drift.items():
        if value is None:
            print(f"  {serial}   -- (not enough background features)")
            continue
        tag = "solid" if value < DRIFT_SOLID else ("marginal" if value < DRIFT_BAD else "MOVED")
        print(f"  {serial} {value:7.2f} px   {tag}")
        if value >= DRIFT_BAD:
            moved.append(serial)

    verdict_lines, failures, warnings = [], 0, 0
    if calib_python is None:
        print("\nsolve: skipped (calibration interpreter not found; pass --calib-python)")
    else:
        meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
        low, high = meta.get("world_frames", [0, 0])
        print(f"\nsolving (this takes a few minutes) -- world frames {low}:{high}")
        out = solve_and_grade(session, board, calib_python, f"{low}:{high}")
        if out:
            print("\ncalibration quality")
            for line in out.splitlines():
                match = VERDICT_LINE.match(line)
                if match:
                    verdict_lines.append(line.rstrip())
                    failures += match.group(1) == "FAIL"
                    warnings += match.group(1) == "WARN"
            for line in verdict_lines:
                print(line.rstrip())

    print("\n" + bar)
    # Drift is a diagnosis, not the verdict. Cameras that move TOGETHER keep their
    # relative pose, and relative pose is the whole content of the extrinsics -- so
    # a large image shift does not by itself mean the calibration is wrong. Let the
    # quality table judge, and use drift to explain what it found.
    if moved and failures:
        print(f"VERDICT: UNUSABLE -- {failures} check(s) failed, and "
              f"{', '.join(moved)} moved during the session.")
        print("Movement is the first thing to fix: when the rig changes mid-session,")
        print("every step fits itself and none of them fit each other, which no amount")
        print("of re-shooting repairs. Secure those mounts, confirm with")
        print("check_stability.py, then capture again.")
        code = 1
    elif moved:
        print(f"VERDICT: MARGINAL -- every check passed, but "
              f"{', '.join(moved)} moved during the session.")
        print("They may have moved together, which leaves the relative pose intact.")
        print("Usable, but secure the mounts before the numbers stop being lucky.")
        code = 0
    elif calib_python is None or not verdict_lines:
        print("VERDICT: not graded -- the solve did not complete. Run detect_corners /")
        print("calibrate / validate by hand to see why.")
        code = 1
    elif failures:
        print(f"VERDICT: NOT USABLE -- {failures} check(s) failed"
              + (f", {warnings} marginal" if warnings else "") + ".")
        print("Do not collect training data with this: the cameras are the shape")
        print("ground truth, so calibration error becomes label error you cannot")
        print("detect later. See the failing rows above.")
        code = 1
    elif warnings:
        print(f"VERDICT: MARGINAL -- {warnings} check(s) are close to the limit.")
        print("Usable for a rehearsal, not for a dataset you intend to keep.")
        code = 0
    else:
        print("VERDICT: GOOD -- every check passed. Safe to collect data.")
        code = 0
    print(bar)
    print(f"\nsession      {session}")
    print(f"calibration  {session / 'calibration.json'}")
    print("\nlook at it:")
    print(f"  python calibration\\inspect_calib.py {session / 'calibration.json'} "
          f"{session / 'detections.npz'} --session {session}")
    return code
