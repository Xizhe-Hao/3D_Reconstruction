#!/usr/bin/env python3
"""Does the rig hold still? Watch the static scene drift, camera by camera.

RUN WITH THE **CAPTURE** ENVIRONMENT (needs PySpin):

    cd "sensor-data-collection\\Python files"
    .\\.venv\\Scripts\\Activate.ps1
    python ..\\..\\calibration\\check_stability.py --minutes 20

Why this exists
---------------
Extrinsics are a statement about where the cameras are. If a camera moves after
the solve -- or *during* it -- the calibration describes a rig that no longer
exists, and nothing downstream can detect that: the corner detector still works,
the board is still flat, the intrinsics are still right, and the numbers just
come out mysteriously bad.

One session failed exactly that way. Solved on any single capture step it gave
0.5-1.2 px; solved on all of them together, 7.9 px, because two cameras had
drifted between steps. The image evidence was unambiguous once looked for: over
half an hour the static background moved 22 px in one camera and 6.6 px in
another, while the other two moved 0.0 and 1.4 px.

This measures that directly and needs no board, no calibration and no capture:
it locks onto whatever is already in view -- the test frame, the fixture, the
bench -- and reports how far each camera has drifted from where it started.

    < 1 px    solid
    1-3 px    marginal; a short session may survive it
    > 3 px    fix the mount before calibrating -- at ~0.1 mm per pixel this is
              already the size of the deformation being measured

If drift only appears in the first minutes, it is the cameras warming up: let
them run and start when the numbers flatten. If it grows without settling, it is
mechanical -- the clamp, the ball head, the rail, or cable tension pulling on the
body.
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

from tune_exposure import prepare_for_capture, throttle  # noqa: E402

SOLID, MARGINAL = 1.0, 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=20.0,
                        help="how long to watch; a calibration session is ~30 min")
    parser.add_argument("--every", type=float, default=60.0, help="seconds between checks")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--trigger-hz", type=float, default=24.2)
    parser.add_argument("--bayer", default="RG")
    parser.add_argument("--mm-per-px", type=float, default=0.1,
                        help="object-plane scale, for reporting drift in millimetres")
    return parser.parse_args()


def drift_px(reference, current, orb, matcher, mask=None):
    """Median displacement of the scene between two frames of the same camera.

    A homography over ORB matches, rather than a plain image correlation: it is
    unaffected by anything that moved inside the frame (a hand, the board) as
    long as most of what is visible is the static rig, which RANSAC enforces. That
    assumption is what `mask` exists for: on an idle rig nothing needs masking,
    but a large, feature-dense object moving through the frame -- a calibration
    board, say -- can out-vote the background and become the dominant motion.
    """
    ka, da = reference
    kb, db = orb.detectAndCompute(current, mask)
    if da is None or db is None or len(db) < 30:
        return None, 0
    matches = sorted(matcher.match(da, db), key=lambda m: m.distance)[:800]
    if len(matches) < 30:
        return None, 0
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


def verdict(px):
    if px is None:
        return "  ?  "
    if px < SOLID:
        return "solid"
    if px < MARGINAL:
        return "marg."
    return "DRIFT"


def main() -> int:
    args = parse_args()
    bayer = getattr(cv, f"COLOR_Bayer{args.bayer}2GRAY")

    system = PySpin.System.GetInstance()
    cameras = system.GetCameras()
    if cameras.GetSize() == 0:
        print("no cameras found (is SpinView open?)")
        cameras.Clear(); system.ReleaseInstance()
        return 1

    handles = []
    for index in range(cameras.GetSize()):
        camera = cameras[index]
        camera.Init()
        node = PySpin.CStringPtr(camera.GetTLDeviceNodeMap().GetNode("DeviceSerialNumber"))
        handles.append((node.GetValue() if PySpin.IsReadable(node) else "?", camera))
    handles.sort(key=lambda item: item[0])

    orb = cv.ORB_create(6000)
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    reference, history = {}, {s: [] for s, _ in handles}

    def grab(camera):
        try:
            image = camera.GetNextImage(args.timeout_ms)
        except PySpin.SpinnakerException:
            return None
        try:
            if image.IsIncomplete():
                return None
            raw = image.GetNDArray()
            return cv.cvtColor(raw, bayer) if raw.ndim == 2 else cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        finally:
            image.Release()

    print(f"{len(handles)} camera(s): " + ", ".join(s for s, _ in handles))
    print(f"watching for {args.minutes:g} min, sampling every {args.every:g} s.")
    print("Leave the rig ALONE -- do not touch a camera, the bench or a cable.")
    print("Anything may move inside the frame; the fit ignores it.\n")

    try:
        for _, camera in handles:
            throttle(camera, args.fps)
            camera.BeginAcquisition()

        for _, camera in handles:            # discard the first frames after start
            grab(camera)
        for serial, camera in handles:
            frame = grab(camera)
            if frame is None:
                print(f"  {serial}: no frame; skipping")
                continue
            reference[serial] = orb.detectAndCompute(frame, None)

        print("  " + "elapsed".rjust(8) + "".join(f"{s[-4:]:>12}" for s, _ in handles))
        started = time.monotonic()
        while time.monotonic() - started < args.minutes * 60:
            target = time.monotonic() + args.every
            while time.monotonic() < target:
                for _, camera in handles:     # keep the streams fresh
                    grab(camera)
                time.sleep(0.2)
            row = []
            for serial, camera in handles:
                frame = grab(camera)
                if frame is None or serial not in reference:
                    row.append(f"{'--':>12}"); continue
                px, _ = drift_px(reference[serial], frame, orb, matcher)
                history[serial].append(px)
                row.append(f"{px:8.2f} px" if px is not None else f"{'--':>12}")
            print(f"  {(time.monotonic() - started) / 60:6.1f}m " + "".join(row))
    except KeyboardInterrupt:
        print()
    finally:
        for _, camera in handles:
            try:
                camera.EndAcquisition()
            except PySpin.SpinnakerException:
                pass

        print()
        print(f"{'camera':>12} {'final drift':>12} {'worst':>9} {'verdict':>9}")
        bad = []
        for serial, _ in handles:
            seen = [v for v in history[serial] if v is not None]
            if not seen:
                print(f"{serial:>12} {'--':>12}"); continue
            final, worst = seen[-1], max(seen)
            note = ""
            if worst >= MARGINAL:
                bad.append(serial); note = "  <- secure this mount"
            print(f"{serial:>12} {final:9.2f} px {worst:6.2f} px {verdict(worst):>9}"
                  f"   ({final * args.mm_per_px:.2f} mm at the object){note}")

        # throttle() capped the free-run rate; a camera left capped silently drops
        # every hardware trigger that lands inside its enforced idle period.
        for _, camera in handles:
            try:
                prepare_for_capture(camera, args.trigger_hz)
            except PySpin.SpinnakerException:
                pass
            camera.DeInit()
        del camera, handles
        cameras.Clear()
        system.ReleaseInstance()

        if bad:
            print()
            print(f"{len(bad)} camera(s) drifted past {MARGINAL:g} px: {', '.join(bad)}")
            print("Calibrating now would solve for a rig that stops existing halfway")
            print("through the session -- every step fits itself and none of them fit")
            print("each other. Secure the mounts first, then run this again.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
