#!/usr/bin/env python3
"""Find a working exposure before spending a capture session on it.

RUN THIS WITH THE **CAPTURE** ENVIRONMENT (it needs PySpin, not OpenCV):

    cd "sensor-data-collection\\Python files"
    .\\.venv\\Scripts\\Activate.ps1

    # try several exposures, one grab each, and save previews
    python ..\\..\\calibration\\tune_exposure.py --sweep 6000,12000,20000,30000 --gain-db 4 ^
        --preview-dir C:\\tactile_calib\\exposure

    # commit to one value and leave the cameras set to it
    python ..\\..\\calibration\\tune_exposure.py --set 20000 --gain-db 4

The cameras are put into free-run for the grab and left with `TriggerMode Off`,
which is what `capture_3blackfly_sensor_force.py` expects to find (it arms the
trigger itself).

What to aim for, with the calibration board held in the working volume:

* **p99 between 190 and 230** -- the white squares near the top of the range
* **clipped < 0.1 %** -- a blown-out white square has no corner left to find
* the black squares (p05) comfortably above 0 so the two levels are distinct

Brightness is measured on the raw Bayer values, which is correct: saturation
happens per photosite, before any demosaicing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import PySpin
except ImportError:
    sys.exit(
        "PySpin not found. Activate the capture environment:\n"
        '  cd "sensor-data-collection\\Python files"\n'
        "  .\\.venv\\Scripts\\Activate.ps1"
    )


def set_enum(nodemap, name, value):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
    if not (PySpin.IsAvailable(node) and PySpin.IsWritable(node)):
        return False
    entry = node.GetEntryByName(value)
    if not (PySpin.IsAvailable(entry) and PySpin.IsReadable(entry)):
        return False
    node.SetIntValue(entry.GetValue())
    return True


def set_float(nodemap, name, value):
    node = PySpin.CFloatPtr(nodemap.GetNode(name))
    if not (PySpin.IsAvailable(node) and PySpin.IsWritable(node)):
        return None
    clamped = min(max(float(value), node.GetMin()), node.GetMax())
    # ExposureTime declares no increment on a Blackfly S; GetInc() would raise.
    if node.HasInc():
        step = node.GetInc()
        if step > 0:
            clamped = node.GetMin() + round((clamped - node.GetMin()) / step) * step
    node.SetValue(clamped)
    return clamped


def set_bool(nodemap, name, value) -> bool:
    node = PySpin.CBooleanPtr(nodemap.GetNode(name))
    if not (PySpin.IsAvailable(node) and PySpin.IsWritable(node)):
        return False
    node.SetValue(bool(value))
    return True


def throttle(camera, fps: float) -> None:
    """Cap the free-run frame rate WHILE PROBING.

    Left uncapped, a Blackfly S streams at its maximum rate the moment
    acquisition starts, and 3.1 MB frames at ~50 fps overrun the USB3 budget --
    which Spinnaker reports as incomplete images, not as an error.

    This MUST be undone by `prepare_for_capture` before a triggered run: a
    camera capped at 5 fps enforces a 200 ms idle period, and with
    TriggerOverlap Off every 41 ms Arduino pulse lands inside it and is
    discarded. The capture then records zero images while still looking healthy.
    """
    nodemap = camera.GetNodeMap()
    set_bool(nodemap, "AcquisitionFrameRateEnable", True)
    set_float(nodemap, "AcquisitionFrameRate", fps)
    set_enum(camera.GetTLStreamNodeMap(), "StreamBufferHandlingMode", "NewestOnly")


def prepare_for_capture(camera, trigger_hz: float) -> dict:
    """Leave a camera in the state a hardware-triggered capture needs.

    Returns what the trigger period allows, so a too-long exposure is caught
    here instead of after a two-minute capture that saved nothing.
    """
    nodemap = camera.GetNodeMap()
    set_bool(nodemap, "AcquisitionFrameRateEnable", False)   # the trigger sets the rate
    # Without this a pulse arriving during exposure or readout is dropped
    # outright, which caps usable exposure well below the trigger period.
    overlap = set_enum(nodemap, "TriggerOverlap", "ReadOut")
    set_enum(nodemap, "TriggerMode", "Off")                  # capture arms it itself

    exposure = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime")).GetValue()
    rate_node = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionResultingFrameRate"))
    achievable = rate_node.GetValue() if PySpin.IsReadable(rate_node) else float("nan")
    return {
        "exposure_us": exposure,
        "gain_db": PySpin.CFloatPtr(nodemap.GetNode("Gain")).GetValue(),
        "overlap": overlap,
        "max_fps": achievable,
        "ok": not np.isfinite(achievable) or achievable >= trigger_hz,
    }


def serial_of(camera) -> str:
    node = PySpin.CStringPtr(camera.GetTLDeviceNodeMap().GetNode("DeviceSerialNumber"))
    return node.GetValue() if PySpin.IsReadable(node) else "?"


def bayer_to_gray(raw: np.ndarray) -> np.ndarray:
    """Half-resolution grey by averaging each 2x2 Bayer cell (no OpenCV needed)."""
    raw = raw.astype(np.uint16)
    height = raw.shape[0] - raw.shape[0] % 2
    width = raw.shape[1] - raw.shape[1] % 2
    block = raw[:height, :width].reshape(height // 2, 2, width // 2, 2)
    return block.mean(axis=(1, 3)).astype(np.uint8)


def grab(camera, timeout_ms: int):
    """Return (image, None) or (None, reason). Never swallows the reason."""
    try:
        camera.BeginAcquisition()
    except PySpin.SpinnakerException as error:
        return None, f"BeginAcquisition: {error}"
    try:
        data = None
        incomplete = 0
        reason = "?"
        # Frames already in flight can predate the new exposure, so keep the
        # last complete one rather than the first.
        for _ in range(4):
            try:
                image = camera.GetNextImage(timeout_ms)
            except PySpin.SpinnakerException as error:
                return None, f"GetNextImage: {error}"
            try:
                if image.IsIncomplete():
                    incomplete += 1
                    status = image.GetImageStatus()
                    reason = f"{status}: {PySpin.Image.GetImageStatusDescription(status)}"
                else:
                    data = np.array(image.GetNDArray(), copy=True)
            finally:
                image.Release()
        if data is None:
            return None, f"{incomplete} incomplete frames, last status {reason}"
        return data, None
    finally:
        try:
            camera.EndAcquisition()
        except PySpin.SpinnakerException:
            pass


def stats(raw: np.ndarray) -> dict:
    flat = raw.reshape(-1)
    return {
        "mean": float(flat.mean()),
        "p05": float(np.percentile(flat, 5)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "max": int(flat.max()),
        "clipped": 100.0 * float((flat >= 250).mean()),
    }


def verdict(row: dict) -> str:
    if row["clipped"] > 0.5:
        return "too bright"
    if row["p99"] < 120:
        return "too dark"
    if row["p99"] < 190:
        return "dim"
    if row["p99"] > 240 or row["clipped"] > 0.1:
        return "near clipping"
    return "GOOD"


def run_prepare(handles, args) -> int:
    print()
    print(f"trigger rate {args.trigger_hz:.1f} Hz  ->  period "
          f"{1e6 / args.trigger_hz:.0f} us")
    print()
    failures = []
    for serial, camera in handles:
        info = prepare_for_capture(camera, args.trigger_hz)
        state = "ok" if info["ok"] else "TOO SLOW"
        print(f"  {serial}: {info['exposure_us']:7.0f} us  {info['gain_db']:5.2f} dB  "
              f"max {info['max_fps']:5.1f} fps  overlap={'ReadOut' if info['overlap'] else 'n/a'}"
              f"   [{state}]")
        if not info["ok"]:
            failures.append((serial, info))

    print()
    if failures:
        print("These cameras cannot keep up with the trigger. Every pulse that arrives")
        print("while the sensor is busy is DISCARDED, so the capture would record zero")
        print("images without reporting an error.")
        worst = min(info["max_fps"] for _, info in failures)
        room = min(info["exposure_us"] for _, info in failures) * worst / args.trigger_hz
        print(f"Drop the exposure to roughly {room * 0.85:.0f} us and add gain to")
        print("compensate:")
        print(f"  python tune_exposure.py --auto --max-exposure-us {room * 0.85:.0f} --max-gain-db 20")
        return 1

    print("All cameras are ready. Run the capture WITHOUT --exposure-us / --gain-db.")
    return 0


def auto_tune(handles, args) -> int:
    """Bring every camera to the same brightness, each with its own settings.

    Sensor response is linear in both exposure time and linear gain, so one
    measurement predicts the correction almost exactly and this converges in two
    or three grabs. Exposure is spent first because it is noise-free; gain is
    only used for what exposure cannot reach inside the trigger period.

    Cameras do NOT need a common exposure. The Arduino pulse synchronises when
    each exposure *starts*, and with the board held still the length of the
    exposure is irrelevant. Matching brightness matters more than matching
    settings when the four views are lit this unevenly.
    """
    print()
    print(f"target p99 = {args.target_p99:.0f}   "
          f"limits: exposure <= {args.max_exposure_us:.0f} us, gain <= {args.max_gain_db:.1f} dB")
    print()
    results = {}
    for serial, camera in handles:
        nodemap = camera.GetNodeMap()
        exposure = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime")).GetValue()
        gain = PySpin.CFloatPtr(nodemap.GetNode("Gain")).GetValue()
        row = None
        for step in range(args.iterations):
            raw, reason = grab(camera, args.timeout_ms)
            if raw is None:
                print(f"  {serial}: FAILED: {reason}")
                break
            row = stats(raw)
            print(f"  {serial}  {exposure:7.0f} us  {gain:5.2f} dB  ->  "
                  f"p99 {row['p99']:5.1f}  clipped {row['clipped']:5.2f}%  {verdict(row)}")
            if verdict(row) == "GOOD":
                break
            # Linear response: this is the factor we are short (or over) by.
            wanted = args.target_p99 / max(row["p99"], 1.0)
            if step == args.iterations - 1:
                break
            headroom = args.max_exposure_us / exposure
            exposure_factor = min(wanted, headroom)
            exposure = set_float(nodemap, "ExposureTime", exposure * exposure_factor)
            remaining = wanted / exposure_factor
            if remaining > 1.02:
                gain = min(gain + 20.0 * np.log10(remaining), args.max_gain_db)
                gain = set_float(nodemap, "Gain", gain)
            elif remaining < 0.98 and gain > 0:
                gain = max(gain + 20.0 * np.log10(remaining), 0.0)
                gain = set_float(nodemap, "Gain", gain)
        if row is not None:
            results[serial] = (exposure, gain, row)
            if args.preview_dir:
                raw, _ = grab(camera, args.timeout_ms)
                if raw is not None:
                    from PIL import Image
                    Image.fromarray(bayer_to_gray(raw)).save(
                        args.preview_dir / f"{serial}_tuned.png")

    print()
    print("final settings, now stored in each camera:")
    for serial, (exposure, gain, row) in sorted(results.items()):
        print(f"  {serial}: {exposure:7.0f} us  {gain:5.2f} dB   "
              f"p99 {row['p99']:5.1f}  clipped {row['clipped']:5.2f}%   {verdict(row)}")

    short = [s for s, (e, g, r) in results.items() if verdict(r) in ("too dark", "dim")]
    if short:
        print()
        print(f"still short of the target: {', '.join(sorted(short))}")
        print("Exposure and gain are used up, so the remaining fix is physical:")
        print("  1. OPEN THE APERTURE one stop on those lenses -- 2x the light for free,")
        print("     no noise and no motion blur. Costs depth of field, so re-check that")
        print("     the whole board stays sharp afterwards.")
        print("  2. Add light. A lamp aimed at the working volume beats any setting here.")
        print("  3. Raise --max-gain-db (noisier corners, but ChArUco tolerates it).")

    print()
    for serial, camera in handles:
        prepare_for_capture(camera, args.trigger_hz)
    print("Frame-rate cap removed and TriggerOverlap set to ReadOut; verify with")
    print("  python tune_exposure.py --prepare")
    print()
    print("These settings live in the cameras. Run the capture WITHOUT --exposure-us")
    print("and --gain-db so it does not overwrite them:")
    print("  python capture_3blackfly_sensor_force.py --port COM3 --cameras 4 \\")
    print("      --no-force-export-monitor --frames 480 --output C:\\tactile_calib\\dryrun3")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sweep", help="comma-separated exposures in us, e.g. 6000,12000,20000")
    group.add_argument("--set", type=float, help="apply one exposure in us and stop")
    group.add_argument("--show", action="store_true", help="report current settings only")
    group.add_argument("--prepare", action="store_true",
                       help="leave the cameras ready for a hardware-triggered capture "
                            "and report whether the exposure fits the trigger period")
    group.add_argument("--auto", action="store_true",
                       help="tune each camera individually to --target-p99, spending "
                            "exposure first and gain only when exposure runs out")
    parser.add_argument("--target-p99", type=float, default=205.0)
    parser.add_argument("--max-exposure-us", type=float, default=30000.0,
                        help="keep below the ~41 ms trigger period, with margin")
    parser.add_argument("--max-gain-db", type=float, default=18.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--trigger-hz", type=float, default=24.2,
                        help="Arduino scan rate; exposure must leave room for it")
    parser.add_argument("--gain-db", type=float, default=None)
    parser.add_argument("--preview-dir", type=Path, default=None,
                        help="save a half-resolution preview per camera per exposure")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--free-run-fps", type=float, default=5.0,
                        help="frame rate cap while probing; low keeps USB3 from "
                             "overrunning and returning incomplete images")
    args = parser.parse_args()

    system = PySpin.System.GetInstance()
    cameras = system.GetCameras()
    if cameras.GetSize() == 0:
        print("no cameras found (is SpinView open?)")
        cameras.Clear()
        system.ReleaseInstance()
        return 1

    exposures = ([float(v) for v in args.sweep.split(",")] if args.sweep
                 else [args.set] if args.set else [])
    if args.auto or args.prepare:
        exposures = []

    handles = []
    for index in range(cameras.GetSize()):
        camera = cameras[index]
        camera.Init()
        handles.append((serial_of(camera), camera))
    handles.sort(key=lambda item: item[0])
    print(f"{len(handles)} camera(s): {', '.join(s for s, _ in handles)}")

    try:
        if args.prepare:
            return run_prepare(handles, args)

        if args.show or (not exposures and not args.auto):
            for serial, camera in handles:
                nodemap = camera.GetNodeMap()
                exposure = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime")).GetValue()
                gain = PySpin.CFloatPtr(nodemap.GetNode("Gain")).GetValue()
                print(f"  {serial}: {exposure:.0f} us, {gain:.2f} dB")
            return 0

        for serial, camera in handles:
            nodemap = camera.GetNodeMap()
            set_enum(nodemap, "TriggerMode", "Off")
            set_enum(nodemap, "AcquisitionMode", "Continuous")
            set_enum(nodemap, "ExposureAuto", "Off")
            set_enum(nodemap, "ExposureMode", "Timed")
            if args.gain_db is not None:
                set_enum(nodemap, "GainAuto", "Off")
                set_float(nodemap, "Gain", args.gain_db)
            throttle(camera, args.free_run_fps)

        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)

        if args.auto:
            return auto_tune(handles, args)

        print()
        print(f"{'exposure':>10} {'camera':>10} {'mean':>7} {'p50':>6} {'p99':>6} "
              f"{'max':>5} {'clipped':>9}  verdict")
        best = {}
        for exposure_us in exposures:
            for serial, camera in handles:
                actual = set_float(camera.GetNodeMap(), "ExposureTime", exposure_us)
                raw, reason = grab(camera, args.timeout_ms)
                if raw is None:
                    print(f"{actual:9.0f}u {serial:>10}   FAILED: {reason}")
                    continue
                row = stats(raw)
                mark = verdict(row)
                print(f"{actual:9.0f}u {serial:>10} {row['mean']:7.1f} {row['p50']:6.0f} "
                      f"{row['p99']:6.0f} {row['max']:5d} {row['clipped']:8.2f}%  {mark}")
                best.setdefault(serial, []).append((exposure_us, row, mark))
                if args.preview_dir:
                    from PIL import Image
                    name = f"{serial}_{int(actual)}us.png"
                    Image.fromarray(bayer_to_gray(raw)).save(args.preview_dir / name)
            print()

        if args.sweep:
            print("suggested exposure per camera (highest one that is not clipping):")
            for serial, rows in best.items():
                good = [r for r in rows if r[2] == "GOOD"] or \
                       [r for r in rows if r[2] in ("dim", "near clipping")]
                if good:
                    pick = max(good, key=lambda r: r[1]["p99"] if r[2] != "near clipping" else -1)
                    print(f"  {serial}: {pick[0]:.0f} us   (p99 {pick[1]['p99']:.0f}, "
                          f"clipped {pick[1]['clipped']:.2f}%)")
                else:
                    print(f"  {serial}: none of the tried values worked -- widen the sweep")
            print()
            print("Cameras do NOT have to share an exposure: the Arduino pulse")
            print("synchronises when each exposure starts, and with the board held")
            print("still its length does not matter. Prefer --auto, which matches")
            print("brightness by tuning each camera separately. To force one value:")
            print("  python tune_exposure.py --set <us> --gain-db <db>")
        else:
            print(f"cameras left at {exposures[0]:.0f} us"
                  + (f", {args.gain_db:.2f} dB" if args.gain_db is not None else "")
                  + ", TriggerMode Off -- ready for capture_3blackfly_sensor_force.py")
        return 0
    finally:
        # Spinnaker refuses to release the system while any Python name still
        # references a camera, so drop each one explicitly rather than relying
        # on the loop variable going out of scope.
        while handles:
            _, camera = handles.pop()
            try:
                camera.DeInit()
            except PySpin.SpinnakerException:
                pass
            del camera
        cameras.Clear()
        system.ReleaseInstance()


if __name__ == "__main__":
    raise SystemExit(main())
