#!/usr/bin/env python3
"""Put every Blackfly S back on automatic exposure and gain.

RUN THIS WITH THE **CAPTURE** ENVIRONMENT, not the calibration one -- it needs
PySpin:

    cd "sensor-data-collection\\Python files"
    .\\.venv\\Scripts\\Activate.ps1
    python ..\\..\\calibration\\reset_camera_auto_exposure.py

Why this exists: passing `--exposure-us` / `--gain-db` to
`capture_3blackfly_sensor_force.py` sets `ExposureAuto` and `GainAuto` to Off
and pins manual values. The capture script restores `TriggerMode` on exit but
not those two, and the camera keeps them in volatile memory -- across programs,
across SpinView, until it loses power. So a single calibration run silently
leaves the whole rig on fixed exposure, which looks like "all my cameras went
dark at once".

`--show` reports the current state without changing anything.
"""

from __future__ import annotations

import argparse
import sys

try:
    import PySpin
except ImportError:
    sys.exit(
        "PySpin not found. Activate the capture environment:\n"
        '  cd "sensor-data-collection\\Python files"\n'
        "  .\\.venv\\Scripts\\Activate.ps1"
    )


def read_enum(nodemap, name: str) -> str:
    node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
    if not (PySpin.IsAvailable(node) and PySpin.IsReadable(node)):
        return "n/a"
    entry = node.GetCurrentEntry()
    return entry.GetSymbolic() if PySpin.IsReadable(entry) else "n/a"


def read_float(nodemap, name: str):
    node = PySpin.CFloatPtr(nodemap.GetNode(name))
    if not (PySpin.IsAvailable(node) and PySpin.IsReadable(node)):
        return None
    return node.GetValue()


def write_enum(nodemap, name: str, value: str) -> bool:
    node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
    if not (PySpin.IsAvailable(node) and PySpin.IsWritable(node)):
        return False
    entry = node.GetEntryByName(value)
    if not (PySpin.IsAvailable(entry) and PySpin.IsReadable(entry)):
        return False
    node.SetIntValue(entry.GetValue())
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="only report the current state")
    parser.add_argument("--exposure-auto", default="Continuous",
                        choices=("Continuous", "Once", "Off"))
    parser.add_argument("--gain-auto", default="Continuous",
                        choices=("Continuous", "Once", "Off"))
    args = parser.parse_args()

    system = PySpin.System.GetInstance()
    cameras = system.GetCameras()
    count = cameras.GetSize()
    if count == 0:
        print("no cameras found (is SpinView still open?)")
        cameras.Clear()
        system.ReleaseInstance()
        return 1

    print(f"{count} camera(s)")
    for index in range(count):
        camera = cameras[index]
        camera.Init()
        nodemap = camera.GetNodeMap()
        serial = read_enum(camera.GetTLDeviceNodeMap(), "DeviceSerialNumber")
        if serial == "n/a":
            node = PySpin.CStringPtr(
                camera.GetTLDeviceNodeMap().GetNode("DeviceSerialNumber"))
            serial = node.GetValue() if PySpin.IsReadable(node) else "?"

        before = (read_enum(nodemap, "ExposureAuto"), read_float(nodemap, "ExposureTime"),
                  read_enum(nodemap, "GainAuto"), read_float(nodemap, "Gain"),
                  read_enum(nodemap, "TriggerMode"))
        print(f"  {serial}: ExposureAuto={before[0]} ({before[1]:.0f} us), "
              f"GainAuto={before[2]} ({before[3]:.2f} dB), TriggerMode={before[4]}")

        if not args.show:
            write_enum(nodemap, "TriggerMode", "Off")   # also left set if a run crashed
            ok_exposure = write_enum(nodemap, "ExposureAuto", args.exposure_auto)
            ok_gain = write_enum(nodemap, "GainAuto", args.gain_auto)
            after = (read_enum(nodemap, "ExposureAuto"), read_float(nodemap, "ExposureTime"),
                     read_enum(nodemap, "GainAuto"), read_float(nodemap, "Gain"))
            print(f"    -> ExposureAuto={after[0]} ({after[1]:.0f} us), "
                  f"GainAuto={after[2]} ({after[3]:.2f} dB)"
                  + ("" if ok_exposure and ok_gain else "   [some nodes were not writable]"))

        camera.DeInit()
        del camera

    cameras.Clear()
    system.ReleaseInstance()
    if args.show:
        print("\nnothing changed (--show)")
    else:
        print("\nAuto exposure/gain restored. Note that these settings live in the "
              "camera's volatile memory: unplugging the cameras also resets them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
