#!/usr/bin/env python3
"""Prove the hardware trigger actually delivers images, before capturing 27 GB.

RUN WITH THE **CAPTURE** ENVIRONMENT (needs PySpin and pyserial):

    cd "sensor-data-collection\\Python files"
    .\\.venv\\Scripts\\Activate.ps1
    python ..\\..\\calibration\\check_trigger.py --port COM3

It arms the cameras exactly the way `capture_3blackfly_sensor_force.py` does,
runs the same Arduino READY/START handshake, and then counts the images each
camera receives over a few seconds -- optionally repeating at several exposure
times to find where the cameras stop keeping up with the trigger.

This exists because a camera that cannot service the trigger reports nothing at
all: `capture_3blackfly_sensor_force.py` runs happily to completion and writes a
session with zero images. Ten seconds here beats two minutes of empty capture.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

try:
    import PySpin
except ImportError:
    sys.exit("PySpin not found -- activate the capture environment (.venv).")
try:
    import serial
except ImportError:
    sys.exit("pyserial not found -- activate the capture environment (.venv).")

ARDUINO_READY = b"M16_READY"
ARDUINO_ACK = b"M16_ACK"
COMMAND_START = b"M16_START\n"
COMMAND_PING = b"M16_PING\n"
# The Arduino stops pulsing when the host goes quiet;
# capture_3blackfly_sensor_force keeps it alive with a ping twice a second.
# Without one, this tool saw a burst of 47 frames and then silence -- and so
# reported a perfectly good trigger as a broken one, at any window length.
HEARTBEAT_INTERVAL_S = 0.5
COMMAND_STOP = b"M16_STOP\n"


def enum_value(nodemap, name):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
    if not (PySpin.IsAvailable(node) and PySpin.IsReadable(node)):
        return "n/a"
    return node.GetCurrentEntry().GetSymbolic()


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
    if node.HasInc():
        step = node.GetInc()
        if step > 0:
            clamped = node.GetMin() + round((clamped - node.GetMin()) / step) * step
    node.SetValue(clamped)
    return clamped


def arm(camera, trigger_source: str) -> None:
    """The exact sequence capture_3blackfly_sensor_force.py uses."""
    nodemap = camera.GetNodeMap()
    set_enum(nodemap, "TriggerMode", "Off")
    set_enum(nodemap, "TriggerSelector", "FrameStart")
    set_enum(nodemap, "TriggerSource", trigger_source)
    set_enum(nodemap, "TriggerActivation", "RisingEdge")
    set_enum(nodemap, "ExposureMode", "Timed")
    set_enum(nodemap, "AcquisitionMode", "Continuous")
    set_enum(nodemap, "TriggerMode", "On")


def report_state(serial_number, camera) -> None:
    nodemap = camera.GetNodeMap()
    exposure = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime")).GetValue()
    rate_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
    enabled = rate_enable.GetValue() if PySpin.IsReadable(rate_enable) else "n/a"
    armed_node = PySpin.CBooleanPtr(nodemap.GetNode("TriggerArmed"))
    armed = armed_node.GetValue() if PySpin.IsReadable(armed_node) else "n/a"
    print(f"  {serial_number}: exposure {exposure:.0f} us | "
          f"TriggerMode {enum_value(nodemap, 'TriggerMode')} | "
          f"source {enum_value(nodemap, 'TriggerSource')} | "
          f"activation {enum_value(nodemap, 'TriggerActivation')} | "
          f"overlap {enum_value(nodemap, 'TriggerOverlap')} | "
          f"rate-limit {enabled} | armed {armed}")


def handshake(port: str, baud: int, timeout: float):
    connection = serial.Serial(port, baud, timeout=0.2, write_timeout=1.0)
    time.sleep(0.2)
    connection.reset_input_buffer()
    deadline = time.monotonic() + timeout
    print(f"waiting for Arduino READY on {port}...")
    while time.monotonic() < deadline:
        line = connection.readline().strip()
        if line != ARDUINO_READY:
            continue
        connection.write(COMMAND_START)
        connection.flush()
        while time.monotonic() < deadline:
            response = connection.readline().strip()
            if response == ARDUINO_ACK:
                print("Arduino ACK -- trigger pulses are running")
                return connection
            if response == ARDUINO_READY:
                connection.write(COMMAND_START)
                connection.flush()
    connection.close()
    raise TimeoutError("Arduino handshake timed out")


def count_images(handles, seconds: float, timeout_ms: int) -> dict:
    """Count what each camera receives, one thread per camera.

    Polling the cameras round-robin from one thread measures the poller, not
    the rig: every blocking `GetNextImage` serialises behind the last, and the
    99th-percentile taken for the brightness column costs ~15 ms on a 3 MP
    frame. On this rig that reported 11.5 fps per camera while the cameras were
    really receiving 30 -- a tool meant to catch a broken trigger inventing one
    instead. So: a thread each, timestamps only in the hot path, and brightness
    sampled from a handful of frames.
    """
    counts = {serial_number: [0, 0, [], []] for serial_number, _ in handles}

    def drain(serial_number, camera):
        camera.BeginAcquisition()
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                try:
                    image = camera.GetNextImage(timeout_ms)
                except PySpin.SpinnakerException:
                    continue
                try:
                    if image.IsIncomplete():
                        counts[serial_number][1] += 1
                    else:
                        counts[serial_number][0] += 1
                        counts[serial_number][3].append(time.monotonic())
                        if len(counts[serial_number][2]) < 4:
                            counts[serial_number][2].append(
                                float(np.percentile(image.GetNDArray(), 99)))
                finally:
                    image.Release()
        finally:
            try:
                camera.EndAcquisition()
            except PySpin.SpinnakerException:
                pass

    threads = [threading.Thread(target=drain, args=handle) for handle in handles]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=500_000)
    parser.add_argument("--trigger-source", default="Line0")
    parser.add_argument("--seconds", type=float, default=4.0, help="per exposure value")
    parser.add_argument("--exposures", default=None,
                        help="comma-separated us to try; default is whatever the "
                             "cameras already hold")
    parser.add_argument("--timeout-ms", type=int, default=300)
    parser.add_argument("--handshake-timeout", type=float, default=15.0)
    args = parser.parse_args()

    system = PySpin.System.GetInstance()
    cameras = system.GetCameras()
    if cameras.GetSize() == 0:
        print("no cameras found")
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

    connection = None
    try:
        for _, camera in handles:
            arm(camera, args.trigger_source)
        print("armed state:")
        for serial_number, camera in handles:
            report_state(serial_number, camera)

        connection = handshake(args.port, args.baud, args.handshake_timeout)
        heartbeat_stop = threading.Event()

        def heartbeat():
            while not heartbeat_stop.wait(HEARTBEAT_INTERVAL_S):
                try:
                    connection.write(COMMAND_PING)
                    connection.flush()
                except serial.SerialException:
                    return

        threading.Thread(target=heartbeat, daemon=True).start()
        exposures = ([float(v) for v in args.exposures.split(",")]
                     if args.exposures else [None])

        print()
        print(f"{'exposure':>10} {'camera':>10} {'complete':>10} {'incomplete':>11} "
              f"{'fps':>7} {'p99':>6}  brightness")
        for exposure_us in exposures:
            if exposure_us is not None:
                for _, camera in handles:
                    set_float(camera.GetNodeMap(), "ExposureTime", exposure_us)
            counts = count_images(handles, args.seconds, args.timeout_ms)
            label = f"{exposure_us:.0f}us" if exposure_us else "as-is"
            for serial_number, _ in handles:
                complete, incomplete, samples, _ = counts[serial_number]
                p99 = float(np.median(samples)) if samples else float("nan")
                if not np.isfinite(p99):
                    note = ""
                elif p99 < 120:
                    note = "TOO DARK -- retune the exposure"
                elif p99 > 240:
                    note = "clipping -- lower the exposure"
                else:
                    note = "ok"
                print(f"{label:>10} {serial_number:>10} {complete:>10} {incomplete:>11} "
                      f"{complete / args.seconds:>7.1f} {p99:>6.0f}  {note}")
            print()

        # Span over interval count, not the median gap: images are delivered in
        # small bursts, so consecutive gaps alternate between ~0 and ~2 periods
        # and their median says nothing about the pulse rate.
        rates = [(len(stamps) - 1) / (stamps[-1] - stamps[0])
                 for _, _, _, stamps in counts.values() if len(stamps) > 20]
        if rates:
            print()
            print(f"measured trigger rate {float(np.median(rates)):.1f} Hz "
                  f"({1000.0 / float(np.median(rates)):.1f} ms period)")
            print("Pass this to --trigger-hz everywhere: the exposure headroom check")
            print("in tune_exposure.py is only as right as the period it assumes.")
        print()
        print("A camera reading 0 is not receiving or not")
        print("servicing the trigger; a low but non-zero rate means it is missing")
        print("pulses, usually because the exposure does not fit the trigger period.")
        return 0
    finally:
        if connection is not None:
            try:
                heartbeat_stop.set()
                connection.write(COMMAND_STOP)
                connection.flush()
            except Exception:
                pass
            connection.close()
        while handles:
            _, camera = handles.pop()
            try:
                set_enum(camera.GetNodeMap(), "TriggerMode", "Off")
                camera.DeInit()
            except PySpin.SpinnakerException:
                pass
            del camera
        cameras.Clear()
        system.ReleaseInstance()


if __name__ == "__main__":
    raise SystemExit(main())
