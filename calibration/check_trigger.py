#!/usr/bin/env python3
"""Prove the hardware trigger actually delivers images, before capturing 27 GB.

RUN WITH THE **CAPTURE** ENVIRONMENT (needs PySpin and pyserial):

    cd "sensor-data-collection\\Python files"
    .\\.venv\\Scripts\\Activate.ps1
    python ..\\..\\calibration\\check_trigger.py --port COM3
    python ..\\..\\calibration\\check_trigger.py --board esp32 --snapshot C:\\tactile_data\\trigger_check

It arms the cameras exactly the way `capture_3blackfly_sensor_force.py` does,
runs the same Arduino READY/START handshake, and then counts the images each
camera receives over a few seconds -- optionally repeating at several exposure
times to find where the cameras stop keeping up with the trigger.

`--board esp32` is the ESP32 + AD7466 board flashed with
`Arduino files/Esp32Matrix32CameraTrigger.ino` (trigger on GPIO13). That board
scans, pulses the cameras and answers once per 'w' the host sends, so this tool
is the clock: it sends one 'w' every 1/--trigger-hz seconds and then checks
that every camera delivered exactly as many images as tactile frames came
back. It also prints the 4x4 pads so a dead tactile sensor shows up here too.
`--snapshot DIR` saves one image per camera to look at.

This exists because a camera that cannot service the trigger reports nothing at
all: `capture_3blackfly_sensor_force.py` runs happily to completion and writes a
session with zero images. Ten seconds here beats two minutes of empty capture.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

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

# ---- ESP32 + AD7466 board; keep in step with test/live_sensor_2d.py ----
ESP32_BAUD = 1_000_000
ESP32_VID_PID = (0x10C4, 0xEA60)  # Silicon Labs CP2104 on the Feather
ESP32_FRAME_SIDE = 32
ESP32_FRAME_PIXELS = ESP32_FRAME_SIDE * ESP32_FRAME_SIDE
ESP32_ROI = (28, 0, 4, 4)  # row0, col0, rows, cols of the wired 4x4 pads
ESP32_FLIP_ROWS = True     # pad row R1 is frame row 31
ESP32_REPLY_TIMEOUT_S = 0.5
# The pacer stops this long before the cameras stop listening, so the last
# triggered images are counted instead of being cut off by the window's end.
ESP32_DRAIN_TAIL_S = 1.0


def find_esp32_port() -> str:
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    found = [p for p in ports if (p.vid, p.pid) == ESP32_VID_PID]
    if not found:
        seen = "\n".join(f"  {p.device}: {p.description}" for p in ports) or "  (none)"
        sys.exit("no CP2104 (ESP32 board) serial port found; give --port. Ports seen:\n" + seen)
    if len(found) > 1:
        print(f"several CP2104 ports; using {found[0].device} (give --port to override)")
    return found[0].device


class Esp32Pacer:
    """Sends one 'w' per trigger period and reads the tactile frame that answers it.

    Each 'w' makes the board scan, pulse the camera trigger once and send one
    frame, so `sent` is the number of pulses the cameras should have seen.
    """

    def __init__(self, port: str, baud: int):
        self.port_name = port
        self.port = serial.Serial(port, baud, timeout=0.05, write_timeout=1.0)
        time.sleep(0.3)
        self.pixel_bytes = self._detect_pixel_bytes()
        self.bits = 8 if self.pixel_bytes == 1 else 12
        self.frame_bytes = ESP32_FRAME_PIXELS * self.pixel_bytes
        print(f"ESP32 board on {port} @ {baud} baud, {self.bits}-bit pixels")

    def _read_exact(self, count: int, timeout: float) -> bytes:
        got = bytearray()
        deadline = time.perf_counter() + timeout
        while len(got) < count and time.perf_counter() < deadline:
            got.extend(self.port.read(count - len(got)))
        return bytes(got)

    def _detect_pixel_bytes(self) -> int:
        """One frame's length tells the firmware build: 1025 bytes is 8-bit, 2049 is 12-bit."""
        got = b""
        for _ in range(4):  # opening the port resets the ESP32; it may still be booting
            self.port.reset_input_buffer()
            self.port.write(b"w")
            got = self._read_exact(ESP32_FRAME_PIXELS + 1, 1.0)
            if len(got) == ESP32_FRAME_PIXELS + 1:
                rest = self._read_exact(ESP32_FRAME_PIXELS, 0.4)
                if not rest:
                    return 1
                if len(rest) == ESP32_FRAME_PIXELS:
                    return 2
                got += rest
            time.sleep(0.5)
        raise RuntimeError(
            f"unexpected reply to 'w' from {self.port_name}: {len(got)} bytes (the firmware "
            "sends 1025 or 2049). Is Esp32Matrix32CameraTrigger.ino flashed, and is another "
            "program holding the port?")

    def pace(self, seconds: float, hz: float) -> dict:
        period = 1.0 / hz
        dtype = np.dtype("<u2") if self.pixel_bytes == 2 else np.dtype(np.uint8)
        row0, col0, rows, cols = ESP32_ROI
        stats = {"sent": 0, "ok": 0, "bad": 0, "timeouts": 0,
                 "send_times": [], "latencies": [], "grids": []}
        self.port.reset_input_buffer()
        start = time.perf_counter()
        next_send = start
        while True:
            now = time.perf_counter()
            if now - start >= seconds:
                break
            if now < next_send:
                # Windows sleeps in ~16 ms steps: sleep the bulk, spin the rest.
                if next_send - now > 0.02:
                    time.sleep(next_send - now - 0.02)
                continue
            # A scan slower than the period just runs flat out; it must not
            # build up a debt that is later paid back as a burst.
            next_send = max(next_send + period, now)
            sent_at = time.perf_counter()
            self.port.write(b"w")
            stats["sent"] += 1
            stats["send_times"].append(sent_at)
            reply = self._read_exact(self.frame_bytes + 1, ESP32_REPLY_TIMEOUT_S)
            if len(reply) < self.frame_bytes + 1:
                stats["timeouts"] += 1
                self.port.reset_input_buffer()
                continue
            if reply[-1] != 0x0A:
                stats["bad"] += 1
                time.sleep(0.2)
                self.port.reset_input_buffer()
                continue
            stats["ok"] += 1
            stats["latencies"].append(time.perf_counter() - sent_at)
            full = np.frombuffer(reply[:self.frame_bytes], dtype=dtype)
            full = full.reshape(ESP32_FRAME_SIDE, ESP32_FRAME_SIDE)
            grid = full[row0:row0 + rows, col0:col0 + cols]
            stats["grids"].append(np.array(grid[::-1] if ESP32_FLIP_ROWS else grid))
        return stats

    def close(self) -> None:
        try:
            self.port.close()
        except Exception:
            pass


def report_esp32(stats: dict, counts: dict, hz: float, adc_max: int) -> bool:
    """Print the tactile side and the per-camera verdict; True when everything matched."""
    sent, good = stats["sent"], stats["ok"]
    times = stats["send_times"]
    rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else float("nan")
    latency_ms = 1000.0 * float(np.median(stats["latencies"])) if stats["latencies"] else float("nan")
    print(f"tactile: {sent} requests ('w') = {sent} trigger pulses; {good} good frames, "
          f"{stats['timeouts']} timeouts, {stats['bad']} misframed")
    print(f"         paced at {rate:.2f} Hz (asked {hz:g}); scan + transfer {latency_ms:.0f} ms per frame")
    if np.isfinite(rate) and rate < 0.95 * hz:
        print(f"         the board cannot keep up with {hz:g} Hz -- lower --trigger-hz")
    tactile_ok = sent > 0 and good == sent
    if stats["grids"]:
        stack = np.stack(stats["grids"]).astype(np.float64)
        mean, swing = stack.mean(axis=0), stack.max(axis=0) - stack.min(axis=0)
        print(f"         4x4 pads, mean ADC (0-{adc_max}; pressing lowers it) | swing during the window:")
        for r in range(mean.shape[0]):
            print("           R%d  " % (r + 1)
                  + " ".join(f"{v:6.0f}" for v in mean[r]) + "   |  "
                  + " ".join(f"{v:5.0f}" for v in swing[r]))
        if float(mean.max() - mean.min()) < 1.0 and float(swing.max()) < 1.0:
            print("         every pad reads the same constant -- the sensor looks disconnected")
            tactile_ok = False
    else:
        print("         NO tactile frames came back")

    cameras_ok = True
    for serial_number, (complete, incomplete, _, _) in counts.items():
        got = complete + incomplete
        if got == sent and incomplete == 0:
            verdict = "OK, one image per tactile frame"
        elif got == 0:
            verdict = "NO IMAGES -- trigger not reaching this camera (wiring, level, OPTO_GND)"
        elif got < sent:
            verdict = f"MISSED {sent - got} pulses (weak trigger level, or exposure too long)"
        elif got > sent:
            verdict = f"{got - sent} EXTRA images -- spurious edges on the trigger line"
        else:
            verdict = f"{incomplete} incomplete images (USB bandwidth)"
        cameras_ok = cameras_ok and verdict.startswith("OK")
        print(f"camera {serial_number}: {got} images for {sent} pulses -> {verdict}")
    return tactile_ok and cameras_ok


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


def save_snapshot(image, path) -> None:
    """Save one image, demosaiced when the camera sends Bayer."""
    converted = None
    try:
        try:
            converted = PySpin.ImageProcessor().Convert(image, PySpin.PixelFormat_BGR8)
            converted.Save(str(path))
        except PySpin.SpinnakerException:
            image.Save(str(path))
    finally:
        if converted is not None:
            try:
                converted.Release()
            except PySpin.SpinnakerException:
                pass


def count_images(handles, seconds: float, timeout_ms: int,
                 on_all_started=None, snapshot_dir=None) -> dict:
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
    # Every camera listens for the same window: they all start acquiring, and
    # only then does the clock (and, for the ESP32 board, the pulse train) run.
    all_started = threading.Barrier(len(handles), action=on_all_started)

    def drain(serial_number, camera):
        camera.BeginAcquisition()
        try:
            all_started.wait(timeout=10.0)
        except threading.BrokenBarrierError:
            pass
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
                        if snapshot_dir is not None and counts[serial_number][0] == 1:
                            path = snapshot_dir / f"camera_{serial_number}.jpg"
                            try:
                                save_snapshot(image, path)
                            except PySpin.SpinnakerException as exc:
                                print(f"  {serial_number}: snapshot failed: {exc}")
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
    parser.add_argument("--board", choices=("m16b", "esp32"), default="m16b",
                        help="m16b: Arduino 16x16 board, which free-runs its own trigger "
                             "(default). esp32: ESP32 + AD7466 board, trigger on GPIO13, "
                             "paced from here.")
    parser.add_argument("--port", default=None,
                        help="default: COM3 for m16b; the CP2104 port for esp32")
    parser.add_argument("--baud", type=int, default=None,
                        help="default: 500000 for m16b, 1000000 for esp32")
    parser.add_argument("--trigger-hz", type=float, default=6.0,
                        help="--board esp32: 'w' requests (= trigger pulses) per second")
    parser.add_argument("--snapshot", type=Path, default=None, metavar="DIR",
                        help="save the first image of every camera here as camera_<serial>.jpg")
    parser.add_argument("--trigger-source", default="Line0")
    parser.add_argument("--seconds", type=float, default=4.0, help="per exposure value")
    parser.add_argument("--exposures", default=None,
                        help="comma-separated us to try; default is whatever the "
                             "cameras already hold")
    parser.add_argument("--timeout-ms", type=int, default=300)
    parser.add_argument("--handshake-timeout", type=float, default=15.0)
    args = parser.parse_args()
    esp32 = args.board == "esp32"
    if args.baud is None:
        args.baud = ESP32_BAUD if esp32 else 500_000
    if args.port is None:
        args.port = find_esp32_port() if esp32 else "COM3"
    if args.trigger_hz <= 0:
        parser.error("--trigger-hz must be positive")
    if esp32 and args.seconds < ESP32_DRAIN_TAIL_S + 1.0:
        parser.error(f"--seconds must be at least {ESP32_DRAIN_TAIL_S + 1.0:g} with --board esp32")
    if args.snapshot is not None:
        args.snapshot.mkdir(parents=True, exist_ok=True)

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
    pacer = None
    all_matched = True
    try:
        if esp32:
            # Before arming: finding the firmware's pixel size costs one frame,
            # and so one pulse, which no camera is listening for yet.
            pacer = Esp32Pacer(args.port, args.baud)
        for _, camera in handles:
            arm(camera, args.trigger_source)
        print("armed state:")
        for serial_number, camera in handles:
            report_state(serial_number, camera)

        if not esp32:
            connection = handshake(args.port, args.baud, args.handshake_timeout)
        heartbeat_stop = threading.Event()

        def heartbeat():
            while not heartbeat_stop.wait(HEARTBEAT_INTERVAL_S):
                try:
                    connection.write(COMMAND_PING)
                    connection.flush()
                except serial.SerialException:
                    return

        if not esp32:
            threading.Thread(target=heartbeat, daemon=True).start()
        exposures = ([float(v) for v in args.exposures.split(",")]
                     if args.exposures else [None])

        print()
        print(f"{'exposure':>10} {'camera':>10} {'complete':>10} {'incomplete':>11} "
              f"{'fps':>7} {'p99':>6}  brightness")
        for exposure_us in exposures:
            if exposure_us is not None:
                for serial_number, camera in handles:
                    # ExposureTime is read-only while ExposureAuto runs, and the
                    # write then fails silently: the sweep would test nothing.
                    set_enum(camera.GetNodeMap(), "ExposureAuto", "Off")
                    if set_float(camera.GetNodeMap(), "ExposureTime", exposure_us) is None:
                        print(f"  {serial_number}: could not set the exposure")
                    if enum_value(camera.GetNodeMap(), "GainAuto") != "Off":
                        print(f"  {serial_number}: GainAuto is on and will cancel the "
                              "exposure change -- run tune_exposure.py --auto first")
            if esp32:
                go = threading.Event()
                paced = {}

                def run_pacer():
                    if not go.wait(timeout=30.0):
                        return
                    try:
                        paced.update(pacer.pace(args.seconds - ESP32_DRAIN_TAIL_S,
                                                args.trigger_hz))
                    except serial.SerialException as exc:
                        print(f"ESP32 serial error: {exc}")

                pacer_thread = threading.Thread(target=run_pacer, daemon=True)
                pacer_thread.start()
                counts = count_images(handles, args.seconds, args.timeout_ms,
                                      on_all_started=go.set, snapshot_dir=args.snapshot)
                go.set()  # never leave the pacer waiting if a camera failed to start
                pacer_thread.join()
            else:
                counts = count_images(handles, args.seconds, args.timeout_ms,
                                      snapshot_dir=args.snapshot)
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
            if esp32:
                if paced:
                    matched = report_esp32(paced, counts, args.trigger_hz,
                                           2 ** pacer.bits - 1)
                    all_matched = all_matched and matched
                else:
                    print("the ESP32 pacer did not run")
                    all_matched = False
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
        if args.snapshot is not None:
            print(f"snapshots: {args.snapshot.resolve()}")
        if esp32:
            print()
            print("RESULT: cameras and tactile sensor are in step" if all_matched
                  else "RESULT: MISMATCH -- see the lines above")
            return 0 if all_matched else 2
        print()
        print("A camera reading 0 is not receiving or not")
        print("servicing the trigger; a low but non-zero rate means it is missing")
        print("pulses, usually because the exposure does not fit the trigger period.")
        return 0
    finally:
        if pacer is not None:
            pacer.close()
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
