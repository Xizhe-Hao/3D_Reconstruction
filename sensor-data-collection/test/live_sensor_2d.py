#!/usr/bin/env python3
"""Live 2D pressure image of a tactile sensor array, with bench recording.

A standalone bench tool for checking what the sensor actually sees, before
committing to a full camera + force capture. It shows the array as a smooth
2D image using the same signal definition as the capture pipeline
(V_sensor = Vdrive - V_adc, then baseline subtraction), so what looks good
here looks the same in replay_capture_multimodal.py.

Two boards are supported (--board):
    m16b    the 16x16 Velostat array on the Arduino Nano, M16B firmware
            (500 kbaud, READY/START handshake, framed + checksummed), default
    esp32   the 4x4 Velostat test board on the ESP32 + AD7466: 1 Mbaud, one
            'w' credit per frame, raw 32x32 bytes + '\\n', VREF 3.3 V; only the
            top-left 4x4 of the frame is wired (--roi), ~8 fps free-running

Usage:
    python live_sensor_2d.py COM7                 # 16x16 live, baseline-subtracted
    python live_sensor_2d.py COM7 --mode voltage  # absolute sensor voltage
    python live_sensor_2d.py COM7 --mode raw      # raw 8-bit ADC
    python live_sensor_2d.py --board esp32        # 4x4 board, port found by itself
    python live_sensor_2d.py --board esp32 COM4 --mode raw --labels
    python live_sensor_2d.py --simulate           # no hardware, synthetic press
    python live_sensor_2d.py --list-ports         # find the COM port
    python live_sensor_2d.py --selftest           # headless render check
    python live_sensor_2d.py --follow E:/tactile_sensor_captures --mode raw
                                                  # watch a running capture

    python live_sensor_2d.py COM3 --mode raw --record C:/tactile_data/force_cal
    python live_sensor_2d.py --board esp32 --mode raw --record C:/tactile_data/force_cal_4x4
                                                  # show and record, no cameras

--record writes every frame to <root>/capture_<stamp>/sensor_raw.csv in the
same format as the capture script (host_received_ns, frame_index,
device_millis, r1c1_adc8 ...), plus a session.json with the board facts
(grid, ADC reference, pad size, pitch), so bench_quicklook.py,
align_force_presses.py, bench_report.py and force_model_eval.py work on it
unchanged for either board. Put the IntelliMESUR export in <session>/force/
afterwards. --duration S stops by itself after S seconds.

--follow reads the sensor_raw.csv that capture_3blackfly_sensor_force.py is
writing, so the capture keeps sole ownership of the serial port. Give it a
session folder, a CSV, or the capture root; with the root it shows the newest
capture_* session and switches when the next capture starts.

Keys while the window is focused:
    b        re-capture the no-load baseline
    i        toggle smooth / raw-cell rendering
    v        toggle per-cell value labels
    + / -    display gain up / down
    ] / [    display range up / down
    r        reset the held maximum (--mode raw)
    s        save a PNG snapshot
    q / Esc  quit

Dependencies: pyserial, numpy, matplotlib. No cameras or force gauge needed.
This tool only reads the sensor; --record creates its own session folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---- M16B protocol constants (must match MatrixArrayDual4067Binary.ino) ----
BAUD_RATE = 500_000
MAGIC = b"M16B"
ROWS = 16
COLS = 16
PAYLOAD_BYTES = ROWS * COLS
# version(1) + rows(1) + cols(1) + type(1) + frame_index(4) + millis(4)
HEADER_AFTER_MAGIC = 1 + 1 + 1 + 1 + 4 + 4
FRAME_BYTES = len(MAGIC) + HEADER_AFTER_MAGIC + PAYLOAD_BYTES + 2
PING_INTERVAL_S = 0.5  # the firmware heartbeat times out after 2 s
ADC_MAX = 255.0
SATURATION_ADC = 250  # raw readings at or above this count as saturated

# ---- ESP32 + AD7466 board (the firmware behind test/4X4/matrix_heatmap_ad7466*.py) ----
ESP32_BAUD = 1_000_000
ESP32_FRAME_W = 32
ESP32_FRAME_H = 32
ESP32_VID_PID = (0x10C4, 0xEA60)  # Silicon Labs CP2104 on the ESP32 board
ESP32_DEFAULT_ROI = (0, 0, 4, 4)  # row0, col0, rows, cols of the wired pads inside the 32x32 frame

# Board facts written into session.json by --record and read back by the analysis scripts.
BOARDS = {
    "m16b": {
        "board": "m16b", "protocol": "M16B v1",
        "label": "16x16 Velostat array, Arduino Nano (M16B)",
        "rows": ROWS, "cols": COLS, "baud": BAUD_RATE, "adc_bits": 8,
        "adc_reference_v": 5.0, "sensor_drive_v": 5.0, "pressure_lowers_adc": False,
        "pad_mm": 5.0, "pitch_mm": 7.5, "expected_fps": 6.0,
    },
    "esp32": {
        "board": "esp32-4x4", "protocol": "ESP32 AD7466 raw 32x32, 'w' credits",
        "label": "4x4 Velostat test board, ESP32 + AD7466",
        "rows": ESP32_DEFAULT_ROI[2], "cols": ESP32_DEFAULT_ROI[3], "baud": ESP32_BAUD, "adc_bits": 8,
        "adc_reference_v": 3.3, "sensor_drive_v": 3.3, "pressure_lowers_adc": True,
        "pad_mm": 8.0, "pitch_mm": None, "expected_fps": None,  # free-running, ~8 fps
    },
}

MODE_COLORMAPS = {
    "change": "RdBu_r",
    "voltage": "turbo",
    "raw": "viridis",
}

CELL_COLUMN = re.compile(r"^r(\d+)c(\d+)_adc8$")


def header_grid(head):
    """(rows, cols) from a sensor_raw.csv header; the cells are r{R}c{C}_adc8, row-major."""
    cells = [CELL_COLUMN.match(name.strip()) for name in head]
    cells = [(int(m.group(1)), int(m.group(2))) for m in cells if m]
    if not cells:
        raise ValueError("no r{R}c{C}_adc8 columns in the header")
    return max(r for r, _ in cells), max(c for _, c in cells)


def parse_frames(buffer):
    """Yield (frame_index, millis, 16x16 uint8) and consume `buffer` in place."""
    while True:
        start = buffer.find(MAGIC)
        if start < 0:
            # Keep a partial magic that may be split across two reads.
            if len(buffer) > len(MAGIC) - 1:
                del buffer[: -(len(MAGIC) - 1)]
            return
        if start:
            del buffer[:start]
        if len(buffer) < FRAME_BYTES:
            return

        frame = bytes(buffer[:FRAME_BYTES])
        body = frame[len(MAGIC):-2]
        version, rows, columns, payload_type = body[0], body[1], body[2], body[3]
        frame_index = int.from_bytes(body[4:8], "little")
        millis = int.from_bytes(body[8:12], "little")
        payload = body[12:]
        checksum_expected = int.from_bytes(frame[-2:], "little")

        if (
            version == 1
            and rows == ROWS
            and columns == COLS
            and payload_type == 1
            and (sum(body) & 0xFFFF) == checksum_expected
        ):
            del buffer[:FRAME_BYTES]
            grid = np.frombuffer(payload, dtype=np.uint8).reshape(ROWS, COLS)
            yield frame_index, millis, grid
        else:
            # Corrupt frame: drop only this magic so a good frame that starts
            # inside the corrupt bytes is still found.
            del buffer[: len(MAGIC)]


class SerialSource:
    """Reads M16B frames from the Arduino and keeps the heartbeat alive."""

    shape = (ROWS, COLS)
    board = "m16b"

    def __init__(self, port, baud, handshake_timeout):
        import serial

        self.port_name = port
        self.baud = baud
        self.port = serial.Serial(port, baud, timeout=0.05)
        try:
            # A large receive buffer keeps frames from being lost while a slow
            # redraw (the 3D view especially) blocks this thread.
            self.port.set_buffer_size(rx_size=262_144, tx_size=4096)
        except (AttributeError, OSError, serial.SerialException):
            pass  # Not supported off Windows; the default buffer still works.
        # The Nano auto-resets when the port opens; wait for the reboot.
        time.sleep(2.0)
        self.port.reset_input_buffer()

        print(f"Waiting for M16_READY on {port} ...", flush=True)
        deadline = time.time() + handshake_timeout
        ready = False
        while time.time() < deadline:
            if b"M16_READY" in self.port.readline():
                ready = True
                break
        if not ready:
            self.port.close()
            raise RuntimeError(
                f"No M16_READY within {handshake_timeout:g} s. Check the port, "
                "the 500000 baud rate, and that the firmware is flashed."
            )

        self.port.write(b"M16_START\n")
        self.port.flush()
        print("Sent M16_START; streaming.", flush=True)

        self.buffer = bytearray()
        self.last_ping = time.time()
        self.last_index = None
        self.on_frame = None  # called with every frame, e.g. by --record

        # Reading, the heartbeat and on_frame run on their own thread, so a
        # window drag, a click or a locked screen that blocks the GUI cannot
        # starve the firmware's 2 s heartbeat or drop recorded frames. It
        # starts on the first read_latest(), after on_frame has been set.
        self.lock = threading.Lock()
        self.newest = None
        self.stop_event = threading.Event()
        self.error = None
        self.thread = None

    def meta(self):
        """Board facts for session.json."""
        return {**{k: v for k, v in BOARDS["m16b"].items() if k != "label"},
                "port": self.port_name, "baud": self.baud}

    def _reader(self):
        try:
            while not self.stop_event.is_set():
                now = time.time()
                if now - self.last_ping >= PING_INTERVAL_S:
                    self.port.write(b"M16_PING\n")
                    self.last_ping = now

                chunk = self.port.read(4096)
                if chunk:
                    self.buffer.extend(chunk)

                for frame_index, millis, grid in parse_frames(self.buffer):
                    # The firmware sends each frame twice (once before and
                    # once after the camera trigger); keep the first copy only.
                    if frame_index == self.last_index:
                        continue
                    self.last_index = frame_index
                    if self.on_frame is not None:
                        self.on_frame(time.time_ns(), frame_index, millis, grid)
                    with self.lock:
                        self.newest = (frame_index, millis, grid)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc
                print(f"Sensor reader stopped: {exc}", file=sys.stderr, flush=True)

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def read_latest(self):
        """The newest frame since the last call, or None."""
        self.start()
        with self.lock:
            newest, self.newest = self.newest, None
        return newest

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        try:
            self.port.write(b"M16_STOP\n")
            self.port.flush()
        except Exception:
            pass
        try:
            self.port.close()
        except Exception:
            pass
        print("Stopped; M16_STOP sent.")


def find_esp32_port():
    """The COM port of the ESP32 board (CP2104), else the only non-Bluetooth USB serial port."""
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    found = [p for p in ports if (p.vid, p.pid) == ESP32_VID_PID]
    if not found:
        found = [p for p in ports if p.vid is not None and "Bluetooth" not in (p.description or "")]
    if not found:
        seen = "\n".join(f"  {p.device}: {p.description}" for p in ports) or "  (none)"
        raise RuntimeError(
            "no USB serial port found. Ports seen:\n" + seen +
            "\nIf Device Manager shows 'CP2104 USB to UART Bridge Controller' with a warning "
            "icon, install the Silicon Labs CP210x driver and re-plug the ESP32."
        )
    if len(found) > 1:
        print(f"Several USB serial ports; using {found[0].device}. Give the port to override.", flush=True)
    return found[0].device


class Esp32Source:
    """Reads raw 32x32 frames from the ESP32 + AD7466 board and crops the wired pads.

    The host sends one 'w' per frame it wants; the board answers each with
    32 x 32 bytes (8-bit, VREF 3.3 V, row-major) followed by '\\n'. There is no
    header, index or checksum, so the frame index and the millis column are
    made here from the host clock, and the '\\n' after every 1024th byte is
    the only framing check. PIPELINE_TARGET credits are kept in flight so the
    board scans continuously (about 8 fps, 124 ms per 32x32 scan).
    """

    PIPELINE_TARGET = 8
    PIPELINE_LOW = 3
    READ_TIMEOUT_S = 0.5  # no byte for this long: the credits were lost, start over

    def __init__(self, port=None, baud=ESP32_BAUD, roi=ESP32_DEFAULT_ROI):
        import serial

        self.port_name = port or find_esp32_port()
        self.baud = baud
        self.row0, self.col0, rows, cols = roi
        if not (0 <= self.row0 and self.row0 + rows <= ESP32_FRAME_H
                and 0 <= self.col0 and self.col0 + cols <= ESP32_FRAME_W and rows > 0 and cols > 0):
            raise ValueError(f"--roi {roi} does not fit the {ESP32_FRAME_H}x{ESP32_FRAME_W} frame")
        self.shape = (rows, cols)
        self.board = "esp32"
        self.port = serial.Serial(self.port_name, baud, timeout=0.01)
        try:
            self.port.set_buffer_size(rx_size=262_144, tx_size=4096)
        except (AttributeError, OSError, serial.SerialException):
            pass
        time.sleep(0.3)
        self.port.reset_input_buffer()
        print(f"ESP32 board on {self.port_name} @ {baud} baud; pads = frame rows "
              f"{self.row0}-{self.row0 + rows - 1}, cols {self.col0}-{self.col0 + cols - 1}", flush=True)

        self.on_frame = None  # called with every frame, e.g. by --record
        self.lock = threading.Lock()
        self.newest = None
        self.stop_event = threading.Event()
        self.error = None
        self.thread = None
        self.frame_index = 0
        self.resyncs = 0
        self.t0 = time.perf_counter()

    def meta(self):
        """Board facts for session.json."""
        facts = {k: v for k, v in BOARDS["esp32"].items() if k != "label"}
        facts.update(port=self.port_name, baud=self.baud, rows=self.shape[0], cols=self.shape[1],
                     roi={"row0": self.row0, "col0": self.col0, "rows": self.shape[0], "cols": self.shape[1],
                          "frame_rows": ESP32_FRAME_H, "frame_cols": ESP32_FRAME_W})
        return facts

    def _refill(self, credits):
        need = self.PIPELINE_TARGET - credits
        if need > 0:
            self.port.write(b"w" * need)
            return credits + need
        return credits

    def _reader(self):
        frame_bytes = ESP32_FRAME_W * ESP32_FRAME_H
        rows, cols = self.shape
        buf = bytearray()
        credits = self._refill(0)
        last_byte = time.time()
        try:
            while not self.stop_event.is_set():
                chunk = self.port.read(4096)
                if chunk:
                    buf.extend(chunk)
                    last_byte = time.time()
                elif time.time() - last_byte > self.READ_TIMEOUT_S:
                    # The board only sends against credit, so silence means the
                    # credits were lost (reset, glitch): drop everything, re-request.
                    self.port.reset_input_buffer()
                    buf.clear()
                    credits = self._refill(0)
                    last_byte = time.time()
                    self.resyncs += 1
                    continue
                while len(buf) >= frame_bytes + 1:
                    if buf[frame_bytes] != 0x0A:
                        # Not on a frame boundary: skip to the next newline and try again
                        # (a data byte can be 10 too, so the next check settles it).
                        cut = buf.find(b"\n")
                        del buf[: cut + 1 if cut >= 0 else len(buf)]
                        credits -= 1  # that frame's credit is spent
                        self.resyncs += 1
                        if credits <= self.PIPELINE_LOW:
                            credits = self._refill(credits)
                        continue
                    full = np.frombuffer(bytes(buf[:frame_bytes]), dtype=np.uint8)
                    full = full.reshape(ESP32_FRAME_H, ESP32_FRAME_W)
                    grid = np.ascontiguousarray(full[self.row0:self.row0 + rows, self.col0:self.col0 + cols])
                    del buf[: frame_bytes + 1]
                    credits -= 1
                    if credits <= self.PIPELINE_LOW:
                        credits = self._refill(credits)
                    self.frame_index += 1
                    millis = int((time.perf_counter() - self.t0) * 1000.0)
                    if self.on_frame is not None:
                        self.on_frame(time.time_ns(), self.frame_index, millis, grid)
                    with self.lock:
                        self.newest = (self.frame_index, millis, grid)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc
                print(f"Sensor reader stopped: {exc}", file=sys.stderr, flush=True)

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def read_latest(self):
        """The newest frame since the last call, or None."""
        self.start()
        with self.lock:
            newest, self.newest = self.newest, None
        return newest

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        try:
            self.port.close()
        except Exception:
            pass
        print(f"Stopped; {self.frame_index} frames, {self.resyncs} resyncs.")


class SessionRecorder:
    """Writes every frame to a capture-style session: sensor_raw.csv plus session.json."""

    def __init__(self, root, shape=(ROWS, COLS), meta=None, tool="live_sensor_2d.py --record"):
        self.session = Path(root).expanduser().resolve() / datetime.now().strftime(
            "capture_%Y%m%d_%H%M%S"
        )
        (self.session / "force").mkdir(parents=True)
        self.rows, self.cols = shape
        self.handle = (self.session / "sensor_raw.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.writer = csv.writer(self.handle)
        self.writer.writerow(
            ["host_received_ns", "frame_index", "device_millis"]
            + [f"r{r + 1}c{c + 1}_adc8" for r in range(self.rows) for c in range(self.cols)]
        )
        self.frames = 0
        self.dropped = 0
        self.last_index = None
        self.first_ns = None
        self.last_ns = None
        self.last_flush = time.monotonic()
        self.meta = {
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            **(meta or {}),
            "rows": self.rows, "cols": self.cols,
        }
        self._write_meta()
        print(f"Recording to {self.session}", flush=True)

    def _write_meta(self, extra=None):
        meta = dict(self.meta)
        meta.update(extra or {})
        (self.session / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def __call__(self, host_ns, frame_index, millis, grid):
        if self.last_index is not None and frame_index > self.last_index + 1:
            self.dropped += frame_index - self.last_index - 1
        self.last_index = frame_index
        if self.first_ns is None:
            self.first_ns = host_ns
        self.last_ns = host_ns
        self.writer.writerow([host_ns, frame_index, millis] + grid.ravel().tolist())
        self.frames += 1
        now = time.monotonic()
        if now - self.last_flush >= 1.0:
            self.last_flush = now
            self.handle.flush()

    def close(self):
        self.handle.close()
        duration = (self.last_ns - self.first_ns) / 1e9 if self.frames > 1 else 0.0
        self._write_meta({
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "sensor_stats": {
                "frames": self.frames, "dropped_frames": self.dropped, "duration_s": round(duration, 3),
                "fps_measured": round((self.frames - 1) / duration, 3) if duration > 0 else None,
            },
        })
        print(
            f"Recorded {self.frames} frames ({self.dropped} dropped, {duration:.1f} s) to "
            f"{self.session / 'sensor_raw.csv'}\n"
            f"Put the IntelliMESUR export in {self.session / 'force'}",
            flush=True,
        )


class CsvFollowSource:
    """Tails sensor_raw.csv of a running capture instead of opening the port."""

    CHECK_NEWER_S = 2.0

    def __init__(self, path):
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} does not exist")
        self.follow_root = self.path.is_dir() and not (
            self.path / "sensor_raw.csv"
        ).exists()
        self.handle = None
        self.csv_path = None
        self.remainder = ""
        self.last_check = 0.0
        self.switched = False  # set when a new session is picked up
        self.shape = (ROWS, COLS)
        self.fields = 3 + ROWS * COLS  # host_received_ns, frame_index, device_millis, cells
        if self.follow_root:
            print(f"Watching {self.path} for capture_* sessions ...", flush=True)
        self._open_newest()

    def _newest_csv(self):
        if self.path.is_file():
            return self.path
        if not self.follow_root:
            return self.path / "sensor_raw.csv"
        sessions = sorted(
            candidate
            for candidate in self.path.glob("capture_*/sensor_raw.csv")
        )
        return sessions[-1] if sessions else None

    def _open_newest(self):
        self.last_check = time.time()
        newest = self._newest_csv()
        if newest is None or newest == self.csv_path:
            return
        if self.handle is not None:
            self.handle.close()
        # Plain reads do not lock the file on Windows, so the capture keeps
        # writing. Start at the end: only frames from now on are shown.
        self.handle = newest.open("r", encoding="utf-8", newline="")
        head = self.handle.readline().strip().split(",")
        try:
            self.shape = header_grid(head)
        except ValueError:
            pass  # header not written yet; keep the previous grid
        self.fields = 3 + self.shape[0] * self.shape[1]
        self.handle.seek(0, 2)
        self.csv_path = newest
        self.remainder = ""
        self.switched = True
        print(f"Following {newest} ({self.shape[0]}x{self.shape[1]})", flush=True)

    def read_latest(self):
        if self.follow_root and time.time() - self.last_check >= self.CHECK_NEWER_S:
            self._open_newest()
        if self.handle is None:
            return None
        chunk = self.handle.read()
        if not chunk:
            return None
        lines = (self.remainder + chunk).split("\n")
        self.remainder = lines.pop()  # partial line until the next flush
        for line in reversed(lines):
            fields = line.strip().split(",")
            if len(fields) != self.fields or not fields[1].isdigit():
                continue
            grid = np.array(fields[3:], dtype=np.uint8).reshape(self.shape)
            return int(fields[1]), int(fields[2]), grid
        return None

    def close(self):
        if self.handle is not None:
            self.handle.close()
        print("Stopped following the capture.")


class SimulatedSource:
    """Synthetic stream so the display can be checked without hardware."""

    def __init__(self, fps, baseline_adc, amplitude, paced=True, warmup_s=1.5,
                 shape=(ROWS, COLS), lowers_adc=True):
        self.fps = fps
        self.baseline_adc = baseline_adc
        self.amplitude = amplitude
        self.paced = paced
        # Stay unloaded for warmup_s so the viewer captures a clean no-load
        # baseline before the synthetic press starts, exactly as on the bench.
        self.warmup_s = warmup_s
        self.shape = tuple(shape)
        self.lowers_adc = lowers_adc
        self.period = 1.0 / fps
        self.start = time.perf_counter()
        self.last_emit = 0.0
        self.frame_index = 0
        self.random = np.random.default_rng(7)
        self.rows, self.columns = np.indices(self.shape)

    def read_latest(self):
        now = time.perf_counter()
        if self.paced and now - self.last_emit < self.period:
            return None
        self.last_emit = now

        elapsed = (
            now - self.start if self.paced else self.frame_index * self.period
        )
        # A press that circles the array while its force pulses, so the 2D
        # image, the centroid marker, and the trend plot all show motion.
        active = max(0.0, elapsed - self.warmup_s)
        n_rows, n_cols = self.shape
        radius = 0.25 * min(n_rows, n_cols)
        center_row = (n_rows - 1) / 2 + radius * math.sin(active * 0.7)
        center_column = (n_cols - 1) / 2 + radius * math.cos(active * 0.7)
        # Ramp in from zero so the press never contaminates the baseline.
        press = 0.0 if active <= 0.0 else 0.5 * (1.0 - math.cos(active * 1.6))
        distance_squared = (
            (self.rows - center_row) ** 2 + (self.columns - center_column) ** 2
        )
        sigma = max(0.6, 2.2 * min(n_rows, n_cols) / 16.0)
        blob = np.exp(-distance_squared / (2.0 * sigma**2))

        # The 4x4 ESP32 board (and the capture pipeline's convention): pressing
        # lowers the ADC value, which raises V_sensor. The 16x16 board's rises.
        direction = -1.0 if self.lowers_adc else 1.0
        values = self.baseline_adc + direction * self.amplitude * press * blob
        values += self.random.normal(0.0, 0.8, self.shape)
        grid = np.clip(np.rint(values), 0, 255).astype(np.uint8)

        self.frame_index += 1
        return self.frame_index, int(elapsed * 1000), grid

    def close(self):
        print("Simulated stream stopped.")


def list_serial_ports():
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for item in ports:
        tag = "   <- ESP32 board (CP2104)" if (item.vid, item.pid) == ESP32_VID_PID else ""
        print(f"  {item.device:10s}  {item.description}{tag}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Live 2D image of a tactile sensor array (16x16 M16B or 4x4 ESP32 board).",
    )
    parser.add_argument(
        "port",
        nargs="?",
        help="Serial port, for example COM7. Omit with --simulate; --board esp32 finds its port by itself.",
    )
    parser.add_argument(
        "--board",
        choices=tuple(BOARDS),
        default="m16b",
        help="m16b: 16x16 array on the Arduino Nano (default). esp32: 4x4 test board on the ESP32 + AD7466.",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("ROW0", "COL0", "ROWS", "COLS"),
        default=None,
        help=f"--board esp32: the wired pads inside the 32x32 frame (default {' '.join(map(str, ESP32_DEFAULT_ROI))}).",
    )
    parser.add_argument("--baud", type=int, default=None, help="Default: the board's rate.")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Generate a synthetic press instead of reading the board.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "Render simulated frames headlessly, save a PNG, and exit. Use "
            "this to verify the display without hardware or a window."
        ),
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Print the available serial ports and exit.",
    )
    parser.add_argument(
        "--mode",
        choices=("change", "voltage", "raw"),
        default="change",
        help=(
            "change: sensor voltage minus no-load baseline (default, best for "
            "seeing contact). voltage: absolute sensor voltage. raw: 8-bit ADC."
        ),
    )
    parser.add_argument("--adc-ref", type=float, default=None, help="ADC reference in V; default: the board's.")
    parser.add_argument("--vdrive", type=float, default=None, help="Sensor drive in V; default: the board's.")
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=24,
        help="No-load frames medianed into the baseline for --mode change.",
    )
    parser.add_argument(
        "--range",
        dest="display_range",
        type=float,
        help="Colour-scale half-range in volts; defaults to 0.5 for change.",
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=4.0,
        help=(
            "--mode change: zero changes of at most this many ADC counts. Row/"
            "column ghosting is a 1-3 count step and real contact jumps past "
            "~30, so 4 hides both noise and ghosting; 0 shows everything."
        ),
    )
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Show baseline minus sensor voltage instead.",
    )
    parser.add_argument("--vmax", type=int, default=255, help="--mode raw only.")
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.05,
        help="Volts of change above which a cell counts as contact.",
    )
    parser.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bicubic upsampling for a continuous image (default: enabled).",
    )
    parser.add_argument("--labels", action="store_true", help="Per-cell values.")
    parser.add_argument("--cmap", help="Override the colormap.")
    parser.add_argument(
        "--rot90",
        type=int,
        default=0,
        help="Rotate the image by N*90 degrees to match the physical sensor.",
    )
    parser.add_argument("--flip-ud", action="store_true")
    parser.add_argument("--flip-lr", action="store_true")
    parser.add_argument(
        "--trend-seconds",
        type=float,
        default=20.0,
        help="Time span of the signal trend plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for PNG snapshots.",
    )
    parser.add_argument("--handshake-timeout", type=float, default=10.0)
    parser.add_argument(
        "--record",
        type=Path,
        metavar="ROOT",
        help=(
            "Also save every frame to ROOT/capture_<stamp>/sensor_raw.csv "
            "with a session.json (serial port only)."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="S",
        help="Quit by itself after S seconds (for scripted bench runs).",
    )
    parser.add_argument(
        "--follow",
        type=Path,
        metavar="PATH",
        help=(
            "Show a running capture instead of opening the serial port: a "
            "session folder, its sensor_raw.csv, or the capture root."
        ),
    )
    args = parser.parse_args(argv)

    board = BOARDS[args.board]
    if args.baud is None:
        args.baud = board["baud"]
    if args.adc_ref is None:
        args.adc_ref = board["adc_reference_v"]
    if args.vdrive is None:
        args.vdrive = board["sensor_drive_v"]
    if args.roi is None:
        args.roi = list(ESP32_DEFAULT_ROI)
    elif args.board != "esp32":
        parser.error("--roi only applies to --board esp32")
    if args.selftest:
        args.simulate = True
    needs_port = args.board != "esp32"  # the ESP32 board is found by its USB id
    if needs_port and not (args.list_ports or args.simulate or args.port or args.follow):
        parser.error("a serial port is required unless --simulate or --follow is used")
    if args.record and (args.follow or args.simulate or (needs_port and not args.port)):
        parser.error("--record needs the serial board (not --follow or --simulate)")
    if args.baseline_frames < 1:
        parser.error("--baseline-frames must be at least 1")
    if args.deadband < 0:
        parser.error("--deadband must be non-negative")
    if args.gain <= 0:
        parser.error("--gain must be positive")
    if args.display_range is not None and args.display_range <= 0:
        parser.error("--range must be positive")
    if args.adc_ref <= 0 or args.vdrive <= 0:
        parser.error("--adc-ref and --vdrive must be positive")
    if args.contact_threshold < 0:
        parser.error("--contact-threshold must be non-negative")
    if args.trend_seconds <= 0:
        parser.error("--trend-seconds must be positive")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    return args


class LiveSensor2D:
    def __init__(self, args, source):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.args = args
        self.source = source
        self.rows, self.cols = getattr(source, "shape", (ROWS, COLS))
        self.gain = args.gain
        self.smooth = args.smooth
        self.show_labels = args.labels

        self.baseline_samples = []
        self.baseline = None
        self.frame_index = 0
        self.frames_since_tick = 0
        self.fps = 0.0
        self.fps_t0 = time.perf_counter()
        # The trend axis uses the board's own millis column, so it stays a
        # true sensor timeline even when rendering lags behind acquisition.
        self.first_millis = None
        self.trend_times = deque()
        self.trend_values = deque()
        self.held_max = None  # (value, row, column) since start or last reset
        self.closed = False

        if args.mode == "change":
            self.limit = args.display_range or 0.5
            self.label = (
                "Baseline - sensor voltage (V)"
                if args.invert
                else "Sensor voltage - baseline (V)"
            )
        elif args.mode == "voltage":
            self.limit = args.display_range or args.vdrive
            self.label = "Sensor voltage (V)"
        else:
            self.limit = float(args.vmax)
            self.label = "Raw ADC (8-bit)"

        self._build_figure()

    # ---- data ----

    def orient(self, grid):
        if self.args.rot90:
            grid = np.rot90(grid, self.args.rot90)
        if self.args.flip_ud:
            grid = np.flipud(grid)
        if self.args.flip_lr:
            grid = np.fliplr(grid)
        return np.ascontiguousarray(grid)

    def to_voltage(self, grid):
        return self.args.vdrive - grid.astype(np.float64) * (
            self.args.adc_ref / ADC_MAX
        )

    def rearm_baseline(self):
        self.baseline = None
        self.baseline_samples = []

    def to_display(self, grid):
        """Return (matrix, ready). `ready` is False while baselining."""
        if self.args.mode == "raw":
            return grid.astype(np.float64), True

        voltage = self.to_voltage(grid)
        if self.args.mode == "voltage":
            return voltage, True

        if self.baseline is None:
            self.baseline_samples.append(voltage)
            if len(self.baseline_samples) >= self.args.baseline_frames:
                self.baseline = np.median(
                    np.stack(self.baseline_samples), axis=0
                )
                self.baseline_samples = []
            else:
                return np.zeros(grid.shape), False

        sign = -1.0 if self.args.invert else 1.0
        change = voltage - self.baseline
        floor = self.args.deadband * self.args.adc_ref / ADC_MAX
        change = np.where(np.abs(change) > floor + 1e-9, change, 0.0)
        return sign * change * self.gain, True

    # ---- figure ----

    def _build_figure(self):
        plt = self.plt
        rows, cols = self.rows, self.cols
        if self.args.rot90 % 2:
            rows, cols = cols, rows
        self.fig = plt.figure(figsize=(12.5, 6.4))
        try:
            self.fig.canvas.manager.set_window_title(
                f"{self.rows}x{self.cols} tactile sensor - live 2D"
            )
        except AttributeError:
            pass
        grid = self.fig.add_gridspec(
            1,
            2,
            width_ratios=(1.0, 0.92),
            left=0.05,
            right=0.965,
            top=0.90,
            bottom=0.16,
            wspace=0.24,
        )

        self.image_axis = self.fig.add_subplot(grid[0, 0])
        cmap = self.args.cmap or MODE_COLORMAPS[self.args.mode]
        vmin = -self.limit if self.args.mode == "change" else 0.0
        self.image = self.image_axis.imshow(
            np.zeros((rows, cols)),
            cmap=cmap,
            vmin=vmin,
            vmax=self.limit,
            interpolation="bicubic" if self.smooth else "nearest",
            origin="upper",
            extent=(-0.5, cols - 0.5, rows - 0.5, -0.5),
        )
        step = 1 if max(rows, cols) <= 8 else 2
        self.image_axis.set_xticks(range(0, cols, step))
        self.image_axis.set_yticks(range(0, rows, step))
        self.image_axis.set_xticklabels(
            [f"C{index + 1}" for index in range(0, cols, step)], fontsize=8
        )
        self.image_axis.set_yticklabels(
            [f"R{index + 1}" for index in range(0, rows, step)], fontsize=8
        )
        self.image_axis.set_xlabel("Column")
        self.image_axis.set_ylabel("Row")
        colorbar = self.fig.colorbar(
            self.image, ax=self.image_axis, shrink=0.86, pad=0.03
        )
        colorbar.set_label(self.label)
        self.colorbar = colorbar

        self.peak_marker, = self.image_axis.plot(
            [], [], "x", color="#101010", markersize=9, markeredgewidth=1.8
        )
        self.centroid_marker, = self.image_axis.plot(
            [],
            [],
            "o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="#0b0b0b",
            markeredgewidth=1.6,
        )
        self.saturation_marker, = self.image_axis.plot(
            [],
            [],
            "s",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="#e8202a",
            markeredgewidth=1.8,
        )
        self.image_title = self.image_axis.set_title(
            "waiting for frames ...", fontsize=10
        )

        self.label_artists = []
        label_size = 5.2 if rows * cols > 64 else 9.0
        for row in range(rows):
            for column in range(cols):
                self.label_artists.append(
                    self.image_axis.text(
                        column,
                        row,
                        "",
                        ha="center",
                        va="center",
                        fontsize=label_size,
                        color="#111111",
                    )
                )

        self.trend_axis = self.fig.add_subplot(grid[0, 1])
        self.trend_line, = self.trend_axis.plot(
            [], [], color="#c03028", linewidth=1.2
        )
        self.trend_axis.set_xlabel("Time (s)")
        self.trend_axis.set_ylabel(self.trend_label())
        self.trend_axis.grid(True, alpha=0.25)
        self.trend_axis.set_title("Signal trend", fontsize=10)
        if self.args.mode == "raw":
            self.trend_axis.axhline(
                SATURATION_ADC, color="#e8202a", linewidth=1.0, linestyle="--"
            )
            self.trend_axis.text(
                0.01, SATURATION_ADC, " saturation", color="#e8202a",
                fontsize=8, va="bottom",
                transform=self.trend_axis.get_yaxis_transform(),
            )

        self.status = self.fig.text(
            0.5,
            0.035,
            self.status_text(),
            ha="center",
            va="center",
            fontsize=8.5,
            color="#404040",
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def trend_label(self):
        if self.args.mode == "change":
            return "Peak change (V)"
        if self.args.mode == "voltage":
            return "Mean voltage (V)"
        return "Peak cell ADC"

    def status_text(self):
        return (
            f"board={self.args.board}  mode={self.args.mode}  gain={self.gain:.2f}  "
            f"range=+/-{self.limit:.2f}  "
            f"smooth={'on' if self.smooth else 'off'}   |   "
            "b baseline   i smooth   v values   +/- gain   ][ range   "
            + ("r reset max   " if self.args.mode == "raw" else "")
            + "s snapshot   q quit"
        )

    # ---- interaction ----

    def on_key_press(self, event):
        key = event.key
        if key == "b":
            self.rearm_baseline()
        elif key == "r":
            self.held_max = None
        elif key == "i":
            self.smooth = not self.smooth
            self.image.set_interpolation(
                "bicubic" if self.smooth else "nearest"
            )
        elif key in ("+", "="):
            self.gain = min(self.gain * 1.25, 200.0)
        elif key == "-":
            self.gain = max(self.gain / 1.25, 0.01)
        elif key == "]":
            self.set_limit(self.limit * 1.25)
        elif key == "[":
            self.set_limit(max(self.limit / 1.25, 1e-3))
        elif key == "v":
            self.show_labels = not self.show_labels
            if not self.show_labels:
                for artist in self.label_artists:
                    artist.set_text("")
        elif key == "s":
            self.save_snapshot()
        elif key in ("q", "escape"):
            self.plt.close(self.fig)
            return
        self.status.set_text(self.status_text())
        self.fig.canvas.draw_idle()

    def set_limit(self, limit):
        self.limit = limit
        vmin = -limit if self.args.mode == "change" else 0.0
        self.image.set_clim(vmin, limit)

    def save_snapshot(self):
        self.args.output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.args.output / f"sensor_2d_{stamp}.png"
        self.fig.savefig(path, dpi=140)
        print(f"Snapshot: {path}", flush=True)

    def on_close(self, _event):
        self.closed = True

    # ---- update ----

    def update(self, sample):
        frame_index, millis, raw_grid = sample
        if getattr(self.source, "switched", False):
            # --follow picked up a new capture session: start over.
            self.source.switched = False
            self.rearm_baseline()
            self.first_millis = None
            self.held_max = None
            self.trend_times.clear()
            self.trend_values.clear()
        if self.first_millis is None:
            self.first_millis = millis
        grid = self.orient(raw_grid)
        matrix, ready = self.to_display(grid)
        self.frame_index = frame_index

        now = time.perf_counter()
        self.frames_since_tick += 1
        if now - self.fps_t0 >= 1.0:
            self.fps = self.frames_since_tick / (now - self.fps_t0)
            self.frames_since_tick = 0
            self.fps_t0 = now

        self.image.set_data(matrix)

        if not ready:
            collected = len(self.baseline_samples)
            self.image_title.set_text(
                f"collecting no-load baseline {collected}/"
                f"{self.args.baseline_frames} - keep the sensor unloaded"
            )
            self.peak_marker.set_data([], [])
            self.centroid_marker.set_data([], [])
            return

        peak_row, peak_column = np.unravel_index(
            int(np.argmax(matrix)), matrix.shape
        )
        self.peak_marker.set_data([peak_column], [peak_row])

        if self.args.mode == "change":
            contact = matrix > self.args.contact_threshold
            contact_cells = int(contact.sum())
            if contact_cells:
                weights = np.where(contact, matrix, 0.0)
                total = weights.sum()
                rows, columns = np.indices(matrix.shape)
                centroid_row = float((weights * rows).sum() / total)
                centroid_column = float((weights * columns).sum() / total)
                self.centroid_marker.set_data(
                    [centroid_column], [centroid_row]
                )
                contact_text = (
                    f"contact {contact_cells} cells @ "
                    f"(R{centroid_row + 1:.1f}, C{centroid_column + 1:.1f})"
                )
            else:
                self.centroid_marker.set_data([], [])
                contact_text = "no contact"
            trend_value = float(matrix.max())
        elif self.args.mode == "voltage":
            self.centroid_marker.set_data([], [])
            contact_text = f"mean {matrix.mean():.3f} V"
            trend_value = float(matrix.mean())
        else:
            self.centroid_marker.set_data([], [])
            peak = float(matrix[peak_row, peak_column])
            if self.held_max is None or peak > self.held_max[0]:
                self.held_max = (peak, peak_row, peak_column)
            saturated_rows, saturated_columns = np.nonzero(
                matrix >= SATURATION_ADC
            )
            self.saturation_marker.set_data(saturated_columns, saturated_rows)
            held, held_row, held_column = self.held_max
            contact_text = (
                f"peak {peak:.0f}  |  held max {held:.0f} @ "
                f"(R{held_row + 1}, C{held_column + 1})  |  "
                f"saturated {len(saturated_rows)}"
            )
            trend_value = peak

        unit = "" if self.args.mode == "raw" else " V"
        self.image_title.set_text(
            f"frame {frame_index}  |  {self.fps:.1f} fps  |  "
            f"{matrix.min():+.3f} to {matrix.max():+.3f}{unit}\n"
            f"peak @ (R{peak_row + 1}, C{peak_column + 1})  |  {contact_text}"
        )

        elapsed = (millis - self.first_millis) / 1000.0
        self.trend_times.append(elapsed)
        self.trend_values.append(trend_value)
        while (
            self.trend_times
            and elapsed - self.trend_times[0] > self.args.trend_seconds
        ):
            self.trend_times.popleft()
            self.trend_values.popleft()
        self.trend_line.set_data(self.trend_times, self.trend_values)
        self.trend_axis.set_xlim(
            max(0.0, elapsed - self.args.trend_seconds), max(elapsed, 1.0)
        )
        if self.args.mode == "raw":
            self.trend_axis.set_ylim(0.0, ADC_MAX + 5.0)
        else:
            low = min(self.trend_values)
            high = max(self.trend_values)
            margin = max(0.05 * (high - low), 1e-3)
            self.trend_axis.set_ylim(low - margin, high + margin)

        if self.show_labels:
            n_cols = matrix.shape[1]
            for flat_index, artist in enumerate(self.label_artists):
                row, column = divmod(flat_index, n_cols)
                value = matrix[row, column]
                artist.set_text(
                    f"{value:.0f}"
                    if self.args.mode == "raw"
                    else f"{value:+.2f}"
                )

    def run(self):
        self.plt.ion()
        self.plt.show()
        deadline = (
            time.perf_counter() + self.args.duration
            if self.args.duration else None
        )
        try:
            while not self.closed and self.plt.fignum_exists(self.fig.number):
                if deadline is not None and time.perf_counter() >= deadline:
                    print(f"--duration {self.args.duration:g} s reached.", flush=True)
                    break
                sample = self.source.read_latest()
                if sample is not None:
                    self.update(sample)
                    self.fig.canvas.draw_idle()
                self.plt.pause(0.001)
        except KeyboardInterrupt:
            print()
        finally:
            self.source.close()


def simulated_source(args, paced=True, warmup_s=1.5):
    board = BOARDS[args.board]
    shape = (args.roi[2], args.roi[3]) if args.board == "esp32" else (board["rows"], board["cols"])
    # Unloaded levels as measured: ~10 counts on the 16x16 board, ~135 on the 4x4 board.
    baseline_adc = 135.0 if board["pressure_lowers_adc"] else 180.0
    return SimulatedSource(24.0, baseline_adc, 60.0, paced=paced, warmup_s=warmup_s,
                           shape=shape, lowers_adc=board["pressure_lowers_adc"])


def run_selftest(args):
    import matplotlib

    matplotlib.use("Agg")
    source = simulated_source(args, paced=False, warmup_s=args.baseline_frames / 24.0)
    viewer = LiveSensor2D(args, source)
    rendered = 0
    for _ in range(args.baseline_frames + 80):
        sample = source.read_latest()
        if sample is None:
            continue
        viewer.update(sample)
        rendered += 1
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"sensor_2d_selftest_{args.board}.png"
    viewer.fig.savefig(path, dpi=110)
    print(f"Rendered {rendered} simulated {source.shape[0]}x{source.shape[1]} frames.")
    print(f"Baseline ready: {viewer.baseline is not None}")
    print(f"Selftest image: {path}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.list_ports:
        list_serial_ports()
        return 0
    if args.selftest:
        return run_selftest(args)

    import matplotlib

    matplotlib.use("TkAgg")

    if args.simulate:
        print("Simulated sensor stream (no hardware).", flush=True)
        source = simulated_source(args)
    elif args.follow:
        try:
            source = CsvFollowSource(args.follow)
        except Exception as exc:
            print(f"Cannot follow the capture: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            if args.board == "esp32":
                source = Esp32Source(args.port, args.baud, tuple(args.roi))
            else:
                source = SerialSource(args.port, args.baud, args.handshake_timeout)
        except Exception as exc:
            print(f"Cannot start the sensor stream: {exc}", file=sys.stderr)
            return 1

    recorder = None
    if args.record:
        recorder = SessionRecorder(args.record, source.shape, source.meta())
        source.on_frame = recorder
    try:
        LiveSensor2D(args, source).run()
    finally:
        if recorder is not None:
            recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
