#!/usr/bin/env python3
"""Live 2D pressure image of the 16x16 tactile sensor (M16B firmware).

A standalone bench tool for checking what the sensor actually sees, before
committing to a full camera + force capture. It shows the array as a smooth
2D image using the same signal definition as the capture pipeline
(V_sensor = Vdrive - V_adc, then baseline subtraction), so what looks good
here looks the same in replay_capture_multimodal.py.

Usage:
    python live_sensor_2d.py COM7                 # live, baseline-subtracted
    python live_sensor_2d.py COM7 --mode voltage  # absolute sensor voltage
    python live_sensor_2d.py COM7 --mode raw      # raw 8-bit ADC
    python live_sensor_2d.py --simulate           # no hardware, synthetic press
    python live_sensor_2d.py --list-ports         # find the Arduino COM port
    python live_sensor_2d.py --selftest           # headless render check

Keys while the window is focused:
    b        re-capture the no-load baseline
    i        toggle smooth / raw-cell rendering
    v        toggle per-cell value labels
    + / -    display gain up / down
    ] / [    display range up / down
    s        save a PNG snapshot
    q / Esc  quit

Dependencies: pyserial, numpy, matplotlib. No cameras or force gauge needed.
This tool only reads; it never writes into a capture session.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

# ---- Protocol constants (must match MatrixArrayDual4067Binary.ino) ----
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

MODE_COLORMAPS = {
    "change": "RdBu_r",
    "voltage": "turbo",
    "raw": "viridis",
}


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

    def __init__(self, port, baud, handshake_timeout):
        import serial

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

    def read_latest(self):
        now = time.time()
        if now - self.last_ping >= PING_INTERVAL_S:
            self.port.write(b"M16_PING\n")
            self.last_ping = now

        chunk = self.port.read(4096)
        if chunk:
            self.buffer.extend(chunk)

        newest = None
        for frame_index, millis, grid in parse_frames(self.buffer):
            # The firmware sends each frame twice (once before and once after
            # the camera trigger); keep the first copy only.
            if frame_index == self.last_index:
                continue
            self.last_index = frame_index
            newest = (frame_index, millis, grid)
        return newest

    def close(self):
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


class SimulatedSource:
    """Synthetic stream so the display can be checked without hardware."""

    def __init__(self, fps, baseline_adc, amplitude, paced=True, warmup_s=1.5):
        self.fps = fps
        self.baseline_adc = baseline_adc
        self.amplitude = amplitude
        self.paced = paced
        # Stay unloaded for warmup_s so the viewer captures a clean no-load
        # baseline before the synthetic press starts, exactly as on the bench.
        self.warmup_s = warmup_s
        self.period = 1.0 / fps
        self.start = time.perf_counter()
        self.last_emit = 0.0
        self.frame_index = 0
        self.random = np.random.default_rng(7)
        self.rows, self.columns = np.indices((ROWS, COLS))

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
        center_row = 7.5 + 4.0 * math.sin(active * 0.7)
        center_column = 7.5 + 4.0 * math.cos(active * 0.7)
        # Ramp in from zero so the press never contaminates the baseline.
        press = 0.0 if active <= 0.0 else 0.5 * (1.0 - math.cos(active * 1.6))
        distance_squared = (
            (self.rows - center_row) ** 2 + (self.columns - center_column) ** 2
        )
        blob = np.exp(-distance_squared / (2.0 * 2.2**2))

        # Pressing lowers the measured ADC value, which raises V_sensor.
        values = self.baseline_adc - self.amplitude * press * blob
        values += self.random.normal(0.0, 0.8, (ROWS, COLS))
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
        print(f"  {item.device:10s}  {item.description}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Live 2D image of the 16x16 tactile sensor.",
    )
    parser.add_argument(
        "port",
        nargs="?",
        help="Serial port, for example COM7. Omit with --simulate.",
    )
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Generate a synthetic press instead of reading the Arduino.",
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
    parser.add_argument("--adc-ref", type=float, default=5.0)
    parser.add_argument("--vdrive", type=float, default=5.0)
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
    args = parser.parse_args(argv)

    if args.selftest:
        args.simulate = True
    if not args.list_ports and not args.simulate and not args.port:
        parser.error("a serial port is required unless --simulate is used")
    if args.baseline_frames < 1:
        parser.error("--baseline-frames must be at least 1")
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
    return args


class LiveSensor2D:
    def __init__(self, args, source):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.args = args
        self.source = source
        self.gain = args.gain
        self.smooth = args.smooth
        self.show_labels = args.labels

        self.baseline_samples = []
        self.baseline = None
        self.frame_index = 0
        self.frames_since_tick = 0
        self.fps = 0.0
        self.fps_t0 = time.perf_counter()
        # The trend axis uses the Arduino's own device_millis, so it stays a
        # true sensor timeline even when rendering lags behind acquisition.
        self.first_millis = None
        self.trend_times = deque()
        self.trend_values = deque()
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
                return np.zeros((ROWS, COLS)), False

        sign = -1.0 if self.args.invert else 1.0
        return sign * (voltage - self.baseline) * self.gain, True

    # ---- figure ----

    def _build_figure(self):
        plt = self.plt
        self.fig = plt.figure(figsize=(12.5, 6.4))
        try:
            self.fig.canvas.manager.set_window_title(
                "16x16 tactile sensor - live 2D"
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
            np.zeros((ROWS, COLS)),
            cmap=cmap,
            vmin=vmin,
            vmax=self.limit,
            interpolation="bicubic" if self.smooth else "nearest",
            origin="upper",
            extent=(-0.5, COLS - 0.5, ROWS - 0.5, -0.5),
        )
        self.image_axis.set_xticks(range(0, COLS, 2))
        self.image_axis.set_yticks(range(0, ROWS, 2))
        self.image_axis.set_xticklabels(
            [f"C{index + 1}" for index in range(0, COLS, 2)], fontsize=8
        )
        self.image_axis.set_yticklabels(
            [f"R{index + 1}" for index in range(0, ROWS, 2)], fontsize=8
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
        self.image_title = self.image_axis.set_title(
            "waiting for frames ...", fontsize=10
        )

        self.label_artists = []
        for row in range(ROWS):
            for column in range(COLS):
                self.label_artists.append(
                    self.image_axis.text(
                        column,
                        row,
                        "",
                        ha="center",
                        va="center",
                        fontsize=5.2,
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
        return "Mean ADC"

    def status_text(self):
        return (
            f"mode={self.args.mode}  gain={self.gain:.2f}  "
            f"range=+/-{self.limit:.2f}  "
            f"smooth={'on' if self.smooth else 'off'}   |   "
            "b baseline   i smooth   v values   +/- gain   ][ range   "
            "s snapshot   q quit"
        )

    # ---- interaction ----

    def on_key_press(self, event):
        key = event.key
        if key == "b":
            self.rearm_baseline()
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
            contact_text = f"mean {matrix.mean():.1f}"
            trend_value = float(matrix.mean())

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
        low = min(self.trend_values)
        high = max(self.trend_values)
        margin = max(0.05 * (high - low), 1e-3)
        self.trend_axis.set_ylim(low - margin, high + margin)

        if self.show_labels:
            for flat_index, artist in enumerate(self.label_artists):
                row, column = divmod(flat_index, COLS)
                value = matrix[row, column]
                artist.set_text(
                    f"{value:.0f}"
                    if self.args.mode == "raw"
                    else f"{value:+.2f}"
                )

    def run(self):
        self.plt.ion()
        self.plt.show()
        try:
            while not self.closed and self.plt.fignum_exists(self.fig.number):
                sample = self.source.read_latest()
                if sample is not None:
                    self.update(sample)
                    self.fig.canvas.draw_idle()
                self.plt.pause(0.001)
        except KeyboardInterrupt:
            print()
        finally:
            self.source.close()


def run_selftest(args):
    import matplotlib

    matplotlib.use("Agg")
    source = SimulatedSource(
        24.0,
        180.0,
        60.0,
        paced=False,
        warmup_s=args.baseline_frames / 24.0,
    )
    viewer = LiveSensor2D(args, source)
    rendered = 0
    for _ in range(args.baseline_frames + 80):
        sample = source.read_latest()
        if sample is None:
            continue
        viewer.update(sample)
        rendered += 1
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "sensor_2d_selftest.png"
    viewer.fig.savefig(path, dpi=110)
    print(f"Rendered {rendered} simulated frames.")
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
        source = SimulatedSource(24.0, 180.0, 60.0)
    else:
        try:
            source = SerialSource(args.port, args.baud, args.handshake_timeout)
        except Exception as exc:
            print(f"Cannot start the sensor stream: {exc}", file=sys.stderr)
            return 1

    LiveSensor2D(args, source).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
