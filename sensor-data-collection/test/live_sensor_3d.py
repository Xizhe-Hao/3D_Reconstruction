#!/usr/bin/env python3
"""Live 3D bar view of the 16x16 tactile sensor (M16B firmware).

The bench counterpart of the 3D panel in replay_capture_multimodal.py: same
signal definition (V_sensor = Vdrive - V_adc, then baseline subtraction), same
bar geometry and shading, but fed live from the Arduino instead of from a
recorded session. Use it to judge the deformation shape before committing to a
full camera + force capture.

The serial protocol, the frame parser, and the data sources are imported from
live_sensor_2d.py so that the M16B decoding lives in exactly one place.

Usage:
    python live_sensor_3d.py COM3                  # live, baseline-subtracted
    python live_sensor_3d.py COM3 --style surface  # smooth surface, faster
    python live_sensor_3d.py COM3 --mode voltage   # absolute sensor voltage
    python live_sensor_3d.py --simulate            # no hardware, synthetic press
    python live_sensor_3d.py --list-ports          # find the Arduino COM port
    python live_sensor_3d.py --selftest            # headless render + timing

Keys while the window is focused:
    b        re-capture the no-load baseline
    r        toggle slow auto-rotation
    t        toggle bars / surface
    0        reset the viewing angle
    + / -    display gain up / down
    ] / [    display range up / down
    s        save a PNG snapshot
    q / Esc  quit

Drag with the mouse to orbit the view at any time.

Dependencies: pyserial, numpy, matplotlib. No cameras or force gauge needed.
This tool only reads; it never writes into a capture session.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from live_sensor_2d import (
    ADC_MAX,
    COLS,
    ROWS,
    SerialSource,
    SimulatedSource,
    list_serial_ports,
)

CELL_COUNT = ROWS * COLS
BAR_MARGIN = 0.10
# Top, front, back, left, right face brightness, so the bars read as solid.
FACE_SHADES = np.asarray((1.0, 0.78, 0.88, 0.70, 0.82), dtype=np.float32)

MODE_COLORMAPS = {
    "change": "RdBu_r",
    "voltage": "turbo",
    "raw": "viridis",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Live 3D bar view of the 16x16 tactile sensor.",
    )
    parser.add_argument(
        "port",
        nargs="?",
        help="Serial port, for example COM3. Omit with --simulate.",
    )
    parser.add_argument("--baud", type=int, default=500_000)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Generate a synthetic press instead of reading the Arduino.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "Render simulated frames headlessly, report the achievable frame "
            "rate, save a PNG, and exit."
        ),
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Print the available serial ports and exit.",
    )
    parser.add_argument(
        "--style",
        choices=("bars", "surface"),
        default="bars",
        help=(
            "bars: one 3D bar per cell, identical to the replay view. "
            "surface: a smooth surface, noticeably faster to redraw."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("change", "voltage", "raw"),
        default="change",
        help=(
            "change: sensor voltage minus no-load baseline (default). "
            "voltage: absolute sensor voltage. raw: 8-bit ADC."
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
        help="Z-axis half-range in volts; defaults to 0.5 for change.",
    )
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Show baseline minus sensor voltage instead.",
    )
    parser.add_argument("--vmax", type=int, default=255, help="--mode raw only.")
    parser.add_argument("--cmap", help="Override the colormap.")
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--azim", type=float, default=-53.0)
    parser.add_argument(
        "--auto-rotate",
        action="store_true",
        help="Start with the view slowly orbiting.",
    )
    parser.add_argument(
        "--rotate-speed",
        type=float,
        default=12.0,
        help="Auto-rotation speed in degrees per second.",
    )
    parser.add_argument(
        "--rot90",
        type=int,
        default=0,
        help="Rotate the array by N*90 degrees to match the physical sensor.",
    )
    parser.add_argument("--flip-ud", action="store_true")
    parser.add_argument("--flip-lr", action="store_true")
    parser.add_argument(
        "--max-render-fps",
        type=float,
        default=12.0,
        help=(
            "Cap the 3D redraw rate. Acquisition still runs at full speed; "
            "this only skips redraws so the serial buffer never backs up."
        ),
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
    if args.max_render_fps <= 0:
        parser.error("--max-render-fps must be positive")
    return args


class LiveSensor3D:
    def __init__(self, args, source):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.args = args
        self.source = source
        self.gain = args.gain
        self.style = args.style
        self.auto_rotate = args.auto_rotate
        self.elev = args.elev
        self.azim = args.azim

        self.baseline_samples = []
        self.baseline = None
        self.frames_since_tick = 0
        self.fps = 0.0
        self.fps_t0 = time.perf_counter()
        self.last_render = 0.0
        # Geometry build cost only; the canvas draw dominates and is timed
        # separately from the live loop, which knows the true cycle time.
        self.geometry_seconds = deque(maxlen=30)
        self.redraw_seconds = deque(maxlen=30)
        self.closed = False
        self.surface_artist = None

        if args.mode == "change":
            limit = args.display_range or 0.5
            self.z_min, self.z_max = -limit, limit
            self.label = (
                "Baseline - sensor voltage (V)"
                if args.invert
                else "Sensor voltage - baseline (V)"
            )
            # Short axis label: the colorbar beside it already spells out the
            # full quantity, and repeating it crowds the 3D box.
            self.axis_label = "Voltage (V)"
        elif args.mode == "voltage":
            limit = args.display_range or args.vdrive
            self.z_min, self.z_max = 0.0, limit
            self.label = "Sensor voltage (V)"
            self.axis_label = "Voltage (V)"
        else:
            self.z_min, self.z_max = 0.0, float(args.vmax)
            self.label = "Raw ADC (8-bit)"
            self.axis_label = "ADC"

        self._prepare_bar_coordinates()
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

    # ---- geometry ----

    def _prepare_bar_coordinates(self):
        rows, columns = np.indices((ROWS, COLS))
        width = 1.0 - 2.0 * BAR_MARGIN
        self.x0 = columns.ravel().astype(np.float32) + BAR_MARGIN
        self.x1 = self.x0 + width
        self.y0 = rows.ravel().astype(np.float32) + BAR_MARGIN
        self.y1 = self.y0 + width

    def build_bar_geometry(self, matrix):
        """Five quads per cell: top, front, back, left, right."""
        clipped = np.clip(
            np.asarray(matrix, dtype=np.float32).ravel(),
            self.z_min,
            self.z_max,
        )
        z0 = (
            np.zeros_like(clipped)
            if self.z_min >= 0
            else np.minimum(clipped, 0.0)
        )
        z1 = np.maximum(clipped, 0.0)
        faces = np.empty((CELL_COUNT, 5, 4, 3), dtype=np.float32)
        faces[:, 0] = np.stack(
            (
                np.stack((self.x0, self.y0, z1), axis=1),
                np.stack((self.x0, self.y1, z1), axis=1),
                np.stack((self.x1, self.y1, z1), axis=1),
                np.stack((self.x1, self.y0, z1), axis=1),
            ),
            axis=1,
        )
        faces[:, 1] = np.stack(
            (
                np.stack((self.x0, self.y0, z0), axis=1),
                np.stack((self.x0, self.y0, z1), axis=1),
                np.stack((self.x1, self.y0, z1), axis=1),
                np.stack((self.x1, self.y0, z0), axis=1),
            ),
            axis=1,
        )
        faces[:, 2] = np.stack(
            (
                np.stack((self.x0, self.y1, z0), axis=1),
                np.stack((self.x1, self.y1, z0), axis=1),
                np.stack((self.x1, self.y1, z1), axis=1),
                np.stack((self.x0, self.y1, z1), axis=1),
            ),
            axis=1,
        )
        faces[:, 3] = np.stack(
            (
                np.stack((self.x0, self.y0, z0), axis=1),
                np.stack((self.x0, self.y1, z0), axis=1),
                np.stack((self.x0, self.y1, z1), axis=1),
                np.stack((self.x0, self.y0, z1), axis=1),
            ),
            axis=1,
        )
        faces[:, 4] = np.stack(
            (
                np.stack((self.x1, self.y0, z0), axis=1),
                np.stack((self.x1, self.y0, z1), axis=1),
                np.stack((self.x1, self.y1, z1), axis=1),
                np.stack((self.x1, self.y1, z0), axis=1),
            ),
            axis=1,
        )
        colors = np.repeat(self.cmap(self.norm(clipped)), 5, axis=0)
        colors[:, :3] *= np.tile(FACE_SHADES, CELL_COUNT)[:, None]
        np.clip(colors, 0.0, 1.0, out=colors)
        return faces.reshape(CELL_COUNT * 5, 4, 3), colors

    # ---- figure ----

    def _build_figure(self):
        import matplotlib
        import matplotlib.colors as mcolors
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        plt = self.plt
        self.Poly3DCollection = Poly3DCollection
        self.cmap = plt.get_cmap(self.args.cmap or MODE_COLORMAPS[self.args.mode])
        self.norm = mcolors.Normalize(vmin=self.z_min, vmax=self.z_max)

        self.fig = plt.figure(figsize=(11.5, 8.2))
        try:
            self.fig.canvas.manager.set_window_title(
                "16x16 tactile sensor - live 3D"
            )
        except AttributeError:
            pass

        self.axis3d = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.02, right=0.90, top=0.92, bottom=0.10)

        self.bar_collection = None
        if self.style == "bars":
            faces, colors = self.build_bar_geometry(np.zeros((ROWS, COLS)))
            self.bar_collection = Poly3DCollection(
                faces,
                facecolors=colors,
                edgecolors=(0.10, 0.10, 0.10, 0.40),
                linewidths=0.16,
            )
            self.axis3d.add_collection3d(self.bar_collection)

        self.axis3d.set_xlim(0, COLS)
        self.axis3d.set_ylim(ROWS, 0)
        self.axis3d.set_zlim(self.z_min, self.z_max)
        self.axis3d.set_xticks(np.arange(COLS) + 0.5)
        self.axis3d.set_yticks(np.arange(ROWS) + 0.5)
        self.axis3d.set_xticklabels(
            [f"C{index + 1}" for index in range(COLS)], fontsize=6
        )
        self.axis3d.set_yticklabels(
            [f"R{index + 1}" for index in range(ROWS)], fontsize=6
        )
        self.axis3d.set_xlabel("Column")
        self.axis3d.set_ylabel("Row")
        self.axis3d.set_zlabel(self.axis_label)
        self.axis3d.tick_params(axis="x", labelrotation=45)
        self.axis3d.view_init(elev=self.elev, azim=self.azim)
        self.axis3d.set_box_aspect((16, 16, 7.5))

        scalar_mappable = matplotlib.cm.ScalarMappable(
            norm=self.norm, cmap=self.cmap
        )
        scalar_mappable.set_array([])
        colorbar = self.fig.colorbar(
            scalar_mappable, ax=self.axis3d, shrink=0.62, pad=0.07
        )
        colorbar.set_label(self.label)
        self.colorbar = colorbar

        self.title = self.axis3d.set_title("waiting for frames ...", fontsize=10)
        self.status = self.fig.text(
            0.5,
            0.025,
            self.status_text(),
            ha="center",
            va="center",
            fontsize=8.5,
            color="#404040",
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def status_text(self):
        geometry = (
            f"{1000.0 * sum(self.geometry_seconds) / len(self.geometry_seconds):.1f}"
            if self.geometry_seconds
            else "--"
        )
        redraw = (
            f"{len(self.redraw_seconds) / sum(self.redraw_seconds):.1f}"
            if self.redraw_seconds
            else "--"
        )
        return (
            f"style={self.style}  mode={self.args.mode}  gain={self.gain:.2f}  "
            f"z=+/-{self.z_max:.2f}  geom={geometry}ms  redraw={redraw}fps   |   "
            "b baseline   r rotate   t style   0 view   +/- gain   ][ range   "
            "s snapshot   q quit"
        )

    # ---- interaction ----

    def on_key_press(self, event):
        key = event.key
        if key == "b":
            self.rearm_baseline()
        elif key == "r":
            self.auto_rotate = not self.auto_rotate
        elif key == "t":
            self.set_style("surface" if self.style == "bars" else "bars")
        elif key == "0":
            self.elev, self.azim = self.args.elev, self.args.azim
            self.axis3d.view_init(elev=self.elev, azim=self.azim)
        elif key in ("+", "="):
            self.gain = min(self.gain * 1.25, 200.0)
        elif key == "-":
            self.gain = max(self.gain / 1.25, 0.01)
        elif key == "]":
            self.set_range(self.z_max * 1.25)
        elif key == "[":
            self.set_range(max(self.z_max / 1.25, 1e-3))
        elif key == "s":
            self.save_snapshot()
        elif key in ("q", "escape"):
            self.plt.close(self.fig)
            return
        self.status.set_text(self.status_text())
        self.fig.canvas.draw_idle()

    def set_style(self, style):
        if style == self.style:
            return
        self.style = style
        if self.bar_collection is not None:
            self.bar_collection.remove()
            self.bar_collection = None
        if self.surface_artist is not None:
            self.surface_artist.remove()
            self.surface_artist = None
        if style == "bars":
            faces, colors = self.build_bar_geometry(np.zeros((ROWS, COLS)))
            self.bar_collection = self.Poly3DCollection(
                faces,
                facecolors=colors,
                edgecolors=(0.10, 0.10, 0.10, 0.40),
                linewidths=0.16,
            )
            self.axis3d.add_collection3d(self.bar_collection)

    def set_range(self, limit):
        if self.args.mode == "change":
            self.z_min, self.z_max = -limit, limit
        else:
            self.z_min, self.z_max = 0.0, limit
        self.norm.vmin, self.norm.vmax = self.z_min, self.z_max
        self.axis3d.set_zlim(self.z_min, self.z_max)
        self.colorbar.update_normal(self.colorbar.mappable)

    def save_snapshot(self):
        self.args.output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.args.output / f"sensor_3d_{stamp}.png"
        self.fig.savefig(path, dpi=140)
        print(f"Snapshot: {path}", flush=True)

    def on_close(self, _event):
        self.closed = True

    # ---- update ----

    def update(self, sample):
        frame_index, _millis, raw_grid = sample
        grid = self.orient(raw_grid)
        matrix, ready = self.to_display(grid)

        now = time.perf_counter()
        self.frames_since_tick += 1
        if now - self.fps_t0 >= 1.0:
            self.fps = self.frames_since_tick / (now - self.fps_t0)
            self.frames_since_tick = 0
            self.fps_t0 = now

        render_start = time.perf_counter()
        if self.style == "bars":
            faces, colors = self.build_bar_geometry(matrix)
            if not ready:
                colors[:] = (0.55, 0.55, 0.55, 0.45)
            self.bar_collection.set_verts(faces)
            self.bar_collection.set_facecolor(colors)
        else:
            self.draw_surface(matrix, ready)

        if self.auto_rotate:
            self.azim = (self.azim + self.args.rotate_speed / 15.0) % 360.0
            self.axis3d.view_init(elev=self.elev, azim=self.azim)

        if not ready:
            collected = len(self.baseline_samples)
            self.title.set_text(
                f"collecting no-load baseline {collected}/"
                f"{self.args.baseline_frames} - keep the sensor unloaded"
            )
        else:
            peak_row, peak_column = np.unravel_index(
                int(np.argmax(matrix)), matrix.shape
            )
            unit = "" if self.args.mode == "raw" else " V"
            self.title.set_text(
                f"frame {frame_index}  |  {self.fps:.1f} fps in  |  "
                f"{matrix.min():+.3f} to {matrix.max():+.3f}{unit}\n"
                f"peak @ (R{peak_row + 1}, C{peak_column + 1})"
            )
        self.geometry_seconds.append(
            max(time.perf_counter() - render_start, 1e-6)
        )
        self.status.set_text(self.status_text())

    def draw_surface(self, matrix, ready):
        if self.surface_artist is not None:
            self.surface_artist.remove()
        clipped = np.clip(matrix, self.z_min, self.z_max)
        rows, columns = np.mgrid[0:ROWS, 0:COLS]
        facecolors = (
            np.full((ROWS, COLS, 4), 0.55)
            if not ready
            else self.cmap(self.norm(clipped))
        )
        self.surface_artist = self.axis3d.plot_surface(
            columns + 0.5,
            rows + 0.5,
            clipped,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )

    def run(self):
        self.plt.ion()
        self.plt.show()
        minimum_interval = 1.0 / self.args.max_render_fps
        try:
            while not self.closed and self.plt.fignum_exists(self.fig.number):
                sample = self.source.read_latest()
                now = time.perf_counter()
                # Drain the serial buffer every pass, but redraw at most
                # max_render_fps: 3D redraws are far slower than 24 Hz.
                if sample is not None and now - self.last_render >= minimum_interval:
                    if self.last_render:
                        self.redraw_seconds.append(now - self.last_render)
                    self.last_render = now
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
    results = {}
    for style in ("bars", "surface"):
        args.style = style
        source = SimulatedSource(
            24.0,
            180.0,
            60.0,
            paced=False,
            warmup_s=args.baseline_frames / 24.0,
        )
        viewer = LiveSensor3D(args, source)
        rendered = 0
        for _ in range(args.baseline_frames + 60):
            sample = source.read_latest()
            if sample is None:
                continue
            viewer.update(sample)
            rendered += 1

        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"sensor_3d_selftest_{style}.png"
        start = time.perf_counter()
        viewer.fig.savefig(path, dpi=110)
        save_seconds = time.perf_counter() - start
        geometry_ms = (
            1000.0 * sum(viewer.geometry_seconds) / len(viewer.geometry_seconds)
        )
        results[style] = (rendered, geometry_ms, save_seconds, path)
        viewer.plt.close(viewer.fig)

    print(f"Baseline ready: True")
    for style, (rendered, geometry_ms, save_seconds, path) in results.items():
        print(
            f"{style:8s} frames={rendered:3d}  geometry={geometry_ms:6.2f} ms/frame"
            f"  full_draw={save_seconds * 1000:7.1f} ms"
            f"  -> max ~{1.0 / save_seconds:4.1f} fps"
        )
        print(f"         {path}")
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

    LiveSensor3D(args, source).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
