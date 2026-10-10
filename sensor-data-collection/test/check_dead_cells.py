#!/usr/bin/env python3
"""Dead / weak cell check for the 16x16 tactile sensor (M16B firmware).

Two phases, no cameras or force gauge:

  1. Baseline: keep the sensor unloaded for a few seconds. Per-cell no-load
     level and noise are recorded.
  2. Sweep: press over the whole surface (finger, eraser, or a roller) until
     every cell on the right-hand coverage map has turned coloured. Cells that
     stay grey never responded. Press f to finish.

The report classifies every cell and is saved as PNG + JSON:

  dead      never moved more than the response threshold
  inverted  moved, but only opposite to the array's press direction
  weak      responded, but far below the array's typical response
  noisy     no-load noise far above the array's typical noise
  offset    no-load level far from the rest of the array

A row or column where most cells are dead points at the mux channel, the
trace, or the connector rather than the sensing material.

Cell indices are firmware indices (R = row mux channel + 1, C = column mux
channel + 1), not rotated or flipped, so they map directly onto the wiring.

Usage:
    python check_dead_cells.py COM7
    python check_dead_cells.py COM7 --baseline-seconds 5 --min-counts 6
    python check_dead_cells.py --simulate     # synthetic sensor with faults
    python check_dead_cells.py --selftest     # headless, verifies detection
    python check_dead_cells.py --list-ports

Keys while the window is focused:
    f        finish the sweep and write the report
    r        clear the sweep (keep the baseline)
    b        re-capture the baseline, then sweep again
    q / Esc  quit (writes the report if a sweep was recorded)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_sensor_2d import (  # noqa: E402
    BAUD_RATE,
    COLS,
    ROWS,
    SerialSource,
    list_serial_ports,
)

LINE_FAULT_FRACTION = 0.75
CATEGORY_CODES = {"ok": 0, "weak": 1, "noisy": 2, "offset": 3, "inverted": 4, "dead": 5}


class FaultySimulatedSource:
    """Synthetic sensor with known faults; a press rasters over every cell."""

    DEAD_CELLS = [(3, 5), (10, 1)]
    DEAD_COLUMN = 12
    WEAK_CELL = (7, 7)
    SHORTED_CELL = (14, 9)  # stuck near full scale
    NOISY_CELL = (0, 15)

    def __init__(self, fps=60.0, paced=True, warmup_s=2.0):
        self.period = 1.0 / fps
        self.paced = paced
        self.warmup_s = warmup_s
        self.start = time.perf_counter()
        self.last_emit = 0.0
        self.frame_index = 0
        self.random = np.random.default_rng(3)
        self.rows, self.columns = np.indices((ROWS, COLS))
        self.offset = 6.0 + self.random.normal(0.0, 1.0, (ROWS, COLS))
        self.gain = 120.0 * self.random.uniform(0.7, 1.3, (ROWS, COLS))
        for cell in self.DEAD_CELLS:
            self.gain[cell] = 0.0
        self.gain[:, self.DEAD_COLUMN] = 0.0
        self.gain[self.WEAK_CELL] *= 0.08

    def read_latest(self):
        now = time.perf_counter()
        if self.paced and now - self.last_emit < self.period:
            return None
        self.last_emit = now
        elapsed = (
            now - self.start if self.paced else self.frame_index * self.period
        )

        values = self.offset + self.random.normal(0.0, 0.7, (ROWS, COLS))
        active = elapsed - self.warmup_s
        if active > 0.0:
            # Boustrophedon raster, 0.4 s per cell row, so a full pass takes
            # about 6.4 s and every cell gets pressed.
            row_f = (active / 0.4) % ROWS
            row = int(row_f)
            frac = row_f - row
            column = frac * (COLS - 1) if row % 2 == 0 else (1 - frac) * (COLS - 1)
            distance_squared = (self.rows - row) ** 2 + (self.columns - column) ** 2
            values += self.gain * np.exp(-distance_squared / (2.0 * 1.0**2))
        values[self.NOISY_CELL] += self.random.normal(0.0, 12.0)
        values[self.SHORTED_CELL] = 252.0 + self.random.normal(0.0, 0.5)

        self.frame_index += 1
        grid = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        return self.frame_index, int(elapsed * 1000), grid

    def close(self):
        print("Simulated stream stopped.")


class CellCheck:
    """Accumulates baseline and sweep statistics and classifies the cells."""

    def __init__(self, args):
        self.args = args
        self.reset_baseline()

    def reset_baseline(self):
        self.baseline_frames = []
        self.baseline_mean = None
        self.baseline_std = None
        self.reset_sweep()

    def reset_sweep(self):
        self.sweep_frames = 0
        self.max_up = np.zeros((ROWS, COLS))
        self.max_down = np.zeros((ROWS, COLS))

    @property
    def baseline_ready(self):
        return self.baseline_mean is not None

    def add(self, grid):
        """Feed one frame. Returns the signed change from baseline, or None."""
        values = grid.astype(np.float64)
        if not self.baseline_ready:
            self.baseline_frames.append(values)
            if len(self.baseline_frames) >= self.args.baseline_count:
                stack = np.stack(self.baseline_frames)
                self.baseline_mean = stack.mean(axis=0)
                self.baseline_std = stack.std(axis=0)
                self.baseline_frames = []
            return None
        delta = values - self.baseline_mean
        np.maximum(self.max_up, delta, out=self.max_up)
        np.maximum(self.max_down, -delta, out=self.max_down)
        self.sweep_frames += 1
        return delta

    def threshold(self):
        """Per-cell response threshold in ADC counts."""
        return np.maximum(
            self.args.min_counts, self.args.noise_sigmas * self.baseline_std
        )

    def press_sign(self):
        """+1 if pressing raises the ADC on this sensor, -1 if it lowers it."""
        return 1.0 if np.median(self.max_up) >= np.median(self.max_down) else -1.0

    def response(self):
        """Per-cell peak response in the press direction (counts, >= 0)."""
        return self.max_up if self.press_sign() > 0 else self.max_down

    def coverage(self):
        return self.response() >= self.threshold()

    def classify(self):
        threshold = self.threshold()
        sign = self.press_sign()
        response = self.max_up if sign > 0 else self.max_down
        reverse = self.max_down if sign > 0 else self.max_up

        responsive = response >= threshold
        inverted = ~responsive & (reverse >= threshold)
        dead = ~responsive & ~inverted

        typical_response = (
            float(np.median(response[responsive])) if responsive.any() else 0.0
        )
        weak = responsive & (response < self.args.weak_fraction * typical_response)

        noise = self.baseline_std
        typical_noise = float(np.median(noise))
        noisy = noise > max(self.args.noise_factor * typical_noise, 3.0)

        level = self.baseline_mean
        typical_level = float(np.median(level))
        mad = float(np.median(np.abs(level - typical_level))) * 1.4826
        offset = np.abs(level - typical_level) > max(
            self.args.offset_sigmas * mad, self.args.offset_counts
        )

        # One category per cell, most severe first.
        category = np.full((ROWS, COLS), "ok", dtype=object)
        for name, mask in (
            ("offset", offset),
            ("noisy", noisy),
            ("weak", weak),
            ("inverted", inverted),
            ("dead", dead),
        ):
            category[mask] = name

        dead_rows = [
            r + 1 for r in range(ROWS) if dead[r].mean() >= LINE_FAULT_FRACTION
        ]
        dead_columns = [
            c + 1 for c in range(COLS) if dead[:, c].mean() >= LINE_FAULT_FRACTION
        ]

        def cells(mask):
            return [
                {
                    "row": int(r) + 1,
                    "col": int(c) + 1,
                    "response": round(float(response[r, c]), 1),
                    "baseline": round(float(level[r, c]), 1),
                    "noise": round(float(noise[r, c]), 2),
                }
                for r, c in zip(*np.nonzero(mask))
            ]

        return {
            "press_direction": "ADC rises" if sign > 0 else "ADC falls",
            "sweep_frames": self.sweep_frames,
            "typical_response_counts": round(typical_response, 1),
            "typical_noise_counts": round(typical_noise, 2),
            "typical_baseline_counts": round(typical_level, 1),
            "counts": {
                name: int((category == name).sum()) for name in CATEGORY_CODES
            },
            "dead_rows": dead_rows,
            "dead_columns": dead_columns,
            "cells": {
                name: cells(category == name)
                for name in CATEGORY_CODES
                if name != "ok"
            },
            "_category": category,
            "_response": response,
        }


def print_report(report):
    counts = report["counts"]
    total = ROWS * COLS
    print()
    print("=" * 60)
    print(f"Cell check: {counts['ok']}/{total} OK")
    print(
        f"press direction: {report['press_direction']}  |  "
        f"typical response {report['typical_response_counts']} counts  |  "
        f"noise {report['typical_noise_counts']} counts"
    )
    for name in ("dead", "inverted", "weak", "noisy", "offset"):
        listed = report["cells"][name]
        if not listed:
            continue
        print(f"\n{name.upper()} ({len(listed)}):")
        for cell in listed:
            print(
                f"  R{cell['row']:<2d} C{cell['col']:<2d}  "
                f"response {cell['response']:6.1f}  "
                f"baseline {cell['baseline']:6.1f}  noise {cell['noise']:5.2f}"
            )
    if report["dead_rows"]:
        print(f"\nWhole ROW(S) dead: {report['dead_rows']}  -> check row mux / trace")
    if report["dead_columns"]:
        print(
            f"Whole COLUMN(S) dead: {report['dead_columns']}  "
            "-> check column mux / trace"
        )
    print("=" * 60)


def render_report(report, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    category = report["_category"]
    codes = np.vectorize(CATEGORY_CODES.get)(category)
    colors = ["#e8f5e9", "#ffd54f", "#ba68c8", "#4fc3f7", "#ff8a65", "#424242"]

    fig, (ax_map, ax_resp) = plt.subplots(1, 2, figsize=(12, 5.6))
    ax_map.imshow(
        codes, cmap=ListedColormap(colors), vmin=-0.5, vmax=len(colors) - 0.5
    )
    for r in range(ROWS):
        for c in range(COLS):
            name = category[r, c]
            if name != "ok":
                ax_map.text(
                    c, r, name[0].upper(), ha="center", va="center",
                    color="white" if name == "dead" else "black", fontsize=8,
                )
    counts = report["counts"]
    ax_map.set_title(
        f"Cell status: {counts['ok']}/{ROWS * COLS} OK\n"
        + "  ".join(f"{k} {v}" for k, v in counts.items() if k != "ok" and v)
    )
    response = report["_response"]
    im = ax_resp.imshow(response, cmap="viridis")
    ax_resp.set_title(
        f"Peak response during sweep (ADC counts, {report['press_direction']})"
    )
    fig.colorbar(im, ax=ax_resp, fraction=0.046)
    for ax in (ax_map, ax_resp):
        ax.set_xticks(range(COLS), [str(c + 1) for c in range(COLS)], fontsize=7)
        ax.set_yticks(range(ROWS), [str(r + 1) for r in range(ROWS)], fontsize=7)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
    legend = [
        plt.Rectangle((0, 0), 1, 1, color=colors[code]) for code in CATEGORY_CODES.values()
    ]
    ax_map.legend(
        legend, list(CATEGORY_CODES), loc="upper center",
        bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=8, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png = output_dir / f"cell_check_{stamp}.png"
    js = output_dir / f"cell_check_{stamp}.json"
    render_report(report, png)
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    js.write_text(json.dumps(public, indent=2), encoding="utf-8")
    print(f"Report image: {png}")
    print(f"Report data:  {js}")
    return png, js


class CheckWindow:
    """Live view: change from baseline (left), sweep coverage (right)."""

    def __init__(self, args, source):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.args = args
        self.source = source
        self.check = CellCheck(args)
        self.finished = False
        self.last_coverage_draw = 0.0

        self.fig, (self.ax_live, self.ax_cov) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.manager.set_window_title("Tactile sensor cell check")
        self.live_image = self.ax_live.imshow(
            np.zeros((ROWS, COLS)), cmap="RdBu_r", vmin=-40, vmax=40
        )
        self.fig.colorbar(self.live_image, ax=self.ax_live, fraction=0.046,
                          label="ADC change from baseline")
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#9e9e9e")
        self.cov_image = self.ax_cov.imshow(
            np.ma.masked_all((ROWS, COLS)), cmap=cmap, vmin=0, vmax=60
        )
        self.fig.colorbar(self.cov_image, ax=self.ax_cov, fraction=0.046,
                          label="peak response (counts); grey = not yet")
        for ax in (self.ax_live, self.ax_cov):
            ax.set_xticks(range(COLS), [str(c + 1) for c in range(COLS)], fontsize=7)
            ax.set_yticks(range(ROWS), [str(r + 1) for r in range(ROWS)], fontsize=7)
        self.status = self.fig.text(
            0.5, 0.02, "f finish   r clear sweep   b re-baseline   q quit",
            ha="center", fontsize=9,
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.tight_layout(rect=(0, 0.05, 1, 1))

    def on_key(self, event):
        key = (event.key or "").lower()
        if key == "f":
            self.finished = True
        elif key == "r":
            self.check.reset_sweep()
        elif key == "b":
            self.check.reset_baseline()
        elif key in ("q", "escape"):
            self.plt.close(self.fig)

    def update(self, grid):
        delta = self.check.add(grid)
        if delta is None:
            self.ax_live.set_title(
                f"Baseline {len(self.check.baseline_frames)}/"
                f"{self.args.baseline_count}: DO NOT TOUCH the sensor"
            )
            return
        self.live_image.set_data(delta)

        now = time.perf_counter()
        if now - self.last_coverage_draw < 0.2:
            return
        self.last_coverage_draw = now
        response = self.check.response()
        covered = self.check.coverage()
        self.cov_image.set_data(np.ma.masked_where(~covered, response))
        if covered.any():
            self.cov_image.set_clim(0, max(float(np.percentile(response[covered], 95)), 1.0))
        self.ax_live.set_title(
            f"Live change from baseline  |  peak {np.abs(delta).max():.0f} counts"
        )
        self.ax_cov.set_title(
            f"Sweep coverage: {int(covered.sum())}/{ROWS * COLS} cells responded\n"
            "press everywhere until no grey is left, then press f"
        )

    def run(self):
        self.plt.ion()
        self.plt.show()
        try:
            while not self.finished and self.plt.fignum_exists(self.fig.number):
                sample = self.source.read_latest()
                if sample is not None:
                    self.update(sample[2])
                    self.fig.canvas.draw_idle()
                self.plt.pause(0.001)
        except KeyboardInterrupt:
            print()
        finally:
            self.source.close()
        if self.check.baseline_ready and self.check.sweep_frames:
            report = self.check.classify()
            print_report(report)
            save_report(report, self.args.output)
        else:
            print("No sweep recorded; no report written.")


def run_selftest(args):
    source = FaultySimulatedSource(paced=False, warmup_s=args.baseline_count / 60.0)
    check = CellCheck(args)
    # Baseline plus two full raster passes.
    for _ in range(args.baseline_count + int(2 * 6.4 * 60)):
        check.add(source.read_latest()[2])
    report = check.classify()
    print_report(report)
    save_report(report, args.output)

    category = report["_category"]
    expected = {cell: "dead" for cell in source.DEAD_CELLS}
    expected.update({(r, source.DEAD_COLUMN): "dead" for r in range(ROWS)})
    expected[source.WEAK_CELL] = "weak"
    expected[source.NOISY_CELL] = "noisy"
    failures = [
        f"R{r + 1} C{c + 1}: expected {want}, got {category[r, c]}"
        for (r, c), want in expected.items()
        if category[r, c] != want
    ]
    if category[source.SHORTED_CELL] not in ("dead", "offset"):
        failures.append(f"shorted cell classified {category[source.SHORTED_CELL]}")
    if report["dead_columns"] != [source.DEAD_COLUMN + 1]:
        failures.append(f"dead columns {report['dead_columns']}")
    flagged = {(r, c) for r, c in zip(*np.nonzero(category != "ok"))}
    extra = flagged - set(expected) - {source.SHORTED_CELL}
    if extra:
        failures.append(f"false positives: {sorted(extra)}")
    if failures:
        print("SELFTEST FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("SELFTEST PASSED: all injected faults found, no false positives.")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Find dead / weak cells on the 16x16 tactile sensor."
    )
    parser.add_argument("port", nargs="?", help="Serial port, e.g. COM7.")
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--simulate", action="store_true",
                        help="Synthetic sensor with injected faults.")
    parser.add_argument("--selftest", action="store_true",
                        help="Headless check that the injected faults are found.")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--baseline-seconds", type=float, default=3.0,
                        help="Approximate no-load capture time.")
    parser.add_argument("--baseline-fps", type=float, default=30.0,
                        help="Expected frame rate, used to size the baseline.")
    parser.add_argument("--min-counts", type=float, default=8.0,
                        help="A cell must change by at least this many ADC "
                             "counts to count as responding.")
    parser.add_argument("--noise-sigmas", type=float, default=6.0,
                        help="...and by at least this many no-load std devs.")
    parser.add_argument("--weak-fraction", type=float, default=0.25,
                        help="Responding cells below this fraction of the "
                             "median response are flagged weak.")
    parser.add_argument("--noise-factor", type=float, default=5.0,
                        help="Noise above this multiple of the median is noisy.")
    parser.add_argument("--offset-sigmas", type=float, default=8.0)
    parser.add_argument("--offset-counts", type=float, default=30.0,
                        help="Minimum baseline deviation to flag an offset.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "cell_check",
                        help="Directory for the PNG / JSON report.")
    parser.add_argument("--handshake-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not (args.list_ports or args.simulate or args.selftest or args.port):
        parser.error("a serial port is required unless --simulate is used")
    args.baseline_count = max(10, int(round(args.baseline_seconds * args.baseline_fps)))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list_ports:
        list_serial_ports()
        return 0
    if args.selftest:
        import matplotlib

        matplotlib.use("Agg")
        return run_selftest(args)

    import matplotlib

    matplotlib.use("TkAgg")
    if args.simulate:
        source = FaultySimulatedSource()
    else:
        try:
            source = SerialSource(args.port, args.baud, args.handshake_timeout)
        except Exception as exc:
            print(f"Cannot start the sensor stream: {exc}", file=sys.stderr)
            return 1
    CheckWindow(args, source).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
