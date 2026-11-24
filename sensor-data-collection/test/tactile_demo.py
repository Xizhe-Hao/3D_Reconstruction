#!/usr/bin/env python3
"""Show-and-tell view of the 4x4 tactile sensor: 16 pads that light up where you press.

    python sensor-data-collection/test/tactile_demo.py              # board found automatically
    python sensor-data-collection/test/tactile_demo.py --port COM4
    python sensor-data-collection/test/tactile_demo.py --simulate   # no board: press pads with the mouse

Plug the board in, start the script, keep hands off for two seconds while it
zeroes, then press. Each pad's colour and percentage follow how hard it is
pressed, with the raw ADC reading in small print underneath (it falls as the
pad is pressed); the panel on the right names the pad, grades the press and keeps a
10 s trace. The window survives the cable being unplugged and reconnects.

Keys
    Space / R   re-zero (nothing touching the sensor)
    F           full screen
    Up / Down   more / less sensitive
    H / V / T   mirror left-right / mirror up-down / swap rows and columns,
                if the picture does not match the way the sensor lies on the table
                (start with --mirror-h / --mirror-v / --transpose to keep it)
    Q / Esc     quit
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from live_sensor_2d import BOARDS, ESP32_DEFAULT_ROI, Esp32Source  # noqa: E402

BG, PANEL, INK, MUTED, EDGE = "#0f1419", "#161d26", "#e6edf3", "#8b98a5", "#2b3846"
GOOD, WARN = "#3fb950", "#d29922"
PAD_COLOURS = ["#1b2633", "#1d6fa5", "#22b8cf", "#ffd43b", "#ff6b4a"]  # idle -> light -> firm
HISTORY_S = 10.0
EASE_S = 0.07  # the picture glides to each new frame over about this long (the board sends ~9 fps)


class SimSource:
    """Stands in for the board: 12-bit readings that fall where `target` (0..1 per pad) is set."""

    adc_max = 4095.0
    error = None

    def __init__(self, shape, fps=10.0):
        self.shape = shape
        self.period = 1.0 / fps
        self.target = np.zeros(shape)
        self.level = np.zeros(shape)
        self.rng = np.random.default_rng(0)
        self.index, self.t0, self.next = 0, time.perf_counter(), 0.0

    def start(self):
        pass

    def read_latest(self):
        now = time.perf_counter() - self.t0
        if now < self.next:
            return None
        self.next = now + self.period
        self.level += 0.45 * (self.target - self.level)
        grid = 2200.0 - 1700.0 * self.level + self.rng.normal(0, 2.0, self.shape)
        self.index += 1
        return self.index, int(now * 1000), grid.clip(0, 4095).astype(np.uint16)

    def close(self):
        pass


class Link:
    """Keeps a source open in the background: connects, and reconnects after the cable is pulled."""

    def __init__(self, make):
        self.make = make
        self.source = None
        self.generation = 0  # goes up on every (re)connect, so the view knows to zero again
        self.problem = ""
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            if self.source is None:
                try:
                    source = self.make()
                    source.start()
                    self.source, self.problem = source, ""
                    self.generation += 1
                except Exception as exc:  # no port, port busy, board still booting
                    self.problem = str(exc).splitlines()[0]
                    self.stop.wait(2.0)
            elif self.source.error is not None:
                dead, self.source = self.source, None
                self.problem = str(dead.error).splitlines()[0]
                dead.close()
                self.stop.wait(1.0)
            else:
                self.stop.wait(0.3)

    def read(self):
        source = self.source
        if source is None or source.error is not None:
            return None
        return source.read_latest()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4.0)
        if self.source is not None:
            self.source.close()


class Demo:
    def __init__(self, args, link, shape, simulated):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        self.plt, self.args, self.link, self.simulated = plt, args, link, simulated
        self.rows, self.cols = self.shape = shape
        self.lowers = BOARDS["esp32"]["pressure_lowers_adc"]
        self.cmap = LinearSegmentedColormap.from_list("touch", PAD_COLOURS)
        self.full = float(args.full)
        self.mirror_h, self.mirror_v = args.mirror_h, args.mirror_v
        self.transpose = args.transpose and self.rows == self.cols
        self.generation = -1
        self.target = np.zeros(shape)  # 0..1 per pad from the newest frame, sensor layout
        self.shown = np.zeros(shape)   # what is drawn; eases towards target
        self.active = np.zeros(shape, bool)
        self.raw = None  # newest frame as the board sent it (ADC counts), sensor layout
        self.frame_times = deque(maxlen=30)
        self.history = deque()
        self.last = time.perf_counter()
        self.note, self.note_until = "", 0.0
        self.rezero()
        self._remap()
        self._build()

    # ------------------------------------------------------------ signal

    def rezero(self):
        self.base, self.samples, self.t_base = None, [], None
        self.recent = deque(maxlen=3)
        self.target[:] = 0
        self.active[:] = False

    def ingest(self, grid, now):
        source = self.link.source
        scale = (source.adc_max if source is not None else 4095.0) / 255.0
        g = grid.astype(float) / scale  # 8-bit-equivalent counts, like the calibration tools
        if g.shape != self.shape:
            return
        self.raw = np.asarray(grid).astype(int)
        if self.base is None:
            if self.t_base is None:
                self.t_base = now
            self.samples.append(g)
            if now - self.t_base >= self.args.baseline_s and len(self.samples) >= 8:
                stack = np.stack(self.samples)
                self.base = np.median(stack, axis=0)
                # a noisy pad (loose wire, long lead) gets its own, higher bar: 6 sigma from the MAD
                sigma = 1.4826 * np.median(np.abs(stack - self.base), axis=0)
                self.on = np.maximum(self.args.threshold, 6.0 * sigma)
            return
        self.recent.append(g)
        g = np.median(np.stack(self.recent), axis=0)  # drops single-frame glitches, costs ~0.1 s
        delta = (self.base - g) if self.lowers else (g - self.base)
        on, off = self.on, 0.6 * self.on
        self.active = (delta > on) | (self.active & (delta > off))
        # Untouched pads follow slow drift; a pad reading "less than nothing" means the zero was
        # taken while it was pressed, so that one catches up quickly.
        idle = ~self.active
        rate = np.where(delta < -on, 0.25, 0.02)
        self.base[idle] += (rate * (g - self.base))[idle]
        level = np.clip((delta - off) / np.maximum(self.full - off, 1.0), 0.0, 1.0) ** 0.6
        self.target = np.where(self.active, np.maximum(level, 0.06), 0.0)

    # ------------------------------------------------------------ layout

    def _remap(self):
        """idx[r, c] = which sensor pad (flat index) is drawn at row r, column c of the picture."""
        idx = np.arange(self.rows * self.cols).reshape(self.shape)
        if self.transpose:
            idx = idx.T
        if self.mirror_h:
            idx = idx[:, ::-1]
        if self.mirror_v:
            idx = idx[::-1]
        self.idx = np.ascontiguousarray(idx)
        for r in range(self.rows):
            for c in range(self.cols):
                if hasattr(self, "names"):
                    self.names[r][c].set_text(self.pad_code(self.idx[r, c]))

    def pad_code(self, flat):
        r, c = divmod(int(flat), self.cols)
        return f"R{r + 1}C{c + 1}"

    def pad_name(self, flat):
        r, c = divmod(int(flat), self.cols)
        return f"Row {r + 1} · Column {c + 1}"

    def _build(self):
        plt = self.plt
        from matplotlib.patches import FancyBboxPatch, Rectangle
        from matplotlib.widgets import Button

        fig = self.fig = plt.figure(figsize=(13, 7.5), facecolor=BG)
        try:
            fig.canvas.manager.set_window_title("Tactile Sensor Demo")
        except AttributeError:
            pass
        fig.text(0.04, 0.925, "Tactile Sensor", color=INK, fontsize=24, fontweight="bold", va="center")
        fig.text(0.04, 0.875, f"{self.rows} × {self.cols} pressure-sensing array  ·  {self.rows * self.cols} pads"
                 "  ·  press anywhere and watch it light up", color=MUTED, fontsize=11, va="center")
        self.conn = fig.text(0.96, 0.925, "", color=MUTED, fontsize=11, ha="right", va="center")

        ax = self.ax = fig.add_axes([0.04, 0.09, 0.50, 0.75], facecolor=BG)
        ax.set_xlim(-0.56, self.cols - 0.44)
        ax.set_ylim(self.rows - 0.44, -0.56)
        ax.set_aspect("equal", anchor="W")
        ax.axis("off")
        self.glows, self.pads, self.values, self.names, self.raws = [], [], [], [], []
        for r in range(self.rows):
            row = ([], [], [], [], [])
            for c in range(self.cols):
                glow = FancyBboxPatch((c - 0.49, r - 0.49), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.16",
                                      fc=PAD_COLOURS[-1], ec="none", alpha=0.0, zorder=1)
                pad = FancyBboxPatch((c - 0.42, r - 0.42), 0.84, 0.84, boxstyle="round,pad=0,rounding_size=0.12",
                                     fc=PAD_COLOURS[0], ec=EDGE, lw=1.5, zorder=2)
                ax.add_patch(glow)
                ax.add_patch(pad)
                value = ax.text(c, r + 0.02, "", color=INK, fontsize=20, fontweight="bold",
                                ha="center", va="center", zorder=3)
                name = ax.text(c - 0.35, r - 0.33, "", color=MUTED, fontsize=8, ha="left", va="center", zorder=3)
                raw = ax.text(c, r + 0.32, "", color=MUTED, fontsize=8, ha="center", va="center", zorder=3)
                for store, item in zip(row, (glow, pad, value, name, raw)):
                    store.append(item)
            for store, items in zip((self.glows, self.pads, self.values, self.names, self.raws), row):
                store.append(items)
        self._remap()

        panel = self.panel = fig.add_axes([0.60, 0.44, 0.36, 0.40], facecolor=BG)
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)
        panel.axis("off")
        self.status = panel.text(0, 0.98, "", color=INK, fontsize=25, fontweight="bold", va="top")
        self.sub = panel.text(0, 0.70, "", color=MUTED, fontsize=14, va="top")
        panel.text(0, 0.36, "PRESSURE", color=MUTED, fontsize=9, va="center")
        self.percent = panel.text(1, 0.36, "", color=INK, fontsize=11, ha="right", va="center")
        panel.add_patch(Rectangle((0, 0.16), 1, 0.12, fc=PANEL, ec=EDGE, lw=1))
        self.bar = Rectangle((0, 0.16), 0, 0.12, fc=PAD_COLOURS[1], ec="none")
        panel.add_patch(self.bar)
        for x, word, ha in ((0, "light", "left"), (0.5, "medium", "center"), (1, "firm", "right")):
            panel.text(x, 0.07, word, color=MUTED, fontsize=9, ha=ha, va="center")

        trace = self.trace = fig.add_axes([0.60, 0.19, 0.36, 0.18], facecolor=PANEL)
        trace.set_xlim(-HISTORY_S, 0)
        trace.set_ylim(0, 1.08)
        trace.set_xticks([])
        trace.set_yticks([])
        for s in trace.spines.values():
            s.set_color(EDGE)
        trace.set_title("LAST 10 SECONDS", color=MUTED, fontsize=9, loc="left", pad=6)
        (self.line,) = trace.plot([], [], color=PAD_COLOURS[2], lw=2)

        self.button = Button(fig.add_axes([0.60, 0.075, 0.13, 0.06]), "Re-zero", color=PANEL, hovercolor=EDGE)
        self.button.label.set_color(INK)
        self.button.label.set_fontsize(11)
        for s in self.button.ax.spines.values():
            s.set_color(EDGE)
        self.button.on_clicked(lambda _e: self.rezero())
        self.hint = fig.text(0.745, 0.105, "", color=MUTED, fontsize=9, va="center")
        fig.text(0.04, 0.04, "Space re-zero   ·   F full screen   ·   Up/Down sensitivity   ·   H / V / T flip the view"
                 "   ·   Q quit", color=MUTED, fontsize=9, va="center")

        fig.canvas.mpl_connect("key_press_event", self.on_key)
        if self.simulated:
            for event in ("button_press_event", "motion_notify_event", "button_release_event"):
                fig.canvas.mpl_connect(event, self.on_mouse)

    # ------------------------------------------------------------ input

    def say(self, text, seconds=2.5):
        self.note, self.note_until = text, time.perf_counter() + seconds

    def on_key(self, event):
        key = (event.key or "").lower()
        if key in (" ", "r", "b"):
            self.rezero()
        elif key == "f":
            try:
                self.fig.canvas.manager.full_screen_toggle()
            except Exception:
                pass
        elif key in ("up", "down"):
            self.full = float(np.clip(self.full * (0.8 if key == "up" else 1.25), 15.0, 200.0))
            self.say(f"full scale = {self.full:.0f} counts")
        elif key in ("h", "v", "t"):
            if key == "h":
                self.mirror_h = not self.mirror_h
            elif key == "v":
                self.mirror_v = not self.mirror_v
            elif self.rows == self.cols:
                self.transpose = not self.transpose
            self._remap()
            flags = [f for f, on in (("--transpose", self.transpose), ("--mirror-h", self.mirror_h),
                                     ("--mirror-v", self.mirror_v)) if on]
            self.say("view: " + (" ".join(flags) or "as wired"), 4.0)
        elif key in ("q", "escape"):
            self.plt.close(self.fig)

    def on_mouse(self, event):
        source = self.link.source
        if not isinstance(source, SimSource):
            return
        held = event.name != "button_release_event" and event.button == 1 and event.inaxes is self.ax
        blob = np.zeros(self.shape)
        if held and event.xdata is not None:
            rr, cc = np.mgrid[0:self.rows, 0:self.cols]
            blob = np.exp(-((rr - event.ydata) ** 2 + (cc - event.xdata) ** 2) / (2 * 0.42 ** 2))
        target = np.zeros(self.rows * self.cols)
        target[self.idx.ravel()] = blob.ravel()  # picture position -> sensor pad
        source.target = target.reshape(self.shape)

    # ------------------------------------------------------------ drawing

    def update(self, _frame=None):
        now = time.perf_counter()
        dt, self.last = now - self.last, now
        if self.link.generation != self.generation:
            self.generation = self.link.generation
            self.frame_times.clear()
            self.rezero()
        sample = self.link.read()
        if sample is not None:
            self.frame_times.append(now)
            self.ingest(sample[2], now)
        connected = self.link.source is not None
        stale = connected and (not self.frame_times or now - self.frame_times[-1] > 2.0)
        if not connected or stale:
            self.target[:] = 0
            self.active[:] = False
        self.shown += (self.target - self.shown) * (1.0 - math.exp(-dt / EASE_S))

        view = self.shown.ravel()[self.idx]
        lit = self.active.ravel()[self.idx]
        raw = self.raw.ravel()[self.idx] if self.raw is not None and connected and not stale else None
        for r in range(self.rows):
            for c in range(self.cols):
                v = float(view[r, c])
                self.pads[r][c].set_facecolor(self.cmap(v))
                self.pads[r][c].set_edgecolor(self.cmap(v) if v > 0.02 else EDGE)
                self.glows[r][c].set_facecolor(self.cmap(v))
                self.glows[r][c].set_alpha(0.35 * v)
                self.values[r][c].set_text(f"{v * 100:.0f}" if v >= 0.05 else "")
                self.values[r][c].set_color(BG if v > 0.55 else INK)
                self.names[r][c].set_color(BG if v > 0.3 else MUTED)
                self.raws[r][c].set_text("" if raw is None else f"ADC {raw[r, c]}")
                self.raws[r][c].set_color(BG if v > 0.3 else MUTED)
        n = int(lit.sum())
        peak = float(view.max())

        if not connected:
            head, sub, colour = "Sensor not connected", "Plug in the USB cable. Retrying…", WARN
            self.conn.set_text("offline")
            self.conn.set_color(WARN)
        elif stale:
            head, sub, colour = "Waiting for the sensor…", "Connected, no data yet.", WARN
        elif self.base is None:
            head, sub, colour = "Getting ready…", "Keep hands off the sensor for a moment.", WARN
        elif n == 0:
            head, sub, colour = "Touch the sensor", "Each square is one sensing pad.", INK
        else:
            strongest = self.idx.ravel()[int(np.argmax(view))]
            grade = "Light touch" if peak < 0.34 else "Medium press" if peak < 0.67 else "Firm press"
            if n == 1:
                head, sub = self.pad_name(strongest), grade
            else:
                head, sub = f"{n} pads pressed", f"{grade}, strongest at {self.pad_name(strongest).lower()}"
            colour = self.cmap(max(peak, 0.45))
        self.status.set_text(head)
        self.status.set_color(colour)
        self.sub.set_text(sub)
        if connected:
            fps = 0.0
            if len(self.frame_times) >= 2 and not stale:
                fps = (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])
            self.conn.set_text(("simulated  ·  press pads with the mouse" if self.simulated
                                else f"live  ·  {self.link.source.port_name}") + f"  ·  {fps:.0f} fps")
            self.conn.set_color(WARN if stale else GOOD)
        self.bar.set_width(peak)
        self.bar.set_facecolor(self.cmap(max(peak, 0.3)))
        self.percent.set_text(f"{peak * 100:.0f} %" if n else "")

        self.history.append((now, peak))
        while self.history and now - self.history[0][0] > HISTORY_S:
            self.history.popleft()
        h = np.array(self.history)
        self.line.set_data(h[:, 0] - now, h[:, 1])
        self.hint.set_text(self.note if now < self.note_until else ("" if connected else self.link.problem[:70]))

    def run(self):
        from matplotlib.animation import FuncAnimation

        self._anim = FuncAnimation(self.fig, self.update, interval=33, cache_frame_data=False)
        self.plt.show()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=None, help="serial port of the ESP32 board (default: found automatically)")
    p.add_argument("--simulate", action="store_true", help="no board: press the pads with the mouse")
    p.add_argument("--roi", type=int, nargs=4, metavar=("ROW0", "COL0", "ROWS", "COLS"), default=None,
                   help="window of the 32x32 frame holding the wired pads (default %s)" % (ESP32_DEFAULT_ROI,))
    p.add_argument("--bits", type=int, choices=(8, 12), default=None, help="skip the 8/12-bit firmware check")
    p.add_argument("--threshold", type=float, default=8.0,
                   help="a pad counts as touched above this change, in 8-bit-equivalent counts (default 8; noisy pads get more)")
    p.add_argument("--full", type=float, default=70.0,
                   help="change that shows as 100 %% (default 70 counts; Up/Down adjust it live)")
    p.add_argument("--baseline-s", type=float, default=1.5, help="seconds of no-touch readings used as zero")
    p.add_argument("--mirror-h", action="store_true", help="mirror the picture left-right")
    p.add_argument("--mirror-v", action="store_true", help="mirror the picture up-down")
    p.add_argument("--transpose", action="store_true", help="swap rows and columns in the picture")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--selftest", type=Path, default=None, metavar="PNG", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib

    if args.selftest:
        matplotlib.use("Agg")
        args.simulate = True
    import matplotlib.pyplot as plt

    for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[key] = []  # our keys only; matplotlib's own S/F/Q/L shortcuts would fight them
    plt.rcParams["toolbar"] = "None"
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

    roi = tuple(args.roi) if args.roi else ESP32_DEFAULT_ROI
    shape = (roi[2], roi[3])
    if args.simulate:
        link = Link(lambda: SimSource(shape))
    else:
        link = Link(lambda: Esp32Source(args.port, roi=roi, bits=args.bits))
    demo = Demo(args, link, shape, args.simulate)
    try:
        if args.selftest:
            t_end = time.perf_counter() + args.baseline_s + 3.0
            while time.perf_counter() < t_end:
                if demo.base is not None and link.source is not None:
                    press = np.zeros(shape)
                    press[1, 2], press[1, 1] = 0.9, 0.25
                    link.source.target = press
                demo.update()
                time.sleep(0.03)
            assert demo.base is not None and demo.active[1, 2] and not demo.active[3, 0], "self-test failed"
            demo.fig.savefig(args.selftest, dpi=110, facecolor=BG)
            print(f"self-test ok; wrote {args.selftest}")
            return 0
        if args.fullscreen:
            try:
                demo.fig.canvas.manager.full_screen_toggle()
            except Exception:
                pass
        demo.run()
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
