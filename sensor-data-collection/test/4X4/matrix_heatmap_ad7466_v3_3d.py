# -*- coding: utf-8 -*-
# Velostat pressure matrix live 3D bar chart v3
# One 3D bar per sensing point: no pressure = tallest and greenest; pressed = shorter and redder
#
#   python matrix_heatmap_ad7466_v3_3d.py                       # port found by itself (CP2104)
#   python matrix_heatmap_ad7466_v3_3d.py --port COM4
#   python matrix_heatmap_ad7466_v3_3d.py --record C:/tactile_data/force_cal_4x4
#       also writes <root>/capture_<stamp>/sensor_raw.csv + session.json, the same
#       session layout as live_sensor_2d.py --record, so align_force_presses.py,
#       bench_quicklook.py and bench_report.py read it directly.
#
# The serial protocol and the recorder live in ../live_sensor_2d.py (Esp32Source,
# SessionRecorder); this file only draws.
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_sensor_2d import ESP32_DEFAULT_ROI, Esp32Source, SessionRecorder  # noqa: E402

VREF = 3.3

# ===== Baseline and range =====
BASELINE_S = 1.5    # startup zeroing duration (keep unloaded); press 'r' to re-zero
V_RED      = 0.25   # voltage (V) at or below this = shortest, reddest bar

# ===== Appearance =====
EASE       = 0.35   # display smoothing factor 0-1, smaller = softer
MIN_HEIGHT = 0.04   # minimum bar height (so bars never vanish completely)
BAR_SIZE   = 0.72   # bar footprint as a fraction of its cell
SPIN       = False  # True: slowly rotate the view
VIEW_ELEV  = 55     # elevation angle: larger = more top-down (90 = straight down), smaller = more side-on
VIEW_AZIM  = -50    # horizontal rotation angle
# Switch view while running: 1=steep top-down  2=default oblique  3=low side view  4=straight top-down
VIEW_PRESETS = {'1': (65, -50), '2': (55, -50), '3': (18, -50), '4': (90, -90)}

BG    = "#0a0e17"
CYAN  = "#4de8ff"
DIM   = "#5c6b86"
GREEN = "#2dff8a"
RED   = "#ff3b4e"
AMBER = "#ffd23f"


def parse_args():
    p = argparse.ArgumentParser(description="Live 3D bars of the 4x4 ESP32 Velostat board.")
    p.add_argument("--port", default=None, help="COM port; default: the CP2104 found on USB")
    p.add_argument("--roi", type=int, nargs=4, metavar=("ROW0", "COL0", "ROWS", "COLS"),
                   default=list(ESP32_DEFAULT_ROI), help="wired pads inside the 32x32 frame")
    p.add_argument("--record", type=Path, metavar="ROOT",
                   help="also save every frame to ROOT/capture_<stamp>/sensor_raw.csv (+ session.json)")
    p.add_argument("--baseline-s", type=float, default=BASELINE_S)
    return p.parse_args()


args = parse_args()
ROI_I, ROI_J, ROI_Row, ROI_Column = args.roi

try:
    source = Esp32Source(args.port, roi=tuple(args.roi))
except Exception as exc:
    raise SystemExit(f"[serial] {exc}")

recorder = None
if args.record:
    recorder = SessionRecorder(args.record, source.shape, source.meta(),
                               tool="matrix_heatmap_ad7466_v3_3d.py --record")

latest = {"roi": np.full((ROI_Row, ROI_Column), VREF, dtype=np.float32)}
baseline = {"v": None, "acc": [], "t_end": time.time() + args.baseline_s}


def start_baseline():
    baseline["v"] = None
    baseline["acc"] = []
    baseline["t_end"] = time.time() + args.baseline_s


def on_frame(host_ns, frame_index, millis, grid):
    """Runs on the reader thread for every frame: keep the newest, build the baseline, record."""
    roi = grid.astype(np.float32) * (VREF / 255.0)
    latest["roi"] = roi
    if baseline["v"] is None:
        baseline["acc"].append(roi)
        if time.time() >= baseline["t_end"]:
            baseline["v"] = np.mean(baseline["acc"], axis=0)
    if recorder is not None:
        recorder(host_ns, frame_index, millis, grid)


source.on_frame = on_frame
source.start()

# ===== Display =====
cmap = LinearSegmentedColormap.from_list(
    "pressure", ["#ff1f3d", "#ff7a2f", "#ffd23f", "#7dff6a", "#18e07a"])

plt.rcParams["font.family"] = ["Consolas", "DejaVu Sans Mono"]
fig = plt.figure(figsize=(8, 7), facecolor=BG)
try:
    fig.canvas.manager.set_window_title("Velostat Tactile Matrix 3D")
except Exception:
    pass

ax = fig.add_axes([0.02, 0.04, 0.96, 0.84], projection='3d', facecolor=BG)
ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
ax.set_box_aspect((1, 1, 0.6))

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((0.06, 0.09, 0.15, 1.0))
    axis._axinfo["grid"]["color"] = (0.30, 0.91, 1.0, 0.12)
    axis._axinfo["axisline"]["color"] = CYAN
ax.set_xlim(-0.5, ROI_Column - 0.5)
ax.set_ylim(-0.5, ROI_Row - 0.5)
ax.set_zlim(0, 1.0)
ax.invert_yaxis()
ax.set_xticks(range(ROI_Column))
ax.set_yticks(range(ROI_Row))
ax.set_xticklabels([f"C{ROI_J + j}" for j in range(ROI_Column)])
ax.set_yticklabels([f"R{ROI_I + i}" for i in range(ROI_Row)])
ax.set_zticks([0, 0.5, 1.0])
ax.set_zticklabels(["MAX", "", "NONE"])
ax.tick_params(colors=DIM, labelsize=8)

fig.text(0.04, 0.94, "VELOSTAT TACTILE MATRIX  3D", color=CYAN, fontsize=17,
         fontweight="bold", ha="left", va="center")
fig.text(0.04, 0.905, f"{ROI_Row}x{ROI_Column} SENSOR ARRAY  //  BAR DROPS UNDER PRESSURE"
         + ("  //  RECORDING" if recorder else ""),
         color=DIM, fontsize=9, ha="left", va="center")
status_txt = fig.text(0.96, 0.94, "", color=CYAN, fontsize=10, fontweight="bold",
                      ha="right", va="center")
peak_txt = fig.text(0.04, 0.03, "", color=CYAN, fontsize=11, fontweight="bold",
                    ha="left", va="center")
fig.text(0.96, 0.03, "[R] RE-ZERO   [1-4] VIEW", color=DIM, fontsize=9, ha="right", va="center")

xs, ys = np.meshgrid(np.arange(ROI_Column), np.arange(ROI_Row))
xs = xs.ravel() - BAR_SIZE / 2
ys = ys.ravel() - BAR_SIZE / 2

shown = np.ones((ROI_Row, ROI_Column), dtype=np.float32)
state = {"blink": 0, "bars": None}


def draw_bars(level):
    if state["bars"] is not None:
        state["bars"].remove()
    h = np.maximum(level.ravel(), MIN_HEIGHT)
    colors = cmap(level.ravel())
    state["bars"] = ax.bar3d(xs, ys, np.zeros_like(h), BAR_SIZE, BAR_SIZE, h,
                             color=colors, shade=True, edgecolor=(0, 0, 0, 0.35),
                             linewidth=0.4)


def update(_):
    global shown
    state["blink"] += 1
    if SPIN:
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM + state["blink"] * 0.4)

    base = baseline["v"]
    if base is None:
        dots = "." * (1 + (state["blink"] // 8) % 3)
        status_txt.set_text(f"CALIBRATING{dots}")
        status_txt.set_color(AMBER)
        peak_txt.set_text("KEEP SENSOR UNLOADED")
        peak_txt.set_color(AMBER)
        shown = np.ones_like(shown)
        draw_bars(shown)
        return []

    span = np.maximum(base - V_RED, 1e-3)
    level = np.clip((latest["roi"] - V_RED) / span, 0, 1)   # 1=no pressure (tall/green) 0=full pressure (short/red)
    shown = shown + EASE * (level - shown)
    draw_bars(shown)

    load = float((1.0 - shown).max()) * 100.0
    status_txt.set_text("BASELINE LOCKED" + (f"  //  REC {recorder.frames}" if recorder else ""))
    status_txt.set_color(GREEN)
    peak_txt.set_text(f"PEAK LOAD  {load:5.1f} %")
    peak_txt.set_color(RED if load > 66 else (AMBER if load > 25 else GREEN))
    return []


def on_key(event):
    if event.key == 'r':
        start_baseline()
    elif event.key in VIEW_PRESETS:
        ax.view_init(elev=VIEW_PRESETS[event.key][0], azim=VIEW_PRESETS[event.key][1])


fig.canvas.mpl_connect('key_press_event', on_key)
ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
try:
    plt.show()
finally:
    source.close()
    if recorder is not None:
        recorder.close()
