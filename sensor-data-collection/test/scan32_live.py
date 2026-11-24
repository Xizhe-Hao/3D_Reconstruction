"""Live view of the full 32x32 frame from the ESP32 + AD7466 board, for finding which pins a sensor is on.

Every cell is shown as "counts below the open-channel level" (the frame median,
about 2326 with nothing connected): an unconnected channel is 0, a connected
unloaded pad is 10-50, a pressed pad is several hundred up to ~1350.

Left: the current frame. Right: the largest value each cell has reached since
the last reset, so pressing every pad once leaves the whole footprint visible.

    python sensor-data-collection/test/scan32_live.py            # port found automatically
    python sensor-data-collection/test/scan32_live.py --port COM4

Keys: r = reset the peak-hold panel, q = quit. Hover a cell to read its raw ADC value.
"""
import argparse
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.colors import PowerNorm
from serial.tools import list_ports

N = 32
ACTIVE_COUNTS = 10  # a row/column is listed once any of its cells has been this far below open level


def find_port():
    for p in list_ports.comports():
        if p.vid == 0x10C4 and p.pid == 0xEA60:  # CP2104 on the ESP32 Feather
            return p.device
    raise SystemExit("no CP210x port found; give --port")


class Reader(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.ser = serial.Serial(port, baud, timeout=0.05)
        time.sleep(0.5)
        self.ser.reset_input_buffer()
        self.frame, self.count, self.error = None, 0, None
        self.stop = threading.Event()
        self.size = len(self._request())  # 1025 = 8-bit firmware, 2049 = 12-bit
        if self.size == 2 * N * N + 1:
            self.dtype, self.scale = "<u2", 1.0
        elif self.size == N * N + 1:
            self.dtype, self.scale = np.uint8, 16.0  # show 8-bit frames on the 12-bit scale
        else:
            raise SystemExit(f"unexpected reply to 'w': {self.size} bytes (expected 1025 or 2049)")

    def _request(self):
        self.ser.write(b"w")
        got, quiet, t_end = bytearray(), None, time.time() + 2.0
        while time.time() < t_end and len(got) < 2 * N * N + 1:
            chunk = self.ser.read(4096)
            if chunk:
                got.extend(chunk)
                quiet = time.time()
            elif quiet and time.time() - quiet > 0.3:
                break
        return bytes(got)

    def run(self):
        try:
            while not self.stop.is_set():
                b = self._request()
                if len(b) == self.size:
                    self.frame = np.frombuffer(b[:-1], dtype=self.dtype).reshape(N, N).astype(float) * self.scale
                    self.count += 1
        except Exception as exc:  # port unplugged etc.
            self.error = exc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--vmax", type=float, default=1400, help="top of the colour scale in 12-bit counts")
    args = ap.parse_args()

    reader = Reader(args.port or find_port(), args.baud)
    reader.start()

    ink, muted, grid = "#1f2328", "#59636e", "#d0d7de"
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), facecolor="white")
    fig.subplots_adjust(left=0.045, right=0.93, top=0.88, bottom=0.12, wspace=0.12)
    norm = PowerNorm(gamma=0.4, vmin=0, vmax=args.vmax)  # stretches the low end so unloaded pads (10-50) show
    state = {"raw": np.zeros((N, N)), "peak": np.zeros((N, N))}
    images = []
    for ax, title in zip(axes, ("Now", "Peak hold  (press r to reset)")):
        im = ax.imshow(np.zeros((N, N)), cmap="Blues", norm=norm, interpolation="nearest")
        images.append(im)
        ax.set_title(title, color=ink, fontsize=12, loc="left")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.tick_params(labelsize=7, colors=muted, length=0, labeltop=True, labelright=True)
        ax.set_xticks(np.arange(-0.5, N), minor=True)
        ax.set_yticks(np.arange(-0.5, N), minor=True)
        ax.grid(which="minor", color=grid, linewidth=0.4)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel("frame column", color=muted, fontsize=9)
        ax.set_ylabel("frame row", color=muted, fontsize=9)
        for s in ax.spines.values():
            s.set_color(grid)

        def fmt(x, y, _ax=ax):
            r, c = int(round(y)), int(round(x))
            if 0 <= r < N and 0 <= c < N:
                return f"row {r}  col {c}   raw ADC {state['raw'][r, c]:.0f}   peak below open {state['peak'][r, c]:.0f}"
            return ""

        ax.format_coord = fmt
    cax = fig.add_axes([0.945, 0.2, 0.012, 0.6])
    cb = fig.colorbar(images[1], cax=cax, ticks=[0, 25, 100, 300, 600, 1000, args.vmax])
    cb.set_label("counts below open-channel level (12-bit)", color=muted, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=muted)
    fig.suptitle("ESP32 32x32 pin scan", color=ink, fontsize=14, x=0.045, ha="left")
    status = fig.text(0.045, 0.035, "", color=ink, fontsize=11, family="monospace", va="bottom")

    def on_key(event):
        if event.key == "r":
            state["peak"][:] = 0
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        if reader.error:
            status.set_text(f"serial error: {reader.error}")
            return
        f = reader.frame
        if f is None:
            return
        below = np.median(f) - f  # 0 for an open channel, positive for a connected or pressed pad
        state["raw"] = f
        state["peak"] = np.maximum(state["peak"], below)
        images[0].set_data(np.clip(below, 0, None))
        images[1].set_data(state["peak"])
        rows = np.where(state["peak"].max(1) > ACTIVE_COUNTS)[0]
        cols = np.where(state["peak"].max(0) > ACTIVE_COUNTS)[0]
        r, c = np.unravel_index(below.argmax(), below.shape)
        status.set_text(
            f"rows seen: {' '.join(map(str, rows)) or '-'}     cols seen: {' '.join(map(str, cols)) or '-'}\n"
            f"strongest now: row {r} col {c} = {below[r, c]:.0f}     open level {np.median(f):.0f}     frame {reader.count}"
        )

    anim = FuncAnimation(fig, update, interval=60, cache_frame_data=False)  # noqa: F841 (keep a reference)
    try:
        plt.show()
    finally:
        reader.stop.set()
        reader.join(timeout=1.0)
        reader.ser.close()


if __name__ == "__main__":
    main()
