# -*- coding: utf-8 -*-
# Velostat pressure matrix live heatmap: high voltage (no pressure) = green, low voltage (pressure) = red
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import numpy as np
import time
import threading
from collections import deque

# ===== Serial port / protocol (matches the ESP32 firmware) =====
SERIAL_PORT = None         # None = auto-detect the ESP32 (Silicon Labs CP210x) port; or force e.g. 'COM6'
BAUD_RATE   = 1_000_000
FRAME_W     = 32
FRAME_H     = 32
FRAME_BYTES = FRAME_W * FRAME_H
VREF        = 3.3
READ_TIMEOUT_S = 0.2

# ===== Top-left ROI =====
ROI_Column = 4
ROI_Row    = 4
ROI_I = 0
ROI_J = 0

# ===== Startup baseline (no pressure) =====
# After startup, sample for BASELINE_S seconds; each point's mean voltage is its "no pressure" baseline = greenest.
# Voltage at or below V_RED (V, absolute) = reddest. Press 'r' at any time to re-zero.
BASELINE_S = 1.5
V_RED      = 0.25

# ===== Handshake pipelining =====
PIPELINE_TARGET = 8
PIPELINE_LOW    = 3


def _open_serial():
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    port = SERIAL_PORT
    if port is None:
        cands = [p for p in ports if (p.vid, p.pid) == (0x10C4, 0xEA60)]   # CP210x on the ESP32 board
        if not cands:
            cands = [p for p in ports if p.vid is not None and "Bluetooth" not in (p.description or "")]
        if cands:
            port = cands[0].device
            if len(cands) > 1:
                print(f"[serial] several USB serial ports found, using {port}; set SERIAL_PORT to override")
    if port is None:
        seen = "\n".join(f"  {p.device}: {p.description}" for p in ports) or "  (none)"
        raise SystemExit(
            "[serial] no USB serial port found. Ports seen:\n" + seen +
            "\nIf Device Manager shows 'CP2104 USB to UART Bridge Controller' with a warning icon, "
            "install the Silicon Labs CP210x driver, then re-plug the ESP32.")
    print(f"[serial] opening {port} @ {BAUD_RATE} baud")
    return serial.Serial(port, BAUD_RATE, timeout=0.001)


ser = _open_serial()


def _read_exact(nbytes, watchdog_s=None):
    buf = bytearray(nbytes)
    mv  = memoryview(buf)
    got = 0
    t0  = time.time()
    while got < nbytes:
        n = ser.readinto(mv[got:])
        if n:
            got += n
            t0 = time.time()
        else:
            time.sleep(0.0005)
        if watchdog_s is not None and (time.time() - t0) > watchdog_s:
            return None
    return buf


def _refill_pipeline(credits):
    need = PIPELINE_TARGET - credits
    if need > 0:
        ser.write(b'w' * need)
        return credits + need
    return credits


latest = {"roi": np.full((ROI_Row, ROI_Column), VREF, dtype=np.float32)}
baseline = {"v": None, "acc": [], "t_end": time.time() + BASELINE_S}


def start_baseline():
    baseline["v"] = None
    baseline["acc"] = []
    baseline["t_end"] = time.time() + BASELINE_S


def reader_thread_fn():
    credits = _refill_pipeline(0)
    while True:
        buf = _read_exact(FRAME_BYTES, watchdog_s=READ_TIMEOUT_S)
        if buf is None:
            ser.reset_input_buffer()
            credits = _refill_pipeline(0)
            continue
        _ = ser.read(1)  # optional '\n'

        mat_u8 = np.frombuffer(buf, dtype=np.uint8).reshape(FRAME_H, FRAME_W)
        mat_v  = mat_u8.astype(np.float32) * (VREF / 255.0)
        roi = mat_v[ROI_I:ROI_I + ROI_Row, ROI_J:ROI_J + ROI_Column].copy()
        latest["roi"] = roi
        if baseline["v"] is None:
            baseline["acc"].append(roi)
            if time.time() >= baseline["t_end"]:
                baseline["v"] = np.mean(baseline["acc"], axis=0)

        credits -= 1
        if credits <= PIPELINE_LOW:
            credits = _refill_pipeline(credits)


threading.Thread(target=reader_thread_fn, daemon=True).start()

# ===== Heatmap =====
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(np.ones((ROI_Row, ROI_Column)), cmap='RdYlGn', vmin=0, vmax=1,
               interpolation='nearest', origin='upper')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(f"1 = baseline / no pressure (green),  0 = {V_RED} V (red)")
ax.set_xticks(range(ROI_Column))
ax.set_yticks(range(ROI_Row))
ax.set_xticklabels([f"C{ROI_J + j}" for j in range(ROI_Column)])
ax.set_yticklabels([f"R{ROI_I + i}" for i in range(ROI_Row)])
ax.set_title(f"Velostat {ROI_Row}x{ROI_Column} heatmap")

texts = [[ax.text(j, i, "", ha="center", va="center", fontsize=10, color="black")
          for j in range(ROI_Column)] for i in range(ROI_Row)]


def update(_):
    base = baseline["v"]
    if base is None:
        ax.set_title("Calibrating baseline... keep sensor unloaded")
        im.set_data(np.ones((ROI_Row, ROI_Column)))
        return [im]
    ax.set_title(f"Velostat {ROI_Row}x{ROI_Column} heatmap  (press 'r' to re-zero)")
    span = np.maximum(base - V_RED, 1e-3)
    level = np.clip((latest["roi"] - V_RED) / span, 0, 1)  # baseline=1 (green), V_RED=0 (red)
    im.set_data(level)
    for i in range(ROI_Row):
        for j in range(ROI_Column):
            texts[i][j].set_text(f"{latest['roi'][i, j]:.2f}")
    return [im]


def on_key(event):
    if event.key == 'r':
        start_baseline()


fig.canvas.mpl_connect('key_press_event', on_key)


ani = animation.FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()
