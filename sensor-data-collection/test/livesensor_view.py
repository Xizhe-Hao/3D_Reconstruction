"""Live viewer for the 16x16 tactile sensor (MatrixArrayDual4067Binary firmware).

Connects to the Arduino Nano over serial, performs the M16 handshake
(M16_READY -> M16_START -> M16_ACK), keeps the heartbeat alive with
M16_PING, parses the binary "M16B" frames, and shows the 16x16 ADC map
as a live heatmap. No cameras or force gauge needed.

Usage:
    python live_sensor_view.py COM7          # replace COM7 with your port
    python live_sensor_view.py COM7 --raw    # also print per-frame stats

Dependencies:
    pip install pyserial numpy matplotlib

Press Ctrl+C (or close the window) to stop; M16_STOP is sent on exit.
This is a standalone diagnostic tool: it does not modify or replace
anything in the sensor-data-collection pipeline.
"""

import argparse
import sys
import time

import numpy as np
import serial

# ---- Protocol constants (must match the firmware) ----
BAUD_RATE = 500000
MAGIC = b"M16B"
ROWS = 16
COLS = 16
PAYLOAD_BYTES = ROWS * COLS
# magic(4) + version(1) + rows(1) + cols(1) + type(1) + frame_idx(4) + millis(4)
HEADER_AFTER_MAGIC = 1 + 1 + 1 + 1 + 4 + 4
FRAME_BYTES = 4 + HEADER_AFTER_MAGIC + PAYLOAD_BYTES + 2  # = 274
PING_INTERVAL_S = 0.5  # firmware heartbeat timeout is 2 s


def parse_frames(buffer: bytearray):
    """Yield (frame_index, millis, 16x16 uint8 array) for each valid frame.

    Consumes parsed/invalid bytes from `buffer` in place.
    """
    while True:
        start = buffer.find(MAGIC)
        if start < 0:
            # Keep the last 3 bytes in case a magic is split across reads.
            if len(buffer) > 3:
                del buffer[:-3]
            return
        if start > 0:
            del buffer[:start]
        if len(buffer) < FRAME_BYTES:
            return

        frame = bytes(buffer[:FRAME_BYTES])
        body = frame[4:-2]  # everything after magic, before checksum
        version, rows, cols, ptype = body[0], body[1], body[2], body[3]
        frame_idx = int.from_bytes(body[4:8], "little")
        millis = int.from_bytes(body[8:12], "little")
        payload = body[12:]
        checksum_expected = int.from_bytes(frame[-2:], "little")
        checksum_actual = sum(body) & 0xFFFF

        if (
            version == 1
            and rows == ROWS
            and cols == COLS
            and ptype == 1
            and checksum_actual == checksum_expected
        ):
            del buffer[:FRAME_BYTES]
            grid = np.frombuffer(payload, dtype=np.uint8).reshape(ROWS, COLS)
            yield frame_idx, millis, grid
        else:
            # Corrupt frame: drop this magic and rescan.
            del buffer[:4]


def main():
    ap = argparse.ArgumentParser(description="Live 16x16 tactile sensor viewer")
    ap.add_argument("port", help="Serial port, e.g. COM7 or /dev/ttyUSB0")
    ap.add_argument("--raw", action="store_true", help="print per-frame stats")
    ap.add_argument("--vmax", type=int, default=255, help="heatmap max (default 255)")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    ser = serial.Serial(args.port, BAUD_RATE, timeout=0.05)
    time.sleep(2.0)  # Nano auto-resets on port open; wait for reboot
    ser.reset_input_buffer()

    # ---- Handshake: wait for M16_READY, then START ----
    print("Waiting for M16_READY ...")
    deadline = time.time() + 10
    ready = False
    while time.time() < deadline:
        line = ser.readline()
        if b"M16_READY" in line:
            ready = True
            break
    if not ready:
        print("No M16_READY received. Check port, baud (500000), and firmware.")
        ser.close()
        sys.exit(1)

    ser.write(b"M16_START\n")
    print("Sent M16_START, streaming... (Ctrl+C or close window to stop)")

    # ---- Live plot ----
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("16x16 tactile sensor - live")
    image = ax.imshow(
        np.zeros((ROWS, COLS), dtype=np.uint8),
        cmap="viridis",
        vmin=0,
        vmax=args.vmax,
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label="ADC (8-bit)")
    title = ax.set_title("waiting for frames ...")

    buffer = bytearray()
    last_ping = time.time()
    last_idx = None
    frames = 0
    fps_t0 = time.time()
    fps = 0.0

    try:
        while plt.fignum_exists(fig.number):
            now = time.time()
            if now - last_ping >= PING_INTERVAL_S:
                ser.write(b"M16_PING\n")
                last_ping = now

            chunk = ser.read(4096)
            if chunk:
                buffer.extend(chunk)

            newest = None
            for frame_idx, millis, grid in parse_frames(buffer):
                if frame_idx == last_idx:
                    continue  # firmware sends each frame twice; keep first copy
                last_idx = frame_idx
                newest = (frame_idx, millis, grid)
                frames += 1
                if args.raw:
                    r, c = np.unravel_index(int(grid.argmax()), grid.shape)
                    print(
                        f"frame {frame_idx:6d}  t={millis} ms  "
                        f"min={grid.min():3d} max={grid.max():3d} "
                        f"mean={grid.mean():6.1f}  peak@(row {r}, col {c})"
                    )

            if newest is not None:
                frame_idx, millis, grid = newest
                if now - fps_t0 >= 1.0:
                    fps = frames / (now - fps_t0)
                    frames = 0
                    fps_t0 = now
                image.set_data(grid)
                title.set_text(
                    f"frame {frame_idx}   max={grid.max()}   {fps:.1f} fps"
                )
                fig.canvas.draw_idle()

            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ser.write(b"M16_STOP\n")
            ser.flush()
        except Exception:
            pass
        ser.close()
        print("Stopped, M16_STOP sent.")


if __name__ == "__main__":
    main()