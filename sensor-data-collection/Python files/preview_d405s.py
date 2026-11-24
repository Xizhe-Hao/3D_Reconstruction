#!/usr/bin/env python3
"""Live view of every connected RealSense (D405): colour next to depth, one row per camera.

    python preview_d405s.py

For aiming the cameras. Each row is labelled with the camera's serial number;
the cross marks the image centre and the number next to it is the distance
there. Depth is coloured from near (red) to far (blue) over --near-mm..--far-mm;
black is "no depth" (too close, too far, or hidden from one of the two lenses).
The D405 measures from about 70 mm; 150-300 mm is a good working distance.

Keys (window focused):  s  save a snapshot of the view    q / Esc  quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
    import pyrealsense2 as rs
except ImportError as exc:
    sys.exit(f"{exc}; activate the capture environment (.venv)")

WINDOW = "D405 live view  -  s: snapshot   q: quit"
USB2_MODES = [(480, 270, 15), (424, 240, 15), (640, 480, 15), (480, 270, 6), (424, 240, 6)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="848x480", help="colour and depth size (they must be equal on a D405)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--near-mm", type=float, default=100.0)
    parser.add_argument("--far-mm", type=float, default=400.0)
    parser.add_argument("--snapshot-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))

    devices = []
    for _ in range(10):  # tolerate a camera that is still settling after being plugged in
        try:
            devices = list(rs.context().query_devices())
            serials = sorted(d.get_info(rs.camera_info.serial_number) for d in devices)
            break
        except RuntimeError as exc:
            print(f"USB error while listing cameras ({str(exc).splitlines()[0][:70]}); retrying", flush=True)
            time.sleep(1.0)
    if not devices:
        sys.exit("no RealSense camera found")

    def start(camera) -> bool:
        """(Re)start one camera; False while it is unplugged, busy or on a USB 2 link."""
        try:  # a camera in the middle of being replugged makes the enumeration itself fail
            found = [d for d in rs.context().query_devices()
                     if d.get_info(rs.camera_info.serial_number) == camera["serial"]]
            usb = found[0].get_info(rs.camera_info.usb_type_descriptor) if found else ""
        except RuntimeError as exc:
            camera["problem"] = f"USB error while looking for the camera: {str(exc).splitlines()[0][:70]}"
            return False
        if not found:
            camera["problem"] = "not connected - check the cable"
            return False
        # A half-seated plug or a USB 2 port/cable gives a USB 2 link. The full-size streams do not
        # fit it, but a small one does, and that is enough for aiming the camera.
        modes = USB2_MODES if usb.startswith("2") else [(width, height, args.fps)]
        for w, h, fps in modes:
            config = rs.config()
            config.enable_device(camera["serial"])
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            pipeline = rs.pipeline()
            try:
                profile = pipeline.start(config)
            except RuntimeError as exc:
                camera["problem"] = f"USB {usb}: {str(exc).splitlines()[0]}"
                continue
            camera["scale_mm"] = profile.get_device().first_depth_sensor().get_depth_scale() * 1000.0
            camera["usb"], camera["mode"] = usb, (w, h, fps)
            camera["pipeline"], camera["times"], camera["last_frame"] = pipeline, [], time.perf_counter()
            print(f"{camera['serial']}: streaming {w}x{h} @ {fps} fps, USB {usb}", flush=True)
            return True
        return False

    def lost(camera, why) -> None:
        """A cable pulled while aiming must not close the window: show it and keep retrying."""
        print(f"{camera['serial']}: lost ({why}); retrying", flush=True)
        if camera["pipeline"] is not None:
            try:
                camera["pipeline"].stop()
            except RuntimeError:
                pass
        camera["pipeline"], camera["retry_at"] = None, time.perf_counter() + 2.0
        row = np.zeros((height, 2 * width, 3), np.uint8)
        camera["row"] = row
        camera["why"] = why

    cameras = []
    for serial in serials:
        camera = {"serial": serial, "pipeline": None, "scale_mm": 0.1, "usb": "?", "times": [],
                  "retry_at": 0.0, "last_frame": 0.0, "row": np.zeros((height, 2 * width, 3), np.uint8)}
        if not start(camera):
            print(f"{serial}: could not start ({camera.get('problem')})", flush=True)
            lost(camera, "did not start")
        cameras.append(camera)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(2 * width, 1500), min(len(cameras) * height, 900))
    try:
        while True:
            for camera in cameras:
                if camera["pipeline"] is None:
                    if time.perf_counter() >= camera["retry_at"] and not start(camera):
                        camera["retry_at"] = time.perf_counter() + 2.0
                    if camera["pipeline"] is None:
                        row = np.zeros((height, 2 * width, 3), np.uint8)
                        cv2.putText(row, f"D405 {camera['serial']}", (20, height // 2 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                        cv2.putText(row, camera.get("problem") or camera.get("why", ""), (20, height // 2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
                        camera["row"] = row
                    continue
                try:
                    ok, frames = camera["pipeline"].try_wait_for_frames(30)
                except RuntimeError as exc:
                    lost(camera, str(exc).splitlines()[0])
                    continue
                if not ok:
                    if time.perf_counter() - camera["last_frame"] > 3.0:
                        lost(camera, "no frames for 3 s")
                    continue
                camera["last_frame"] = time.perf_counter()
                depth_frame, colour_frame = frames.get_depth_frame(), frames.get_color_frame()
                if not depth_frame or not colour_frame:
                    continue
                now = time.perf_counter()
                camera["times"] = [t for t in camera["times"] if now - t < 2.0] + [now]
                depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.float32) * camera["scale_mm"]
                colour = np.asanyarray(colour_frame.get_data()).copy()
                valid = depth_mm > 0
                level = np.clip((args.far_mm - depth_mm) / (args.far_mm - args.near_mm), 0.0, 1.0)
                depth_view = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_JET)
                depth_view[~valid] = 0

                cy, cx = depth_mm.shape[0] // 2, depth_mm.shape[1] // 2
                patch = depth_mm[cy - 6:cy + 7, cx - 6:cx + 7]
                patch = patch[patch > 0]
                centre = f"{np.median(patch):.0f} mm" if patch.size else "no depth"
                for view in (colour, depth_view):
                    cv2.drawMarker(view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 28, 2)
                    cv2.putText(view, centre, (cx + 18, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(view, centre, (cx + 18, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                fps = (len(camera["times"]) - 1) / max(camera["times"][-1] - camera["times"][0], 1e-6)
                label = (f"D405 {camera['serial']}   USB {camera['usb']}   {fps:.0f} fps   "
                         f"depth valid {100 * valid.mean():.0f}%")
                cv2.putText(colour, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(colour, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                row = np.hstack([colour, depth_view])
                if row.shape[:2] != (height, 2 * width):
                    row = cv2.resize(row, (2 * width, height), interpolation=cv2.INTER_LINEAR)
                    note = (f"USB {camera['usb']} link: low-resolution preview "
                            f"({camera['mode'][0]}x{camera['mode'][1]}). OK for aiming, NOT for recording.")
                    cv2.putText(row, note, (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                    cv2.putText(row, note, (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                camera["row"] = row

            view = np.vstack([camera["row"] for camera in cameras])
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                path = args.snapshot_dir / time.strftime("d405_view_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(str(path), view)
                print(f"saved {path}")
    finally:
        for camera in cameras:
            if camera["pipeline"] is not None:
                try:
                    camera["pipeline"].stop()
                except RuntimeError:
                    pass
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
