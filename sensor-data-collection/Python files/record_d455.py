#!/usr/bin/env python3
"""Record the overhead RealSense D455 alongside a Blackfly capture.

The D455 is an independent depth reference for the reconstruction, not a
finer one: at its 52 cm minimum range it resolves ~0.8 mm per pixel on the
sponge and ~1-2 mm in depth, against 0.16 mm for the calibrated Blackfly rig.
It is there to catch systematic errors in the multi-view depth on large
deformations.

Synchronisation is by timestamp, not by trigger: the D455 free-runs at 30 fps
and `align_d455.py` later picks, for every Blackfly frame set, the D455 frame
closest in time (at most half a D455 period, 17 ms, away). On Windows without
the RealSense metadata driver the D455 timestamps are host arrival times
(timestamp domain `system_time`), the same kind of clock the Blackfly capture
logs (`time.time_ns()` at arrival), so the two compare directly up to a
roughly constant latency difference.

Usage, in a second terminal while capture_3blackfly_sensor_force.py runs:

    python record_d455.py --output C:\\tactile_data\\test_press

    python record_d455.py --preview          # aim the camera, no recording

Start this BEFORE the Blackfly capture and stop it (Ctrl+C) AFTER, so the
recording covers every frame set. Output: `d455_<timestamp>.db3` (rosbag2; depth +
colour, lossless, with intrinsics) and `d455_<timestamp>.json` (device,
settings, intrinsics, extrinsics).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 missing:  python -m pip install pyrealsense2 --no-deps")

MIN_RANGE_M = 0.33      # D455 minimum depth at 848x480 with no disparity shift
DISPARITY_LEVELS = 126  # disparity search range of the D4 depth ASIC


def intrinsics_dict(profile) -> dict:
    i = profile.as_video_stream_profile().get_intrinsics()
    return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy,
            "ppx": i.ppx, "ppy": i.ppy, "model": str(i.model), "coeffs": list(i.coeffs)}


def extrinsics_dict(a, b) -> dict:
    e = a.get_extrinsics_to(b)
    return {"rotation_col_major": list(e.rotation), "translation_m": list(e.translation)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="folder for the recording")
    parser.add_argument("--preview", action="store_true",
                        help="show depth and colour live to aim the camera; records nothing")
    parser.add_argument("--depth-size", default="848x480",
                        help="848x480 is the D455's best depth resolution")
    parser.add_argument("--color-size", default="848x480",
                        help="colour is for calibration and checking by eye; 1280x720 "
                             "triples the file size (~100 MB/s at 30 fps)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preset", choices=["default", "high_accuracy", "high_density"],
                        default="default",
                        help="depth visual preset; high_accuracy drops uncertain pixels")
    parser.add_argument("--laser-power", type=float, default=None,
                        help="projector power in mW (0 turns it off); default unchanged")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after N seconds; 0 runs until Ctrl+C")
    parser.add_argument("--near", action="store_true",
                        help="close-range setup for ~40 cm: 1280x720 depth and a "
                             "disparity shift for a 0.33 m minimum range (depth noise "
                             "~0.2 mm at 40 cm in theory, against ~0.3 mm at 848x480)")
    parser.add_argument("--min-range-m", type=float, default=None,
                        help="closest depth to measure; sets the disparity shift. The "
                             "farthest is then about 3x this -- the background drops out")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="stop when this file appears (used by capture_with_d455.py, "
                             "which cannot send Ctrl+C to a child on Windows)")
    parser.add_argument("--genlock", action="store_true",
                        help="hardware-trigger the DEPTH from the Arduino D9 pulse "
                             "(inter_cam_sync_mode 4: one depth frame per pulse). Needs the "
                             "sync cable with a 5 V -> 1.8 V divider; colour keeps free-running")
    parser.add_argument("--exposure-us", type=float, default=None,
                        help="manual depth exposure; default 8000 with --genlock "
                             "(auto exposure is unreliable when triggered), else auto")
    args = parser.parse_args()
    if args.near:
        args.depth_size = "1280x720"
        if args.min_range_m is None:
            args.min_range_m = 0.33
    if args.genlock and args.exposure_us is None:
        args.exposure_us = 8000.0
    # A triggered sensor must be able to run faster than the trigger arrives.
    args.depth_fps = 90 if args.genlock else args.fps
    if not args.preview and args.output is None:
        parser.error("--output is required unless --preview is used")
    return args


def configure(args):
    dw, dh = (int(v) for v in args.depth_size.split("x"))
    cw, ch = (int(v) for v in args.color_size.split("x"))
    config = rs.config()
    config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, args.depth_fps)
    config.enable_stream(rs.stream.color, cw, ch, rs.format.rgb8, args.fps)
    return config


def apply_options(device, args):
    depth_sensor = device.first_depth_sensor()
    for sensor in device.query_sensors():
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1)
    if args.preset != "default" and depth_sensor.supports(rs.option.visual_preset):
        preset = {"high_accuracy": rs.rs400_visual_preset.high_accuracy,
                  "high_density": rs.rs400_visual_preset.high_density}[args.preset]
        depth_sensor.set_option(rs.option.visual_preset, int(preset))
    if args.exposure_us is not None:
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        depth_sensor.set_option(rs.option.exposure, float(args.exposure_us))
    if args.laser_power is not None and depth_sensor.supports(rs.option.laser_power):
        if args.laser_power <= 0 and depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        else:
            depth_sensor.set_option(rs.option.laser_power, float(args.laser_power))
    return depth_sensor.get_depth_scale()


def set_disparity_shift(args) -> dict:
    """Shift the disparity search so the D455 sees closer than its default.

    The depth ASIC searches DISPARITY_LEVELS disparities starting at the shift,
    so the measurable range is f*B/(shift + levels) .. f*B/shift. The setting
    lives in the camera until power-off, so it is always written -- 0 when no
    close range was asked for.
    """
    devices = rs.context().query_devices()
    device = devices[0]
    advanced = rs.rs400_advanced_mode(device)
    if not advanced.is_enabled():
        if args.min_range_m is None:
            return {"disparity_shift": 0}
        sys.exit("advanced mode is off; enable it once with the RealSense Viewer or "
                 "rs.rs400_advanced_mode(dev).toggle_advanced_mode(True), then retry")
    width, height = (int(v) for v in args.depth_size.split("x"))
    sensor = device.first_depth_sensor()
    fx, ir1, ir2 = None, None, None
    for profile in sensor.get_stream_profiles():
        video = profile.as_video_stream_profile()
        if profile.stream_type() == rs.stream.depth and (video.width(), video.height()) == (width, height):
            fx = video.get_intrinsics().fx
        if profile.stream_type() == rs.stream.infrared:
            if profile.stream_index() == 1 and ir1 is None:
                ir1 = profile
            if profile.stream_index() == 2 and ir2 is None:
                ir2 = profile
    baseline = abs(ir1.get_extrinsics_to(ir2).translation[0]) if ir1 and ir2 else 0.095
    shift = 0
    if args.min_range_m is not None and fx:
        shift = max(0, int(np.ceil(fx * baseline / args.min_range_m - DISPARITY_LEVELS)))
    table = advanced.get_depth_table()
    table.disparityShift = shift
    advanced.set_depth_table(table)
    near = fx * baseline / (shift + DISPARITY_LEVELS) if fx else None
    far = fx * baseline / shift if fx and shift else None
    return {"disparity_shift": shift, "min_range_m": near, "max_range_m": far,
            "baseline_m": baseline, "depth_fx": fx}


def set_sync_mode(args):
    """Set (or clear) genlock before streaming starts; the mode persists in the
    camera until power-off, so a plain recording explicitly sets it back to 0."""
    devices = rs.context().query_devices()
    if len(devices) == 0:
        sys.exit("no RealSense device found")
    sensor = devices[0].first_depth_sensor()
    if not sensor.supports(rs.option.inter_cam_sync_mode):
        if args.genlock:
            sys.exit("this camera/firmware does not support inter_cam_sync_mode")
        return
    sensor.set_option(rs.option.inter_cam_sync_mode, 4.0 if args.genlock else 0.0)


def preview(args):
    import cv2

    args.genlock = False                      # preview always free-runs
    args.depth_fps = args.fps
    set_sync_mode(args)
    shift = set_disparity_shift(args)
    near = shift.get("min_range_m") or MIN_RANGE_M
    print(f"depth {args.depth_size}, disparity shift {shift['disparity_shift']}: measures "
          f"{near:.2f}-" + (f"{shift['max_range_m']:.2f} m" if shift.get("max_range_m") else "far"))
    pipeline = rs.pipeline()
    profile = pipeline.start(configure(args))
    scale = apply_options(profile.get_device(), args)
    print("preview: q to quit. Aim so the sponge is in the middle and the centre "
          f"distance reads more than {near + 0.03:.2f} m.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
            color = np.asanyarray(frames.get_color_frame().get_data())
            h, w = depth.shape
            centre = depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
            valid = centre > 0
            distance = float(np.median(centre[valid])) if valid.any() else 0.0
            shown = cv2.applyColorMap(
                np.clip((depth - near) / 0.5 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            shown[depth == 0] = 0
            cv2.rectangle(shown, (w // 3, h // 3), (2 * w // 3, 2 * h // 3), (255, 255, 255), 1)
            warn = "  TOO CLOSE" if 0 < distance < near + 0.03 else ""
            cv2.putText(shown, f"centre {distance:.3f} m  valid {valid.mean():.0%}{warn}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            small = cv2.resize(cv2.cvtColor(color, cv2.COLOR_RGB2BGR), (w, int(w * color.shape[0] / color.shape[1])))
            cv2.imshow("D455 depth | colour", np.vstack([shown, cv2.resize(small, (w, h))]))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        pipeline.stop()


def record(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output.mkdir(parents=True, exist_ok=True)
    bag = args.output / f"d455_{stamp}.db3"
    set_sync_mode(args)
    shift = set_disparity_shift(args)
    config = configure(args)
    config.enable_record_to_file(str(bag))
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    device = profile.get_device()
    scale = apply_options(device, args)
    depth_profile = profile.get_stream(rs.stream.depth)
    color_profile = profile.get_stream(rs.stream.color)
    meta = {
        "device": device.get_info(rs.camera_info.name),
        "serial": device.get_info(rs.camera_info.serial_number),
        "firmware": device.get_info(rs.camera_info.firmware_version),
        "recording": bag.name,
        "started_host_ns": time.time_ns(),
        "fps": args.fps,
        "sync": "genlock" if args.genlock else "timestamp",
        "depth_fps": args.depth_fps,
        "exposure_us": args.exposure_us,
        "depth_size": args.depth_size,
        **{f"depth_{k}": v for k, v in shift.items()},
        "preset": args.preset,
        "laser_power_mw": args.laser_power,
        "depth_scale_m": scale,
        "depth_intrinsics": intrinsics_dict(depth_profile),
        "color_intrinsics": intrinsics_dict(color_profile),
        "depth_to_color": extrinsics_dict(depth_profile, color_profile),
    }
    print(f"recording {bag}  (Ctrl+C to stop)")
    if shift["disparity_shift"]:
        print(f"close range: depth {args.depth_size}, disparity shift "
              f"{shift['disparity_shift']} -> {shift['min_range_m']:.2f}-"
              f"{shift['max_range_m']:.2f} m")
    if args.genlock:
        print("genlock: depth frames arrive only on Arduino trigger pulses -- start the "
              "Blackfly capture now")
    frames_seen, domain, last_print = 0, None, time.time()
    depth_numbers = set()
    started = time.time()
    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(1000)
            now = time.time()
            if ok:
                frames_seen += 1
                depth = frames.get_depth_frame()
                if depth:
                    depth_numbers.add(depth.get_frame_number())
                    if domain is None:
                        domain = str(depth.get_frame_timestamp_domain())
            if now - last_print >= 2.0:
                elapsed = now - started
                print(f"  {len(depth_numbers)} depth frames ({len(depth_numbers) / elapsed:.1f}"
                      f"/s), {frames_seen} frame sets", flush=True)
                last_print = now
            if args.duration and now - started >= args.duration:
                break
            if args.stop_file is not None and args.stop_file.exists():
                break
    except KeyboardInterrupt:
        print()
    finally:
        pipeline.stop()
        if args.genlock:
            args.genlock = False
            set_sync_mode(args)               # leave the camera free-running
    meta.update({"stopped_host_ns": time.time_ns(), "frame_sets": frames_seen,
                 "depth_frames": len(depth_numbers),
                 "timestamp_domain": domain})
    sidecar = bag.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {frames_seen} frame sets -> {bag}")
    print(f"metadata -> {sidecar}")
    if domain and "system" in domain:
        print("note: timestamps are host arrival times (no RealSense metadata driver); "
              "align_d455.py matches them to the Blackfly arrival times.")


def main():
    args = parse_args()
    if args.preview:
        preview(args)
    else:
        record(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
