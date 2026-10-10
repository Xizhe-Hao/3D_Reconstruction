#!/usr/bin/env python3
"""Attach the D455 recording to a Blackfly capture session, frame by frame.

    python align_d455.py <capture session> [<d455_*.db3>]

For every capture frame set (capture_sequence_index) it takes the D455 frame
whose timestamp is closest to the Blackfly arrival time, and writes into the
session:

    d455/depth/<sequence>.png    uint16 depth, raw units (x depth_scale_m = metres)
    d455/color/<sequence>.jpg    colour frame of the same D455 frame set
    d455/frames.csv              sequence, Blackfly time, D455 frame, |dt|
    d455/d455.json               intrinsics, depth scale, depth->colour extrinsics

Without a recording argument the newest d455_*.db3 next to the session (same parent
folder) whose recording covers the capture is used.

Both clocks are host arrival times, so the residual offset is the difference
in delivery latency of the two camera types, a few tens of milliseconds and
roughly constant. --offset-ms shifts the Blackfly times before matching, for
when that offset has been measured (e.g. from a press visible in both).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 missing:  python -m pip install pyrealsense2 --no-deps")


def capture_times(session: Path) -> list[tuple[int, int]]:
    """(capture_sequence_index, median Blackfly arrival ns) for complete sets."""
    per_sequence: dict[int, list[int]] = {}
    folders = sorted(session.glob("camera_*_*"))
    folders = [f for f in folders if f.is_dir()]
    if not folders:
        raise SystemExit(f"no camera_* folders in {session}")
    for folder in folders:
        with (folder / "frames.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("complete", "1") != "1":
                    continue
                per_sequence.setdefault(int(row["capture_sequence_index"]), []).append(
                    int(row["host_received_ns"]))
    complete = len(folders)
    return [(seq, int(np.median(times))) for seq, times in sorted(per_sequence.items())
            if len(times) == complete]


def find_bag(session: Path, first_ns: int, last_ns: int) -> Path:
    candidates = []
    for sidecar in session.parent.glob("d455_*.json"):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        start, stop = meta.get("started_host_ns", 0), meta.get("stopped_host_ns", 0)
        bag = sidecar.parent / meta.get("recording", sidecar.with_suffix(".db3").name)
        if bag.exists() and start <= first_ns and stop >= last_ns:
            candidates.append((start, bag))
    if not candidates:
        raise SystemExit("no D455 recording next to the session covers the whole capture; "
                         "pass the bag explicitly (and start the D455 recording before "
                         "the capture next time)")
    return max(candidates)[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("bag", type=Path, nargs="?")
    parser.add_argument("--offset-ms", type=float, default=0.0,
                        help="added to the Blackfly times before matching")
    parser.add_argument("--max-dt-ms", type=float, default=None,
                        help="flag matches further apart than this; default 0.6 of a D455 period")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    import cv2

    session = args.session.expanduser().resolve()
    targets = capture_times(session)
    if not targets:
        raise SystemExit("no complete frame sets in the session")
    offset_ns = int(round(args.offset_ms * 1e6))
    bag = args.bag or find_bag(session, targets[0][1], targets[-1][1])
    sidecar = bag.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    fps = float(meta.get("fps", 30))
    # Arrival jitter of a few ms on both sides; flag beyond 0.6 of a D455 period.
    max_dt_ms = args.max_dt_ms if args.max_dt_ms is not None else 600.0 / fps
    if meta.get("sync") == "genlock" and args.max_dt_ms is None and len(targets) > 2:
        # One depth frame per trigger: the nearest one is right as long as it is
        # closer than half a trigger period; delivery latency may exceed 20 ms.
        period_ms = float(np.median(np.diff([t for _, t in targets]))) / 1e6
        max_dt_ms = 0.5 * period_ms
    print(f"session  {session}  ({len(targets)} frame sets)")
    print(f"d455     {bag}")

    out = session / "d455"
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "color").mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag), repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    rows = []
    index = 0
    previous = None          # (ts_ns, frame_number, depth array, colour array)

    def emit(candidate, sequence, target_ns):
        ts_ns, number, depth, color = candidate
        cv2.imwrite(str(out / "depth" / f"{sequence:09d}.png"), depth)
        cv2.imwrite(str(out / "color" / f"{sequence:09d}.jpg"),
                    cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        rows.append({"capture_sequence_index": sequence, "blackfly_host_ns": target_ns,
                     "d455_frame_number": number, "d455_timestamp_ns": ts_ns,
                     "dt_ms": round((ts_ns - (target_ns + offset_ns)) / 1e6, 3)})

    try:
        while index < len(targets):
            ok, frames = pipeline.try_wait_for_frames(2000)
            if not ok:
                break                                    # end of the recording
            depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            ts_ns = int(round(depth_frame.get_timestamp() * 1e6))
            current = (ts_ns, int(depth_frame.get_frame_number()),
                       np.asanyarray(depth_frame.get_data()).copy(),
                       np.asanyarray(color_frame.get_data()).copy())
            # Every target this frame has passed is decided between it and the
            # previous frame.
            while index < len(targets) and targets[index][1] + offset_ns <= ts_ns:
                sequence, target_ns = targets[index]
                wanted = target_ns + offset_ns
                best = current if previous is None or \
                    abs(ts_ns - wanted) <= abs(previous[0] - wanted) else previous
                emit(best, sequence, target_ns)
                index += 1
            previous = current
        while index < len(targets) and previous is not None:
            sequence, target_ns = targets[index]
            emit(previous, sequence, target_ns)          # past the end: last frame
            index += 1
    finally:
        pipeline.stop()

    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dts = np.array([abs(r["dt_ms"]) for r in rows])
    late = int((dts > max_dt_ms).sum())
    meta_out = dict(meta)
    meta_out.update({"recording": str(bag), "depth_scale_m": depth_scale,
                     "offset_ms": args.offset_ms, "matched_frame_sets": len(rows),
                     "dt_ms_median": float(np.median(dts)), "dt_ms_max": float(dts.max()),
                     "matches_over_limit": late, "limit_ms": max_dt_ms})
    print(f"matched {len(rows)}/{len(targets)} frame sets; |dt| median "
          f"{np.median(dts):.1f} ms, max {dts.max():.1f} ms")
    if meta.get("sync") == "genlock":
        # Triggered by the same pulse: one distinct depth frame per frame set,
        # and dt is just the difference in delivery latency -- near constant.
        numbers = [r["d455_frame_number"] for r in rows]
        repeats = len(numbers) - len(set(numbers))
        signed = np.array([r["dt_ms"] for r in rows])
        print(f"genlock: {len(set(numbers))} distinct depth frames for {len(rows)} frame sets"
              f"{'' if not repeats else f' -- {repeats} REUSED (missed triggers?)'}; "
              f"dt {np.median(signed):+.1f} ms +- {signed.std():.1f} ms")
        meta_out_extra = {"genlock_reused_frames": repeats,
                          "genlock_dt_ms_median": float(np.median(signed)),
                          "genlock_dt_ms_std": float(signed.std())}
    else:
        meta_out_extra = {}
    meta_out.update(meta_out_extra)
    (out / "d455.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    if late:
        print(f"WARNING: {late} frame set(s) have no D455 frame within {max_dt_ms:.0f} ms "
              "(recording started late or ended early, or frames were dropped)")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
