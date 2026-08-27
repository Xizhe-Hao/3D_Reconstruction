#!/usr/bin/env python3
"""Replay every capture camera, a 16x16 tactile sensor, and force-gauge data.

The camera count is discovered from the session, so a rig with three, four, or
more cameras replays without a code change. The camera previews are laid out on
an automatically sized grid above the sensor and force panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROWS = 16
COLS = 16
CELL_COUNT = ROWS * COLS
ADC_MAX = 255.0
MAX_CAMERAS_PER_PREVIEW_ROW = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay the synchronized camera streams, the 16x16 sensor, "
            "and aligned IntelliMESUR force data."
        )
    )
    parser.add_argument("session", type=Path, help="Capture session directory")
    parser.add_argument(
        "--cameras",
        type=int,
        default=0,
        help=(
            "Expected camera count. 0 discovers it from the session "
            "(export metadata, session.json, or camera_* directories)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("change", "voltage", "relative-max"),
        default="change",
    )
    parser.add_argument(
        "--sensor-source",
        choices=("auto", "raw", "corrected", "corrected-v2", "filled"),
        default="auto",
    )
    parser.add_argument(
        "--missing-sensor",
        choices=("blank", "interpolate", "nearest"),
        default="blank",
        help=(
            "How to display camera frames whose sensor packet was lost. "
            "Interpolation and nearest-neighbor values are visualization only."
        ),
    )
    parser.add_argument("--fps", type=float)
    parser.add_argument("--baseline-frames", type=int, default=24)
    parser.add_argument("--adc-ref", type=float, default=5.0)
    parser.add_argument("--vdrive", type=float, default=5.0)
    parser.add_argument("--range", dest="display_range", type=float)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--filter-alpha", type=float, default=1.0)
    parser.add_argument("--preview-width", type=int, default=480)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.baseline_frames < 1:
        parser.error("--baseline-frames must be at least 1")
    if args.adc_ref <= 0 or args.vdrive <= 0:
        parser.error("--adc-ref and --vdrive must be positive")
    if args.display_range is not None and args.display_range <= 0:
        parser.error("--range must be positive")
    if args.gain <= 0:
        parser.error("--gain must be positive")
    if not 0 < args.filter_alpha <= 1:
        parser.error("--filter-alpha must satisfy 0 < alpha <= 1")
    if args.preview_width < 160:
        parser.error("--preview-width must be at least 160")
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.cameras < 0:
        parser.error("--cameras must be non-negative")
    return args


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_sensor_path(session_dir, source):
    raw_path = session_dir / "sensor_raw.csv"
    corrected_path = session_dir / "sensor_corrected.csv"
    corrected_v2_path = session_dir / "sensor_corrected_v2.csv"
    filled_path = session_dir / "sensor_corrected_filled.csv"
    if source == "raw":
        selected = raw_path
    elif source == "corrected":
        selected = corrected_path
    elif source == "corrected-v2":
        selected = corrected_v2_path
    elif source == "filled":
        selected = filled_path
    else:
        if filled_path.is_file():
            selected = filled_path
        elif corrected_v2_path.is_file():
            selected = corrected_v2_path
        elif corrected_path.is_file():
            selected = corrected_path
        else:
            selected = raw_path
    if not selected.is_file():
        raise RuntimeError(f"Missing sensor file: {selected}")
    return selected


def camera_index_sort_key(path):
    """Order camera_<index>_<serial> paths numerically, not alphabetically."""
    parts = path.name.split("_")
    try:
        return (0, int(parts[1]), path.name)
    except (IndexError, ValueError):
        return (1, 0, path.name)


def camera_directories(session_dir):
    """Return camera_* directories ordered by their capture index."""
    return sorted(
        (path for path in session_dir.glob("camera_*") if path.is_dir()),
        key=camera_index_sort_key,
    )


def resolve_camera_count(session_dir, requested):
    """Decide how many cameras this session contains."""
    if requested:
        return requested
    export_path = session_dir / "multimodal_video_export.json"
    if export_path.is_file():
        export_metadata = read_json(export_path)
        if export_metadata.get("camera_count"):
            return int(export_metadata["camera_count"])
        if export_metadata.get("videos"):
            return len(export_metadata["videos"])
    session_path = session_dir / "session.json"
    if session_path.is_file():
        session_metadata = read_json(session_path)
        if session_metadata.get("camera_count"):
            return int(session_metadata["camera_count"])
        if session_metadata.get("camera_serials"):
            return len(session_metadata["camera_serials"])
    found = len(camera_directories(session_dir))
    if found < 1:
        raise RuntimeError(f"No camera_* directories in {session_dir}")
    return found


def discover_camera_videos(session_dir, camera_count):
    metadata_path = session_dir / "multimodal_video_export.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
    video_names = [str(name) for name in metadata.get("videos", [])]

    search_roots = (
        session_dir,
        session_dir / "training_dataset" / "videos",
    )
    if video_names:
        videos = []
        for name in video_names:
            path = next(
                (
                    root / Path(name).name
                    for root in search_roots
                    if (root / Path(name).name).is_file()
                ),
                None,
            )
            if path is None:
                raise RuntimeError(f"Missing exported camera video: {name}")
            videos.append(path)
    else:
        # No export metadata: fall back to a glob, ordered by camera index so
        # that camera_10_* never sorts before camera_2_*.
        videos = []
        for root in search_roots:
            candidates = sorted(
                root.glob("camera_*fps*.mp4"),
                key=camera_index_sort_key,
            )
            if len(candidates) == camera_count:
                videos = candidates
                break

    if len(videos) != camera_count:
        raise RuntimeError(
            f"Expected {camera_count} exported camera MP4 files, found "
            f"{len(videos)}. Run export_multimodal_mp4.py before replay."
        )
    return videos, metadata


def load_camera_timeline(session_dir, camera_count):
    camera_dirs = camera_directories(session_dir)
    if len(camera_dirs) != camera_count:
        raise RuntimeError(
            f"Expected {camera_count} camera directories, found "
            f"{len(camera_dirs)}"
        )

    camera_maps = []
    for camera_dir in camera_dirs:
        frame_path = camera_dir / "frames.csv"
        if not frame_path.is_file():
            raise RuntimeError(f"Missing {frame_path}")
        rows = read_csv(frame_path)
        complete = {
            int(row["capture_sequence_index"]): row
            for row in rows
            if int(row.get("complete", "1")) == 1
        }
        if not complete:
            raise RuntimeError(f"{camera_dir.name} contains no complete frames")
        camera_maps.append(complete)

    common_sequences = set(camera_maps[0])
    for frame_map in camera_maps[1:]:
        common_sequences &= set(frame_map)
    sequence = np.asarray(sorted(common_sequences), dtype=np.int64)
    if sequence.size == 0:
        raise RuntimeError("The cameras have no common capture sequence")

    camera_host_ns = []
    for frame_map in camera_maps:
        timestamps = []
        for capture_index in sequence:
            row = frame_map[int(capture_index)]
            timestamps.append(int(row["host_received_ns"]))
        camera_host_ns.append(timestamps)

    video_paths, video_metadata = discover_camera_videos(
        session_dir, camera_count
    )
    video_frame_count = int(
        video_metadata.get("frame_count", sequence.size)
    )
    if video_frame_count != sequence.size:
        raise RuntimeError(
            f"Camera timeline has {sequence.size} common frames but MP4 files "
            f"contain {video_frame_count} frames"
        )
    video_fps = float(video_metadata.get("fps", 24.0))
    if video_fps <= 0:
        raise RuntimeError(f"Invalid MP4 frame rate: {video_fps}")
    source_info = video_metadata.get("source_image_info", [])
    if len(source_info) != camera_count:
        source_info = [
            {"width": 2048, "height": 1536}
            for _ in range(camera_count)
        ]

    camera_host_ns = np.asarray(camera_host_ns, dtype=np.int64)
    master_host_ns = np.median(camera_host_ns, axis=0).astype(np.int64)
    return {
        "count": camera_count,
        "directories": [path.name for path in camera_dirs],
        "names": [path.stem for path in video_paths],
        "videos": video_paths,
        "video_info": source_info,
        "video_frame_count": video_frame_count,
        "video_fps": video_fps,
        "sequence": sequence,
        "host_ns": master_host_ns,
        "per_camera_counts": [len(frame_map) for frame_map in camera_maps],
    }


def interpolate_sensor(raw_available, available_positions, policy):
    frame_count = raw_available.shape[0]
    available = np.flatnonzero(available_positions)
    if available.size == 0:
        raise RuntimeError("No sensor frames match the camera timeline")

    result = raw_available.copy()
    estimated = np.zeros(frame_count, dtype=bool)
    if policy == "blank":
        return result, estimated

    missing = np.flatnonzero(~available_positions)
    if missing.size == 0:
        return result, estimated
    if policy == "nearest":
        insertions = np.searchsorted(available, missing)
        left = available[np.maximum(insertions - 1, 0)]
        right = available[np.minimum(insertions, available.size - 1)]
        choose_right = np.abs(right - missing) < np.abs(missing - left)
        nearest = np.where(choose_right, right, left)
        result[missing] = result[nearest]
        estimated[missing] = True
        return result, estimated

    flattened = result.reshape(frame_count, CELL_COUNT)
    x = np.arange(frame_count, dtype=np.float64)
    first, last = int(available[0]), int(available[-1])
    fill_positions = missing[(missing >= first) & (missing <= last)]
    for cell in range(CELL_COUNT):
        flattened[fill_positions, cell] = np.interp(
            x[fill_positions],
            x[available],
            flattened[available, cell],
        )
    estimated[fill_positions] = True
    return result, estimated


def load_sensor(session_dir, camera_sequence, source, missing_policy):
    sensor_path = select_sensor_path(session_dir, source)
    rows = read_csv(sensor_path)
    if not rows:
        raise RuntimeError(f"{sensor_path.name} contains no frames")
    cell_names = [
        f"r{row + 1}c{column + 1}_adc8"
        for row in range(ROWS)
        for column in range(COLS)
    ]
    missing_columns = [name for name in cell_names if name not in rows[0]]
    if missing_columns:
        raise RuntimeError(f"Missing sensor column: {missing_columns[0]}")

    row_by_index = {int(row["frame_index"]): row for row in rows}
    frame_count = camera_sequence.size
    raw = np.full((frame_count, ROWS, COLS), np.nan, dtype=np.float32)
    exact = np.zeros(frame_count, dtype=bool)
    source_estimated = np.zeros(frame_count, dtype=bool)
    row_for_frame = [None] * frame_count
    for position, sequence_index in enumerate(camera_sequence):
        row = row_by_index.get(int(sequence_index))
        if row is None:
            continue
        source_name = row.get("frame_source", "exact") or "exact"
        if source_name == "unavailable":
            continue
        try:
            raw[position] = np.asarray(
                [int(row[name]) for name in cell_names],
                dtype=np.float32,
            ).reshape(ROWS, COLS)
        except (TypeError, ValueError):
            continue
        exact[position] = source_name == "exact"
        source_estimated[position] = source_name != "exact"
        row_for_frame[position] = row

    available = np.isfinite(raw).all(axis=(1, 2))
    raw, visual_estimated = interpolate_sensor(
        raw,
        available,
        missing_policy,
    )
    estimated = source_estimated | visual_estimated
    missing_indices = camera_sequence[~(exact | estimated)]
    return {
        "path": sensor_path,
        "rows": row_for_frame,
        "raw": raw,
        "exact": exact,
        "estimated": estimated,
        "missing_indices": missing_indices,
        "source_row_count": len(rows),
    }


def load_force(session_dir, master_host_ns):
    samples_path = session_dir / "force_samples_aligned.csv"
    alignment_path = session_dir / "force_alignment.json"
    if not samples_path.is_file():
        return None

    rows = read_csv(samples_path)
    if not rows:
        return None
    sample_ns = np.asarray(
        [int(row["estimated_host_ns"]) for row in rows], dtype=np.int64
    )
    load = np.asarray([float(row["force_load"]) for row in rows])
    distance = np.asarray([float(row["force_distance"]) for row in rows])
    force_time = np.asarray([float(row["force_time_s"]) for row in rows])
    valid = (
        np.isfinite(load)
        & np.isfinite(distance)
        & np.isfinite(force_time)
    )
    sample_ns = sample_ns[valid]
    load = load[valid]
    distance = distance[valid]
    force_time = force_time[valid]
    if sample_ns.size < 2:
        return None

    load_at_frame = np.interp(
        master_host_ns.astype(np.float64),
        sample_ns.astype(np.float64),
        load,
        left=np.nan,
        right=np.nan,
    )
    distance_at_frame = np.interp(
        master_host_ns.astype(np.float64),
        sample_ns.astype(np.float64),
        distance,
        left=np.nan,
        right=np.nan,
    )
    alignment = read_json(alignment_path) if alignment_path.is_file() else {}
    return {
        "path": samples_path,
        "sample_ns": sample_ns,
        "time_s": force_time,
        "load": load,
        "distance": distance,
        "load_at_frame": load_at_frame,
        "distance_at_frame": distance_at_frame,
        "alignment": alignment,
    }


def estimate_fps(host_ns):
    intervals = np.diff(host_ns.astype(np.float64)) / 1e9
    intervals = intervals[(intervals > 0) & np.isfinite(intervals)]
    return 1.0 / np.median(intervals) if intervals.size else 24.0


def load_session(session_dir, args):
    session_dir = session_dir.expanduser().resolve()
    if not session_dir.is_dir():
        raise RuntimeError(f"Session directory does not exist: {session_dir}")
    camera_count = resolve_camera_count(session_dir, args.cameras)
    cameras = load_camera_timeline(session_dir, camera_count)
    sensor = load_sensor(
        session_dir,
        cameras["sequence"],
        args.sensor_source,
        args.missing_sensor,
    )
    force = load_force(session_dir, cameras["host_ns"])
    metadata_path = session_dir / "session.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    relative_time_s = (
        cameras["host_ns"] - cameras["host_ns"][0]
    ).astype(np.float64) / 1e9
    return {
        "session_dir": session_dir,
        "cameras": cameras,
        "sensor": sensor,
        "force": force,
        "metadata": metadata,
        "time_s": relative_time_s,
        "fps": cameras["video_fps"],
        "capture_fps": estimate_fps(cameras["host_ns"]),
    }


def ema_filter_with_gaps(values, alpha):
    if alpha >= 1.0:
        return values
    filtered = values.copy()
    previous = None
    for index in range(values.shape[0]):
        current = values[index]
        if not np.isfinite(current).any():
            continue
        if previous is None:
            filtered[index] = current
        else:
            filtered[index] = alpha * current + (1.0 - alpha) * previous
        previous = filtered[index]
    return filtered


def prepare_sensor_data(raw, exact, args):
    voltage = args.vdrive - raw * (args.adc_ref / ADC_MAX)
    valid_baseline = np.flatnonzero(exact)[: args.baseline_frames]
    if valid_baseline.size == 0:
        raise RuntimeError("No exact sensor frames are available for baseline")

    if args.mode == "voltage":
        displayed = voltage
        description = "Sensor voltage"
        z_min, z_max = 0.0, args.display_range or args.vdrive
        cmap = plt.get_cmap("turbo")
        norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
    elif args.mode == "relative-max":
        frame_max = np.nanmax(voltage, axis=(1, 2), keepdims=True)
        sign = -1.0 if args.invert else 1.0
        displayed = sign * (voltage - frame_max) * args.gain
        description = (
            "Maximum - sensor voltage"
            if args.invert
            else "Sensor voltage - maximum"
        )
        limit = args.display_range or 2.0
        z_min, z_max = -limit, limit
        cmap = plt.get_cmap("RdBu_r")
        norm = mcolors.TwoSlopeNorm(vmin=z_min, vcenter=0.0, vmax=z_max)
    else:
        baseline = np.nanmedian(voltage[valid_baseline], axis=0)
        sign = -1.0 if args.invert else 1.0
        displayed = sign * (voltage - baseline) * args.gain
        description = (
            "Baseline - sensor voltage"
            if args.invert
            else "Sensor voltage - baseline"
        )
        limit = args.display_range or 2.0
        z_min, z_max = -limit, limit
        cmap = plt.get_cmap("RdBu_r")
        norm = mcolors.TwoSlopeNorm(vmin=z_min, vcenter=0.0, vmax=z_max)

    displayed = displayed.astype(np.float32, copy=False)
    displayed = ema_filter_with_gaps(displayed, args.filter_alpha)
    return displayed, description, z_min, z_max, cmap, norm


class VideoFrameReader:
    def __init__(
        self,
        path,
        fps,
        source_width,
        source_height,
        preview_width,
    ):
        self.path = path
        self.fps = fps
        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if source_width > preview_width:
            self.width = preview_width
            scaled_height = source_height * preview_width / source_width
            self.height = max(2, int(round(scaled_height / 2.0) * 2))
        else:
            self.width = source_width
            self.height = source_height
        self.frame_bytes = self.width * self.height * 3
        self.process = None
        self.next_index = 0
        self.last_index = -1
        self.last_frame = None

    def _stop(self):
        if self.process is None:
            return
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None

    def _start(self, index):
        self._stop()
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.path),
        ]
        if index:
            command.extend(("-ss", f"{index / self.fps:.9f}"))
        command.extend(
            (
                "-an",
                "-vf",
                f"scale={self.width}:{self.height}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            )
        )
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.next_index = index

    def _read_exact(self):
        chunks = []
        remaining = self.frame_bytes
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining:
            raise RuntimeError(
                f"FFmpeg ended while decoding frame {self.next_index} "
                f"from {self.path.name}"
            )
        return b"".join(chunks)

    def read(self, index):
        index = int(index)
        if index == self.last_index and self.last_frame is not None:
            return self.last_frame
        if (
            self.process is not None
            and self.next_index < index
            and index - self.next_index <= max(12, round(self.fps))
        ):
            while self.next_index < index:
                self._read_exact()
                self.next_index += 1
        elif self.process is None or index != self.next_index:
            self._start(index)
        frame = np.frombuffer(self._read_exact(), dtype=np.uint8).reshape(
            self.height,
            self.width,
            3,
        )
        self.next_index = index + 1
        self.last_index = index
        self.last_frame = frame.copy()
        return self.last_frame

    def close(self):
        self._stop()


def validate_video_decoding(cameras, preview_width):
    indices = sorted(
        {
            0,
            cameras["video_frame_count"] // 2,
            cameras["video_frame_count"] - 1,
        }
    )
    readers = [
        VideoFrameReader(
            path,
            cameras["video_fps"],
            int(info["width"]),
            int(info["height"]),
            preview_width,
        )
        for path, info in zip(cameras["videos"], cameras["video_info"])
    ]
    try:
        for reader in readers:
            for index in indices:
                reader.read(index)
    finally:
        for reader in readers:
            reader.close()
    print(f"MP4 decode check passed at frames: {indices}")


def camera_grid_shape(camera_count):
    """Rows and columns for the camera preview block."""
    columns = min(camera_count, MAX_CAMERAS_PER_PREVIEW_ROW)
    rows = -(-camera_count // columns)
    return rows, columns


class MultimodalReplay:
    BAR_MARGIN = 0.10
    FACE_SHADES = np.asarray((1.0, 0.78, 0.88, 0.70, 0.82), dtype=np.float32)

    def __init__(
        self,
        session,
        sensor_data,
        description,
        z_min,
        z_max,
        cmap,
        norm,
        args,
    ):
        self.session = session
        self.sensor_data = sensor_data
        self.description = description
        self.z_min = z_min
        self.z_max = z_max
        self.cmap = cmap
        self.norm = norm
        self.args = args
        self.camera_count = session["cameras"]["count"]
        self.frame_count = session["cameras"]["sequence"].size
        self.fps = args.fps or session["fps"]
        self.current_frame = min(args.start_frame, self.frame_count - 1)
        self.playing = not args.paused
        self.speed = 1.0
        self.last_tick = time.perf_counter()
        self.frame_accumulator = 0.0
        self.updating_slider = False
        self.video_readers = [
            VideoFrameReader(
                path,
                session["cameras"]["video_fps"],
                int(info["width"]),
                int(info["height"]),
                args.preview_width,
            )
            for path, info in zip(
                session["cameras"]["videos"],
                session["cameras"]["video_info"],
            )
        ]
        self._prepare_bar_coordinates()
        self._build_figure()
        self.show_frame(self.current_frame, force=True)

    def _prepare_bar_coordinates(self):
        rows, columns = np.indices((ROWS, COLS))
        width = 1.0 - 2.0 * self.BAR_MARGIN
        self.x0 = columns.ravel().astype(np.float32) + self.BAR_MARGIN
        self.x1 = self.x0 + width
        self.y0 = rows.ravel().astype(np.float32) + self.BAR_MARGIN
        self.y1 = self.y0 + width

    def _build_figure(self):
        preview_rows, preview_columns = camera_grid_shape(self.camera_count)
        # Give the preview block more of the figure when the cameras wrap onto
        # a second row, and grow the window instead of shrinking each preview.
        figure_height = 10.0 + 2.6 * (preview_rows - 1)
        preview_share = 0.72 + 0.62 * (preview_rows - 1)
        self.fig = plt.figure(figsize=(17, figure_height))
        grid = self.fig.add_gridspec(
            2,
            1,
            height_ratios=(preview_share, 1.28),
            left=0.035,
            right=0.965,
            # Leave room above the previews so their per-camera titles do not
            # collide with the figure-level frame counter.
            top=0.925,
            bottom=0.14,
            hspace=0.14,
        )

        preview_grid = grid[0].subgridspec(
            preview_rows,
            preview_columns,
            hspace=0.18,
            wspace=0.06,
        )
        self.image_artists = []
        for camera_index in range(self.camera_count):
            row, column = divmod(camera_index, preview_columns)
            axis = self.fig.add_subplot(preview_grid[row, column])
            preview = self.video_readers[camera_index].read(
                self.current_frame
            )
            artist = axis.imshow(
                preview,
                interpolation="nearest",
            )
            axis.set_title(
                f"[{camera_index}] "
                f"{self.session['cameras']['names'][camera_index]}",
                fontsize=9,
            )
            axis.set_axis_off()
            self.image_artists.append(artist)

        # A partly filled last preview row leaves blank cells; that is
        # intentional and keeps every preview the same size.
        data_grid = grid[1].subgridspec(1, 4, wspace=0.11)
        self.ax3d = self.fig.add_subplot(data_grid[0, :3], projection="3d")
        initial = self.matrix_for_display(self.current_frame)
        faces, colors = self.build_bar_geometry(initial)
        self.bar_collection = Poly3DCollection(
            faces,
            facecolors=colors,
            edgecolors=(0.10, 0.10, 0.10, 0.40),
            linewidths=0.16,
        )
        self.ax3d.add_collection3d(self.bar_collection)
        self.ax3d.set_xlim(0, COLS)
        self.ax3d.set_ylim(ROWS, 0)
        self.ax3d.set_zlim(self.z_min, self.z_max)
        self.ax3d.set_xticks(np.arange(COLS) + 0.5)
        self.ax3d.set_yticks(np.arange(ROWS) + 0.5)
        self.ax3d.set_xticklabels(
            [f"C{index + 1}" for index in range(COLS)], fontsize=6
        )
        self.ax3d.set_yticklabels(
            [f"R{index + 1}" for index in range(ROWS)], fontsize=6
        )
        self.ax3d.set_xlabel("Column")
        self.ax3d.set_ylabel("Row")
        self.ax3d.set_zlabel("Voltage (V)")
        self.ax3d.tick_params(axis="x", labelrotation=45)
        self.ax3d.view_init(elev=30, azim=-53)
        self.ax3d.set_box_aspect((16, 16, 7.5))

        scalar_mappable = matplotlib.cm.ScalarMappable(
            norm=self.norm, cmap=self.cmap
        )
        scalar_mappable.set_array([])
        colorbar = self.fig.colorbar(
            scalar_mappable, ax=self.ax3d, shrink=0.68, pad=0.055
        )
        colorbar.set_label(f"{self.description} (V)")

        self.label_artists = []
        if self.args.labels:
            for row in range(ROWS):
                for column in range(COLS):
                    self.label_artists.append(
                        self.ax3d.text(
                            column + 0.5,
                            row + 0.5,
                            0.0,
                            "0.00",
                            ha="center",
                            va="bottom",
                            fontsize=3.8,
                        )
                    )

        self._build_force_plot(data_grid)
        self._build_controls()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.add_callback(self.on_timer)
        self.timer.start()

    def _build_force_plot(self, data_grid):
        self.force_axis = self.fig.add_subplot(data_grid[0, 3])
        self.distance_axis = self.force_axis.twinx()
        force = self.session["force"]
        self.force_cursor = None
        self.force_marker = None
        self.distance_marker = None
        if force is None:
            self.force_axis.text(
                0.5,
                0.5,
                "No aligned force data",
                ha="center",
                va="center",
                transform=self.force_axis.transAxes,
            )
            self.force_axis.set_axis_off()
            self.distance_axis.set_axis_off()
            return

        x = (
            force["sample_ns"] - self.session["cameras"]["host_ns"][0]
        ).astype(np.float64) / 1e9
        self.force_axis.plot(
            x, force["load"], color="#d13a33", linewidth=1.1, label="Load"
        )
        self.distance_axis.plot(
            x,
            force["distance"],
            color="#2068a8",
            linewidth=0.8,
            alpha=0.75,
            label="Distance",
        )
        self.force_cursor = self.force_axis.axvline(
            self.session["time_s"][self.current_frame],
            color="#202020",
            linewidth=1.1,
        )
        self.force_marker, = self.force_axis.plot(
            [], [], "o", color="#d13a33", markersize=5
        )
        self.distance_marker, = self.distance_axis.plot(
            [], [], "o", color="#2068a8", markersize=4
        )
        self.force_axis.set_xlabel("Capture time (s)")
        self.force_axis.set_ylabel("Load (N)", color="#d13a33")
        self.distance_axis.set_ylabel("Distance (mm)", color="#2068a8")
        self.force_axis.grid(True, alpha=0.24)
        self.force_axis.set_title("Force gauge")

    def _build_controls(self):
        progress_axis = self.fig.add_axes((0.12, 0.076, 0.70, 0.025))
        self.progress_slider = Slider(
            progress_axis,
            "Frame",
            0,
            self.frame_count - 1,
            valinit=self.current_frame,
            valstep=1,
            valfmt="%0.0f",
        )
        self.progress_slider.on_changed(self.on_progress_changed)

        speed_axis = self.fig.add_axes((0.12, 0.027, 0.28, 0.025))
        self.speed_slider = Slider(
            speed_axis,
            "Speed",
            0.25,
            4.0,
            valinit=1.0,
            valstep=0.25,
            valfmt="%0.2fx",
        )
        self.speed_slider.on_changed(self.on_speed_changed)

        specs = (
            ("|<", 0.48, self.go_first),
            ("<", 0.535, self.go_previous),
            ("Pause" if self.playing else "Play", 0.59, self.toggle_play),
            (">", 0.68, self.go_next),
            (">|", 0.735, self.go_last),
        )
        self.buttons = []
        for label, left, callback in specs:
            width = 0.08 if label in ("Play", "Pause") else 0.045
            axis = self.fig.add_axes((left, 0.020, width, 0.04))
            button = Button(axis, label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.play_button = self.buttons[2]

    def matrix_for_display(self, index):
        matrix = self.sensor_data[index]
        if np.isfinite(matrix).any():
            return matrix
        return np.zeros((ROWS, COLS), dtype=np.float32)

    def build_bar_geometry(self, matrix):
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
        colors[:, :3] *= np.tile(self.FACE_SHADES, CELL_COUNT)[:, None]
        np.clip(colors, 0.0, 1.0, out=colors)
        return faces.reshape(CELL_COUNT * 5, 4, 3), colors

    def sensor_state(self, index):
        sensor = self.session["sensor"]
        if sensor["exact"][index]:
            return "EXACT"
        if sensor["estimated"][index]:
            return f"{self.args.missing_sensor.upper()} (VISUAL ONLY)"
        return "MISSING"

    def show_frame(self, index, force=False):
        index = int(np.clip(index, 0, self.frame_count - 1))
        if index == self.current_frame and not force:
            return
        self.current_frame = index

        for camera_index, artist in enumerate(self.image_artists):
            preview = self.video_readers[camera_index].read(index)
            artist.set_data(preview)

        matrix = self.matrix_for_display(index)
        faces, colors = self.build_bar_geometry(matrix)
        if not np.isfinite(self.sensor_data[index]).any():
            colors[:] = (0.55, 0.55, 0.55, 0.45)
        self.bar_collection.set_verts(faces)
        self.bar_collection.set_facecolor(colors)
        self.update_labels(index, matrix)

        sequence = int(self.session["cameras"]["sequence"][index])
        elapsed = float(self.session["time_s"][index])
        state = self.sensor_state(index)
        finite = np.isfinite(self.sensor_data[index])
        value_range = (
            f"{np.nanmin(self.sensor_data[index]):+.3f} to "
            f"{np.nanmax(self.sensor_data[index]):+.3f} V"
            if finite.any()
            else "unavailable"
        )
        self.ax3d.set_title(
            f"{self.description} | camera sequence {sequence} | "
            f"t={elapsed:.3f}s\nSensor: {state} | range {value_range}",
            color="#b32020" if state == "MISSING" else "#202020",
        )
        self.update_force(index, elapsed)

        self.updating_slider = True
        self.progress_slider.set_val(index)
        self.updating_slider = False
        self.fig.suptitle(
            f"Frame {index + 1}/{self.frame_count} | "
            f"capture_sequence_index={sequence} | "
            f"{self.camera_count} cameras",
            fontsize=11,
            y=0.985,
        )
        self.fig.canvas.draw_idle()

    def update_force(self, index, elapsed):
        force = self.session["force"]
        if force is None:
            return
        load = float(force["load_at_frame"][index])
        distance = float(force["distance_at_frame"][index])
        self.force_cursor.set_xdata([elapsed, elapsed])
        if np.isfinite(load):
            self.force_marker.set_data([elapsed], [load])
        else:
            self.force_marker.set_data([], [])
        if np.isfinite(distance):
            self.distance_marker.set_data([elapsed], [distance])
        else:
            self.distance_marker.set_data([], [])
        load_text = f"{load:+.3f} N" if np.isfinite(load) else "outside run"
        distance_text = (
            f"{distance:+.3f} mm"
            if np.isfinite(distance)
            else "outside run"
        )
        alignment = force["alignment"]
        confidence = alignment.get("confidence", "unknown")
        score = alignment.get("score")
        score_text = f"{float(score):.3f}" if score is not None else "n/a"
        self.force_axis.set_title(
            f"Force gauge | load {load_text}\n"
            f"distance {distance_text} | alignment {score_text} ({confidence})"
        )

    def update_labels(self, index, matrix):
        if not self.label_artists:
            return
        available = np.isfinite(self.sensor_data[index]).any()
        offset = max(0.02, (self.z_max - self.z_min) * 0.012)
        for flat_index, artist in enumerate(self.label_artists):
            row, column = divmod(flat_index, COLS)
            value = float(matrix[row, column])
            visible = float(np.clip(value, self.z_min, self.z_max))
            artist.set_text(f"{value:+.2f}" if available else "--")
            artist.set_position((column + 0.5, row + 0.5))
            artist.set_va("bottom" if visible >= 0 else "top")
            artist.set_3d_properties(
                visible + offset if visible >= 0 else visible - offset,
                zdir="z",
            )

    def set_playing(self, playing):
        self.playing = playing
        self.play_button.label.set_text("Pause" if playing else "Play")
        self.last_tick = time.perf_counter()
        self.frame_accumulator = 0.0
        self.fig.canvas.draw_idle()

    def toggle_play(self, _event=None):
        if self.current_frame >= self.frame_count - 1 and not self.playing:
            self.show_frame(0, force=True)
        self.set_playing(not self.playing)

    def go_first(self, _event=None):
        self.set_playing(False)
        self.show_frame(0, force=True)

    def go_last(self, _event=None):
        self.set_playing(False)
        self.show_frame(self.frame_count - 1, force=True)

    def go_previous(self, _event=None):
        self.set_playing(False)
        self.show_frame(self.current_frame - 1, force=True)

    def go_next(self, _event=None):
        self.set_playing(False)
        self.show_frame(self.current_frame + 1, force=True)

    def on_progress_changed(self, value):
        if self.updating_slider:
            return
        self.set_playing(False)
        self.show_frame(round(value), force=True)

    def on_speed_changed(self, value):
        self.speed = float(value)
        self.last_tick = time.perf_counter()
        self.frame_accumulator = 0.0

    def on_timer(self):
        now = time.perf_counter()
        elapsed = now - self.last_tick
        self.last_tick = now
        if not self.playing:
            return True
        self.frame_accumulator += elapsed * self.fps * self.speed
        advance = int(self.frame_accumulator)
        if advance < 1:
            return True
        self.frame_accumulator -= advance
        target = self.current_frame + advance
        if target >= self.frame_count - 1:
            self.show_frame(self.frame_count - 1, force=True)
            self.set_playing(False)
        else:
            self.show_frame(target, force=True)
        return True

    def on_key_press(self, event):
        if event.key == " ":
            self.toggle_play()
        elif event.key == "left":
            self.go_previous()
        elif event.key == "right":
            self.go_next()
        elif event.key == "home":
            self.go_first()
        elif event.key == "end":
            self.go_last()
        elif event.key in ("escape", "q"):
            plt.close(self.fig)

    def on_close(self, _event):
        self.timer.stop()
        for reader in self.video_readers:
            reader.close()

    def run(self):
        plt.show()


def print_summary(session, args):
    cameras = session["cameras"]
    sensor = session["sensor"]
    force = session["force"]
    exact_count = int(sensor["exact"].sum())
    estimated_count = int(sensor["estimated"].sum())
    missing_count = int((~(sensor["exact"] | sensor["estimated"])).sum())
    print(f"Session: {session['session_dir']}")
    print(
        f"Cameras: count={cameras['count']}, "
        f"common={cameras['sequence'].size}, "
        f"individual={cameras['per_camera_counts']}"
    )
    print(f"Camera directories: {cameras['directories']}")
    print(
        f"Camera source: MP4 | frames="
        f"{[cameras['video_frame_count']] * cameras['count']} | "
        f"fps={cameras['video_fps']:.3f}"
    )
    print(
        f"Sensor: rows={sensor['source_row_count']}, matched={exact_count}, "
        f"estimated={estimated_count}, "
        f"missing_on_camera_timeline={missing_count}"
    )
    print(f"Sensor source: {sensor['path'].name}")
    print(f"Missing sensor policy: {args.missing_sensor}")
    if missing_count:
        values = sensor["missing_indices"]
        preview = ", ".join(str(int(value)) for value in values[:20])
        suffix = " ..." if values.size > 20 else ""
        print(f"Missing capture sequence indices: {preview}{suffix}")
    if force is None:
        print("Force: no force_samples_aligned.csv")
    else:
        valid_frames = int(np.isfinite(force["load_at_frame"]).sum())
        alignment = force["alignment"]
        print(
            f"Force: samples={force['sample_ns'].size}, "
            f"camera_frames_in_force_run={valid_frames}, "
            f"score={alignment.get('score', 'n/a')}, "
            f"confidence={alignment.get('confidence', 'unknown')}"
        )
    print(
        f"Playback rate: {session['fps']:.3f} fps | "
        f"measured capture rate: {session['capture_fps']:.3f} fps"
    )


def main():
    args = parse_args()
    try:
        session = load_session(args.session, args)
        sensor_data, description, z_min, z_max, cmap, norm = (
            prepare_sensor_data(
                session["sensor"]["raw"],
                session["sensor"]["exact"],
                args,
            )
        )
        fps = args.fps or session["fps"]
        if fps <= 0:
            raise RuntimeError("Playback fps must be positive")
    except Exception as exc:
        raise SystemExit(f"Replay setup failed: {exc}") from exc

    print_summary(session, args)
    print(f"Sensor display: {description}")
    if args.check_only:
        validate_video_decoding(
            session["cameras"],
            args.preview_width,
        )
        return 0

    replay = MultimodalReplay(
        session,
        sensor_data,
        description,
        z_min,
        z_max,
        cmap,
        norm,
        args,
    )
    replay.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
