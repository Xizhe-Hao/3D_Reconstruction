#!/usr/bin/env python3
"""Export one video per camera and a camera/sensor/force alignment table.

The number of cameras is read from the capture session (session.json, or the
camera_* directories) so that adding or removing a camera needs no code change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


BAYER_FFMPEG_FORMATS = {
    "BayerRG8": "bayer_rggb8",
    "BayerBG8": "bayer_bggr8",
    "BayerGB8": "bayer_gbrg8",
    "BayerGR8": "bayer_grbg8",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--preset",
        default="medium",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Encode only the first N common frames; 0 encodes all frames.",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=0,
        help=(
            "Expected camera count. 0 takes it from session.json, falling "
            "back to the number of camera_* directories."
        ),
    )
    parser.add_argument(
        "--delete-images",
        action="store_true",
        help=(
            "Delete source images only after every MP4 file decodes to the "
            "expected frame count and alignment files are written."
        ),
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.cameras < 0:
        parser.error("--cameras must be non-negative")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if args.delete_images and args.max_frames:
        parser.error("--delete-images cannot be combined with --max-frames")
    return args


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_ffmpeg():
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is required: python -m pip install imageio-ffmpeg"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def camera_index_sort_key(path):
    """Order camera_<index>_<serial> paths numerically, not alphabetically."""
    parts = path.name.split("_")
    try:
        return (0, int(parts[1]), path.name)
    except (IndexError, ValueError):
        return (1, 0, path.name)


def camera_directories(session):
    """Return camera_* directories ordered by their capture index."""
    return sorted(
        (path for path in session.glob("camera_*") if path.is_dir()),
        key=camera_index_sort_key,
    )


def expected_camera_count(session, session_metadata, requested):
    """Resolve how many cameras this session is supposed to contain."""
    if requested:
        return requested
    recorded = session_metadata.get("camera_count")
    if recorded:
        return int(recorded)
    serials = session_metadata.get("camera_serials")
    if serials:
        return len(serials)
    found = len(camera_directories(session))
    if found < 1:
        raise RuntimeError(f"No camera_* directories in {session}")
    return found


def load_cameras(session, camera_count):
    camera_dirs = camera_directories(session)
    if len(camera_dirs) != camera_count:
        raise RuntimeError(
            f"Expected {camera_count} camera directories, "
            f"found {len(camera_dirs)}"
        )

    maps = []
    for directory in camera_dirs:
        frame_log = directory / "frames.csv"
        if not frame_log.is_file():
            raise RuntimeError(f"Missing {frame_log}")
        mapping = {}
        for row in read_csv(frame_log):
            if int(row.get("complete", "0")) != 1 or not row["image_path"]:
                continue
            sequence = int(row["capture_sequence_index"])
            image_path = session / row["image_path"]
            if not image_path.is_file():
                raise RuntimeError(f"Missing image: {image_path}")
            mapping[sequence] = row
        if not mapping:
            raise RuntimeError(f"{directory.name} contains no usable frames")
        maps.append(mapping)

    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    sequence = sorted(common)
    if not sequence:
        raise RuntimeError("The cameras have no common complete frame")
    return camera_dirs, maps, sequence


def select_sensor_file(session):
    candidates = (
        session / "sensor_corrected_filled.csv",
        session / "sensor_corrected_v2.csv",
        session / "sensor_corrected.csv",
        session / "sensor_raw.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_sensor(session):
    path = select_sensor_file(session)
    if path is None:
        return None
    rows = read_csv(path)
    mapping = {}
    for row_index, row in enumerate(rows):
        frame_index = int(row["frame_index"])
        source = row.get("frame_source", "exact") or "exact"
        mapping[frame_index] = (row_index, source, row)
    return {"path": path, "rows": rows, "mapping": mapping}


def load_force(session):
    path = session / "force_samples_aligned.csv"
    if not path.is_file():
        return None
    rows = read_csv(path)
    sample_ns = []
    force_time = []
    load = []
    distance = []
    for row in rows:
        try:
            sample_ns.append(int(row["estimated_host_ns"]))
            force_time.append(float(row["force_time_s"]))
            load.append(float(row["force_load"]))
            distance.append(float(row["force_distance"]))
        except (KeyError, TypeError, ValueError):
            continue
    if len(sample_ns) < 2:
        return None
    return {
        "path": path,
        "sample_ns": np.asarray(sample_ns, dtype=np.int64),
        "time_s": np.asarray(force_time, dtype=np.float64),
        "load": np.asarray(load, dtype=np.float64),
        "distance": np.asarray(distance, dtype=np.float64),
    }


def inspect_image(path):
    with Image.open(path) as image:
        return {
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "color": image.mode in ("RGB", "RGBA", "CMYK", "YCbCr"),
        }


def encode_camera(
    ffmpeg,
    session,
    camera_dir,
    rows,
    sequence,
    fps,
    crf,
    preset,
    overwrite,
    suffix,
    image_info,
    saved_pixel_format,
):
    output = session / f"{camera_dir.name}_{suffix}.mp4"
    if output.exists() and not overwrite:
        raise RuntimeError(f"Output exists: {output}; use --overwrite")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
    ]
    bayer_ffmpeg_format = BAYER_FFMPEG_FORMATS.get(saved_pixel_format)
    if saved_pixel_format.startswith("Bayer") and not bayer_ffmpeg_format:
        raise RuntimeError(
            f"Unsupported saved Bayer format: {saved_pixel_format}"
        )
    if bayer_ffmpeg_format:
        command.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                bayer_ffmpeg_format,
                "-video_size",
                f"{image_info['width']}x{image_info['height']}",
                "-framerate",
                f"{fps:.9g}",
                "-i",
                "pipe:0",
            ]
        )
    else:
        command.extend(
            [
                "-f",
                "image2pipe",
                "-framerate",
                f"{fps:.9g}",
                "-i",
                "pipe:0",
            ]
        )
    command.extend(
        [
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    print(f"Encoding {camera_dir.name} -> {output.name}")
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for capture_index in sequence:
            image_path = session / rows[capture_index]["image_path"]
            if bayer_ffmpeg_format:
                with Image.open(image_path) as image:
                    if image.mode != "L":
                        image = image.convert("L")
                    process.stdin.write(image.tobytes())
            else:
                with image_path.open("rb") as image_handle:
                    while chunk := image_handle.read(1024 * 1024):
                        process.stdin.write(chunk)
        process.stdin.close()
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise RuntimeError(
            f"FFmpeg failed for {camera_dir.name}: code {return_code}"
        )
    return output


def decoded_frame_count(ffmpeg, video):
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        "-nostats",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Could not verify {video.name}: {result.stderr.strip()}"
        )
    frames = [
        int(line.split("=", 1)[1])
        for line in result.stdout.splitlines()
        if line.startswith("frame=")
    ]
    if not frames:
        raise RuntimeError(f"FFmpeg reported no decoded frames for {video}")
    return frames[-1]


def interpolate_force(force, host_ns):
    if force is None:
        return "", "", "", 0
    first = int(force["sample_ns"][0])
    last = int(force["sample_ns"][-1])
    if host_ns < first or host_ns > last:
        return "", "", "", 0
    x = float(host_ns)
    sample_x = force["sample_ns"].astype(np.float64)
    force_time = np.interp(x, sample_x, force["time_s"])
    load = np.interp(x, sample_x, force["load"])
    distance = np.interp(x, sample_x, force["distance"])
    return (
        f"{force_time:.6f}",
        f"{load:.9g}",
        f"{distance:.9g}",
        1,
    )


def write_alignment(
    session,
    camera_dirs,
    camera_maps,
    sequence,
    fps,
    sensor,
    force,
):
    path = session / "multimodal_video_alignment.csv"
    fieldnames = [
        "video_frame_index",
        "video_time_s",
        "capture_sequence_index",
        "capture_host_ns",
        "capture_time_s",
        "sensor_available",
        "sensor_exact",
        "sensor_frame_source",
        "sensor_csv",
        "sensor_csv_row_index",
        "sensor_frame_index",
        "sensor_device_millis",
        "sensor_host_received_ns",
        "force_in_run",
        "force_time_s",
        "force_load",
        "force_distance",
    ]
    camera_count = len(camera_dirs)
    for camera_index in range(camera_count):
        prefix = f"camera_{camera_index}"
        fieldnames.extend(
            (
                f"{prefix}_directory",
                f"{prefix}_frame_id",
                f"{prefix}_timestamp",
                f"{prefix}_host_received_ns",
                f"{prefix}_image_path",
            )
        )

    first_host_ns = None
    exact_count = 0
    estimated_count = 0
    missing_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for video_index, capture_index in enumerate(sequence):
            camera_rows = [
                camera_maps[index][capture_index]
                for index in range(camera_count)
            ]
            host_ns = int(
                statistics.median(
                    int(row["host_received_ns"]) for row in camera_rows
                )
            )
            if first_host_ns is None:
                first_host_ns = host_ns

            output = {
                "video_frame_index": video_index,
                "video_time_s": f"{video_index / fps:.9f}",
                "capture_sequence_index": capture_index,
                "capture_host_ns": host_ns,
                "capture_time_s": f"{(host_ns - first_host_ns) / 1e9:.9f}",
                "sensor_available": 0,
                "sensor_exact": 0,
                "sensor_frame_source": "missing",
                "sensor_csv": sensor["path"].name if sensor else "",
                "sensor_csv_row_index": "",
                "sensor_frame_index": "",
                "sensor_device_millis": "",
                "sensor_host_received_ns": "",
            }
            sensor_item = (
                sensor["mapping"].get(capture_index) if sensor else None
            )
            if sensor_item is None:
                missing_count += 1
            else:
                row_index, source, sensor_row = sensor_item
                available = source != "unavailable"
                exact = source == "exact"
                if exact:
                    exact_count += 1
                elif available:
                    estimated_count += 1
                else:
                    missing_count += 1
                output.update(
                    {
                        "sensor_available": int(available),
                        "sensor_exact": int(exact),
                        "sensor_frame_source": source,
                        "sensor_csv_row_index": row_index,
                        "sensor_frame_index": sensor_row["frame_index"],
                        "sensor_device_millis": sensor_row["device_millis"],
                        "sensor_host_received_ns": sensor_row[
                            "host_received_ns"
                        ],
                    }
                )

            (
                force_time,
                force_load,
                force_distance,
                force_in_run,
            ) = interpolate_force(force, host_ns)
            output.update(
                {
                    "force_in_run": force_in_run,
                    "force_time_s": force_time,
                    "force_load": force_load,
                    "force_distance": force_distance,
                }
            )
            for camera_index, (directory, camera_row) in enumerate(
                zip(camera_dirs, camera_rows)
            ):
                prefix = f"camera_{camera_index}"
                output.update(
                    {
                        f"{prefix}_directory": directory.name,
                        f"{prefix}_frame_id": camera_row["camera_frame_id"],
                        f"{prefix}_timestamp": camera_row["camera_timestamp"],
                        f"{prefix}_host_received_ns": camera_row[
                            "host_received_ns"
                        ],
                        f"{prefix}_image_path": camera_row["image_path"],
                    }
                )
            writer.writerow(output)
    return path, {
        "exact_sensor_frames": exact_count,
        "estimated_sensor_frames": estimated_count,
        "missing_sensor_frames": missing_count,
    }


def delete_source_images(session, camera_maps, sequence):
    deleted_files = 0
    deleted_bytes = 0
    for mapping in camera_maps:
        for capture_index in sequence:
            path = session / mapping[capture_index]["image_path"]
            size = path.stat().st_size
            path.unlink()
            deleted_files += 1
            deleted_bytes += size
    return deleted_files, deleted_bytes


def main():
    args = parse_args()
    session = args.session.expanduser().resolve()
    if not session.is_dir():
        raise SystemExit(f"Session does not exist: {session}")
    try:
        session_metadata_path = session / "session.json"
        session_metadata = (
            read_json(session_metadata_path)
            if session_metadata_path.is_file()
            else {}
        )
        camera_count = expected_camera_count(
            session, session_metadata, args.cameras
        )
        camera_dirs, camera_maps, sequence = load_cameras(session, camera_count)
        print(f"Cameras: {camera_count} ({[d.name for d in camera_dirs]})")
        if args.max_frames:
            sequence = sequence[: args.max_frames]
        sensor = load_sensor(session)
        force = load_force(session)
        camera_metadata = sorted(
            session_metadata.get("camera_stats", []),
            key=lambda item: int(item.get("camera_index", 0)),
        )
        saved_pixel_formats = [
            (
                str(camera_metadata[index].get("saved_pixel_format", ""))
                if index < len(camera_metadata)
                else ""
            )
            for index in range(camera_count)
        ]
        ffmpeg = find_ffmpeg()
        first_images = [
            session / mapping[sequence[0]]["image_path"]
            for mapping in camera_maps
        ]
        image_info = [inspect_image(path) for path in first_images]
        source_is_color = [
            info["color"] or pixel_format.startswith("Bayer")
            for info, pixel_format in zip(
                image_info, saved_pixel_formats
            )
        ]
        suffix = f"{args.fps:g}fps"
        if args.max_frames:
            suffix += f"_first{len(sequence)}"

        videos = [
            encode_camera(
                ffmpeg,
                session,
                directory,
                mapping,
                sequence,
                args.fps,
                args.crf,
                args.preset,
                args.overwrite,
                suffix,
                info,
                pixel_format,
            )
            for directory, mapping, info, pixel_format in zip(
                camera_dirs,
                camera_maps,
                image_info,
                saved_pixel_formats,
            )
        ]
        verified_counts = [
            decoded_frame_count(ffmpeg, video) for video in videos
        ]
        if any(count != len(sequence) for count in verified_counts):
            raise RuntimeError(
                f"Video verification mismatch: expected {len(sequence)}, "
                f"decoded {verified_counts}"
            )

        alignment, alignment_counts = write_alignment(
            session,
            camera_dirs,
            camera_maps,
            sequence,
            args.fps,
            sensor,
            force,
        )
        metadata_path = session / "multimodal_video_export.json"
        metadata = {
            "fps": args.fps,
            "camera_count": camera_count,
            "camera_directories": [directory.name for directory in camera_dirs],
            "frame_count": len(sequence),
            "capture_sequence_first": sequence[0],
            "capture_sequence_last": sequence[-1],
            "videos": [path.name for path in videos],
            "decoded_frame_counts": verified_counts,
            "source_image_info": image_info,
            "source_is_color": source_is_color,
            "saved_pixel_formats": saved_pixel_formats,
            "alignment": alignment.name,
            "sensor_file": sensor["path"].name if sensor else None,
            "force_file": force["path"].name if force else None,
            **alignment_counts,
            "source_images_deleted": False,
        }
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        if args.delete_images:
            deleted_files, deleted_bytes = delete_source_images(
                session, camera_maps, sequence
            )
            metadata["source_images_deleted"] = True
            metadata["deleted_image_files"] = deleted_files
            metadata["deleted_image_bytes"] = deleted_bytes
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            print(
                f"Deleted {deleted_files} verified source images "
                f"({deleted_bytes / 1024**3:.2f} GiB)"
            )

    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Videos: {[path.name for path in videos]}")
    print(f"Frames: {len(sequence)} at {args.fps:g} fps")
    print(f"Source color: {source_is_color}")
    print(f"Alignment: {alignment}")
    print(
        "Sensor mapping: "
        f"exact={alignment_counts['exact_sensor_frames']}, "
        f"estimated={alignment_counts['estimated_sensor_frames']}, "
        f"missing={alignment_counts['missing_sensor_frames']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
