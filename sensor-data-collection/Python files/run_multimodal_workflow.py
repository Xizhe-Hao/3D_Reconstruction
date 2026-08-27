#!/usr/bin/env python3
"""Run one complete camera/sensor acquisition and post-processing workflow."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CAMERA_COUNT = 4
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "captures"
DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 500_000
IMAGE_SUFFIXES = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".raw"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture four Blackfly S cameras and a 16x16 tactile sensor, "
            "then repair, export, verify, and clean the session."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory in which new capture sessions are created.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help="Arduino serial port, for example COM7.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help="Arduino serial baud rate.",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=DEFAULT_CAMERA_COUNT,
        help=(
            "Number of Blackfly S cameras on the rig "
            f"(default {DEFAULT_CAMERA_COUNT})."
        ),
    )
    parser.add_argument(
        "--camera-serials",
        nargs="+",
        metavar="SERIAL",
        help=(
            "Optional camera serial numbers in the desired output order. "
            "Supply exactly --cameras values."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop capture after N sensor frames; 0 runs until Ctrl+C.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--preset",
        default="fast",
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
    parser.add_argument(
        "--keep-bmp",
        action="store_true",
        help="Keep source camera images after every video passes verification.",
    )
    parser.add_argument(
        "--monitor-force-export",
        action="store_true",
        help=(
            "Use the legacy same-PC IntelliMESUR export monitor. By default "
            "force data is imported later with align_force_curve_only.py."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned capture command without running it.",
    )
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.cameras < 1:
        parser.error("--cameras must be at least 1")
    if args.camera_serials and len(args.camera_serials) != args.cameras:
        parser.error(
            f"--camera-serials expects {args.cameras} serial numbers, "
            f"received {len(args.camera_serials)}"
        )
    if args.baud <= 0:
        parser.error("--baud must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    return args


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def command_text(command):
    return subprocess.list2cmdline([str(item) for item in command])


def run_command(name, command, cwd, dry_run=False):
    print(f"\n[{name}]\n{command_text(command)}", flush=True)
    if dry_run:
        return

    process = subprocess.Popen(command, cwd=cwd)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        print(
            "\nInterrupt received; waiting for the active step to stop safely...",
            flush=True,
        )
        try:
            return_code = process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                return_code = process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{name} failed with exit code {return_code}")


def require_scripts(script_dir):
    scripts = {
        "capture": script_dir / "capture_3blackfly_sensor_force.py",
        "repair": script_dir / "repair_sensor_session.py",
        "export": script_dir / "export_multimodal_mp4.py",
    }
    missing = [str(path) for path in scripts.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing workflow scripts: {missing}")
    return scripts


def capture_sessions(output):
    if not output.is_dir():
        return set()
    return {
        path.resolve()
        for path in output.glob("capture_*")
        if path.is_dir()
    }


def load_valid_video_export(session, camera_count):
    metadata_path = session / "multimodal_video_export.json"
    if not metadata_path.is_file():
        return None
    metadata = read_json(metadata_path)
    frame_count = int(metadata.get("frame_count", 0))
    videos = [session / str(name) for name in metadata.get("videos", [])]
    decoded = [
        int(value) for value in metadata.get("decoded_frame_counts", [])
    ]
    if (
        frame_count < 1
        or len(videos) != camera_count
        or len(decoded) != camera_count
        or any(value != frame_count for value in decoded)
        or any(not path.is_file() for path in videos)
        or not (session / "multimodal_video_alignment.csv").is_file()
    ):
        return None
    return metadata


def camera_image_paths(session, camera_count):
    session_resolved = session.resolve()
    paths = set()
    camera_dirs = sorted(
        path for path in session.glob("camera_*") if path.is_dir()
    )
    if len(camera_dirs) != camera_count:
        raise RuntimeError(
            f"Expected {camera_count} camera directories, "
            f"found {len(camera_dirs)}"
        )
    for camera_dir in camera_dirs:
        frame_log = camera_dir / "frames.csv"
        if not frame_log.is_file():
            raise RuntimeError(f"Missing camera frame log: {frame_log}")
        for row in read_csv(frame_log):
            relative = row.get("image_path", "").strip()
            if not relative:
                continue
            path = (session / relative).resolve()
            try:
                path.relative_to(session_resolved)
            except ValueError as exc:
                raise RuntimeError(
                    f"Unsafe image path outside session: {path}"
                ) from exc
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                raise RuntimeError(f"Unexpected source image type: {path}")
            if path.is_file():
                paths.add(path)
    return sorted(paths)


def clean_source_images(session, camera_count):
    metadata = load_valid_video_export(session, camera_count)
    if metadata is None:
        raise RuntimeError(
            "Source images cannot be deleted because MP4 verification "
            "metadata is missing or invalid"
        )
    paths = camera_image_paths(session, camera_count)
    total_bytes = sum(path.stat().st_size for path in paths)
    print(
        f"\n[cleanup] verified outputs; deleting {len(paths)} source images "
        f"({total_bytes / 1024**3:.2f} GiB)",
        flush=True,
    )

    deleted_files = 0
    deleted_bytes = 0
    for path in paths:
        size = path.stat().st_size
        path.unlink()
        deleted_files += 1
        deleted_bytes += size

    metadata["source_images_deleted"] = True
    metadata["deleted_image_files"] = (
        int(metadata.get("deleted_image_files", 0)) + deleted_files
    )
    metadata["deleted_image_bytes"] = (
        int(metadata.get("deleted_image_bytes", 0)) + deleted_bytes
    )
    write_json_atomic(session / "multimodal_video_export.json", metadata)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    scripts = require_scripts(script_dir)
    python = Path(sys.executable).resolve()
    output = args.output.expanduser().resolve()

    capture_command = [
        python,
        scripts["capture"],
        "--output",
        output,
        "--port",
        args.port,
        "--baud",
        str(args.baud),
        "--cameras",
        str(args.cameras),
    ]
    if args.frames:
        capture_command.extend(("--frames", str(args.frames)))
    if args.camera_serials:
        capture_command.append("--camera-serials")
        capture_command.extend(args.camera_serials)
    if not args.monitor_force_export:
        capture_command.append("--no-force-export-monitor")

    if args.dry_run:
        run_command("capture", capture_command, script_dir, dry_run=True)
        action = (
            "retain source images"
            if args.keep_bmp
            else "delete source images after MP4 verification"
        )
        print(
            "\nAfter capture: repair -> MP4/camera-sensor alignment -> "
            f"{action}. Force data is imported later with "
            "align_force_curve_only.py.",
            flush=True,
        )
        return 0

    output.mkdir(parents=True, exist_ok=True)
    before = capture_sessions(output)
    run_command("capture", capture_command, script_dir)
    created = capture_sessions(output) - before
    if len(created) != 1:
        raise RuntimeError(
            f"Expected one new capture session, found {len(created)}"
        )
    session = next(iter(created))
    print(f"\nWorkflow session: {session}", flush=True)

    run_command(
        "repair",
        [
            python,
            scripts["repair"],
            session,
            "--fill-missing",
        ],
        script_dir,
    )
    run_command(
        "video export",
        [
            python,
            scripts["export"],
            session,
            "--fps",
            str(args.fps),
            "--preset",
            args.preset,
            "--crf",
            str(args.crf),
            "--cameras",
            str(args.cameras),
            "--overwrite",
        ],
        script_dir,
    )
    if load_valid_video_export(session, args.cameras) is None:
        raise RuntimeError(
            "Video export returned success but verification metadata "
            "is incomplete"
        )

    if args.keep_bmp:
        print("\n[cleanup] --keep-bmp selected; source images retained")
    else:
        clean_source_images(session, args.cameras)

    print(
        "\nCamera/sensor processing completed. Import the corresponding "
        f"force CSV with align_force_curve_only.py:\n{session}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nWorkflow failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
