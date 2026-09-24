"""Adapter for the synchronized four-camera ``data/test`` session.

The MVTracker tensors produced here follow the upstream convention:

* RGB: ``[V, T, 3, H, W]``, uint8 RGB
* intrinsics: ``[V, T, 3, 3]``
* extrinsics: ``[V, T, 3, 4]``, world-to-camera
* translations and depths: metres

Images are undistorted before resizing.  The original calibrated intrinsic
matrix is retained during undistortion, then scaled to the requested output
resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm.auto import tqdm


@dataclass(frozen=True)
class CameraSpec:
    index: int
    serial: str
    video_path: Path
    image_size: tuple[int, int]
    intrinsic: np.ndarray
    distortion: np.ndarray
    extrinsic_w2c_m: np.ndarray


@dataclass
class SessionClip:
    rgbs: np.ndarray
    intrinsics: np.ndarray
    extrinsics_w2c_m: np.ndarray
    frame_indices: np.ndarray
    video_times_s: np.ndarray
    camera_serials: list[str]
    source_size: tuple[int, int]


class SessionFormatError(RuntimeError):
    pass


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise SessionFormatError(f"Missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionFormatError(f"Cannot read JSON {path}: {exc}") from exc


def _select_indices(
    frame_count: int, start: int, end: int | None, step: int, target_frames: int | None = None
) -> list[int]:
    if step <= 0:
        raise SessionFormatError("--step must be positive")
    if end is None:
        end = frame_count - 1 if target_frames is not None else min(frame_count - 1, start + 23 * step)
    if start < 0 or end < start or end >= frame_count:
        raise SessionFormatError(
            f"Invalid frame range [{start}, {end}] for {frame_count} synchronized frames"
        )
    if target_frames is not None:
        available = end - start + 1
        if target_frames < 2:
            raise SessionFormatError("--target-frames must be at least 2")
        if target_frames > available:
            raise SessionFormatError(
                f"--target-frames={target_frames} exceeds the {available} source frames in the selected range"
            )
        # Rounded linspace is strictly increasing when target_frames <= available.
        return np.rint(np.linspace(start, end, target_frames)).astype(np.int32).tolist()
    return list(range(start, end + 1, step))


def load_session_metadata(session_dir: Path) -> tuple[list[CameraSpec], list[dict], dict]:
    session_dir = session_dir.expanduser().resolve()
    export = _read_json(session_dir / "multimodal_video_export.json")
    calibration = _read_json(session_dir / "calibration.json")
    alignment_path = session_dir / export.get("alignment", "multimodal_video_alignment.csv")
    if not alignment_path.is_file():
        raise SessionFormatError(f"Missing alignment CSV: {alignment_path}")

    with alignment_path.open("r", newline="", encoding="utf-8-sig") as handle:
        alignment = list(csv.DictReader(handle))
    frame_count = int(export.get("frame_count", -1))
    if frame_count <= 0 or len(alignment) != frame_count:
        raise SessionFormatError(
            f"Alignment has {len(alignment)} rows but export declares {frame_count} frames"
        )
    if int(export.get("camera_count", -1)) != 4:
        raise SessionFormatError("This adapter expects exactly four synchronized cameras")
    if calibration.get("pose_convention") != "X_cam = R_world_to_cam @ X_world + t_world_to_cam":
        raise SessionFormatError(f"Unsupported pose convention: {calibration.get('pose_convention')!r}")
    if calibration.get("units") not in {"millimetre", "millimeter", "mm"}:
        raise SessionFormatError(f"Expected millimetre calibration, got {calibration.get('units')!r}")

    videos = export.get("videos", [])
    calib_by_index = {int(cam["index"]): cam for cam in calibration.get("cameras", [])}
    if len(videos) != 4 or set(calib_by_index) != set(range(4)):
        raise SessionFormatError("Video list and calibration must both contain camera indices 0..3")

    cameras: list[CameraSpec] = []
    for index, video_name in enumerate(videos):
        cam = calib_by_index[index]
        serial = str(cam["serial"])
        if serial not in video_name:
            raise SessionFormatError(f"Camera {index} serial {serial} does not match {video_name}")
        video_path = session_dir / video_name
        if not video_path.is_file():
            raise SessionFormatError(f"Missing camera video: {video_path}")
        R = np.asarray(cam["R_world_to_cam"], dtype=np.float32)
        t_m = np.asarray(cam["t_world_to_cam"], dtype=np.float32) / 1000.0
        cameras.append(
            CameraSpec(
                index=index,
                serial=serial,
                video_path=video_path,
                image_size=tuple(map(int, cam["image_size"])),
                intrinsic=np.asarray(cam["K"], dtype=np.float32),
                distortion=np.asarray(cam["dist"], dtype=np.float32),
                extrinsic_w2c_m=np.concatenate([R, t_m[:, None]], axis=1),
            )
        )
    return cameras, alignment, export


def _decode_camera(
    camera: CameraSpec,
    frame_indices: Iterable[int],
    output_size: tuple[int, int],
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    output_w, output_h = output_size
    source_w, source_h = camera.image_size
    cap = cv2.VideoCapture(str(camera.video_path))
    if not cap.isOpened():
        raise SessionFormatError(f"OpenCV cannot open {camera.video_path}")
    actual_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if actual_size != camera.image_size:
        cap.release()
        raise SessionFormatError(
            f"{camera.video_path.name} is {actual_size}, calibration expects {camera.image_size}"
        )

    map_x, map_y = cv2.initUndistortRectifyMap(
        camera.intrinsic,
        camera.distortion,
        None,
        camera.intrinsic,
        camera.image_size,
        cv2.CV_32FC1,
    )
    frames = []
    try:
        selected = list(frame_indices)
        iterator = tqdm(
            selected,
            desc=f"Decode camera {camera.index} ({camera.serial})",
            unit="frame",
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for frame_index in iterator:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise SessionFormatError(
                    f"Failed decoding synchronized frame {frame_index} from {camera.video_path.name}"
                )
            bgr = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            bgr = cv2.resize(bgr, output_size, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
    finally:
        cap.release()

    K = camera.intrinsic.copy()
    K[0, :] *= output_w / source_w
    K[1, :] *= output_h / source_h
    return np.stack(frames).astype(np.uint8), K


def load_session_clip(
    session_dir: Path,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    output_size: tuple[int, int] = (512, 384),
    max_frames: int = 96,
    target_frames: int | None = None,
    show_progress: bool = False,
) -> SessionClip:
    cameras, alignment, export = load_session_metadata(session_dir)
    frame_indices = _select_indices(int(export["frame_count"]), start, end, step, target_frames)
    if len(frame_indices) > max_frames:
        raise SessionFormatError(
            f"Selected {len(frame_indices)} frames; limit is {max_frames}. Increase --step or shorten the range."
        )

    rgb_views, intrinsics = [], []
    for camera in cameras:
        rgb, K = _decode_camera(camera, frame_indices, output_size, show_progress)
        rgb_views.append(rgb)
        intrinsics.append(K)
    rgbs = np.stack(rgb_views)
    V, T = rgbs.shape[:2]
    intrs = np.repeat(np.stack(intrinsics)[:, None], T, axis=1).astype(np.float32)
    extrs = np.repeat(
        np.stack([camera.extrinsic_w2c_m for camera in cameras])[:, None], T, axis=1
    ).astype(np.float32)
    times = np.asarray([float(alignment[i]["video_time_s"]) for i in frame_indices], dtype=np.float64)
    return SessionClip(
        rgbs=rgbs,
        intrinsics=intrs,
        extrinsics_w2c_m=extrs,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        video_times_s=times,
        camera_serials=[camera.serial for camera in cameras],
        source_size=cameras[0].image_size,
    )


def save_clip(path: Path, clip: SessionClip) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rgbs=clip.rgbs,
        intrinsics=clip.intrinsics,
        extrinsics_w2c_m=clip.extrinsics_w2c_m,
        frame_indices=clip.frame_indices,
        video_times_s=clip.video_times_s,
        camera_serials=np.asarray(clip.camera_serials),
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--target-frames", type=int)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--output", type=Path, default=Path("outputs/test_session_clip.npz"))
    args = parser.parse_args()
    clip = load_session_clip(
        args.session_dir, args.start, args.end, args.step, (args.width, args.height), args.max_frames,
        args.target_frames,
        not args.no_progress,
    )
    output = save_clip(args.output, clip)
    print(
        f"Saved {output}: rgbs={clip.rgbs.shape}, intrinsics={clip.intrinsics.shape}, "
        f"extrinsics={clip.extrinsics_w2c_m.shape}, frames={clip.frame_indices.tolist()}"
    )


if __name__ == "__main__":
    main()
