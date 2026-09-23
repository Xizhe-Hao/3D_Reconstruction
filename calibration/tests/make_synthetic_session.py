#!/usr/bin/env python3
"""Render a fake capture session with known ground truth.

The output directory imitates a real `capture_3blackfly_sensor_force.py`
session down to the details that matter -- `camera_<i>_<serial>/` folders,
`frames.csv` keyed by `capture_sequence_index`, `session.json` announcing raw
`BayerRG8` BMPs -- so the calibration scripts can be exercised end to end
against an answer that is known exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from charuco_board import BoardSpec  # noqa: E402


def look_at(center: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)):
    """World -> camera transform for a camera at `center` aimed at `target`."""
    forward = target - center
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, float)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    return R, -R @ center


def make_texture(spec: BoardSpec, px_per_mm: float, margin_mm: float) -> np.ndarray:
    board = spec.build()
    px_per_square = int(round(spec.square_mm * px_per_mm))
    px_per_mm = px_per_square / spec.square_mm
    board_img = board.generateImage(
        (spec.squares_x * px_per_square, spec.squares_y * px_per_square),
        marginSize=0, borderBits=1)
    margin_px = int(round(margin_mm * px_per_mm))
    texture = np.full((board_img.shape[0] + 2 * margin_px,
                       board_img.shape[1] + 2 * margin_px), 245, np.uint8)
    texture[margin_px:margin_px + board_img.shape[0],
            margin_px:margin_px + board_img.shape[1]] = board_img
    return texture, px_per_mm, margin_px


def distortion_maps(K: np.ndarray, dist: np.ndarray, size: tuple[int, int]):
    """Maps that turn an ideal pinhole render into a lens-distorted image."""
    width, height = size
    grid = np.stack(np.meshgrid(np.arange(width, dtype=np.float32),
                                np.arange(height, dtype=np.float32)), axis=-1)
    undistorted = cv.undistortPoints(grid.reshape(-1, 1, 2), K, dist, None, None, K)
    undistorted = undistorted.reshape(height, width, 2)
    return undistorted[..., 0].copy(), undistorted[..., 1].copy()


def mosaic_bayer_rg(bgr: np.ndarray) -> np.ndarray:
    """GenICam BayerRG8 layout: row 0 is R G R G, row 1 is G B G B."""
    blue, green, red = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    raw = np.empty(bgr.shape[:2], np.uint8)
    raw[0::2, 0::2] = red[0::2, 0::2]
    raw[0::2, 1::2] = green[0::2, 1::2]
    raw[1::2, 0::2] = green[1::2, 0::2]
    raw[1::2, 1::2] = blue[1::2, 1::2]
    return raw


def random_board_pose(rng, board_size, radius=100.0, max_tilt_deg=42.0):
    """A board pose somewhere in the working volume, tilted but facing up.

    The spread matters: intrinsics are only constrained where the board has
    actually been, so the poses sweep the field of view and the depth range
    rather than hovering at one distance.
    """
    axis = rng.normal(size=3)
    axis[2] *= 0.15                       # keep the board roughly facing the cameras
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg))
    R = cv.Rodrigues(axis * angle)[0]
    center = np.array([board_size[0] / 2, board_size[1] / 2, 0.0])
    offset = np.array([rng.uniform(-radius, radius), rng.uniform(-radius, radius),
                       rng.uniform(-170.0, 40.0)])
    # rotate about the board centre so it stays inside the working volume
    t = center + offset - R @ center
    return R, t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--wave-frames", type=int, default=50)
    parser.add_argument("--world-frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--focal", type=float, default=1100.0)
    parser.add_argument("--radius", type=float, default=420.0)
    parser.add_argument("--noise", type=float, default=1.5, help="grey levels of sensor noise")
    parser.add_argument("--supersample", type=int, default=3,
                        help="render at this factor and area-average down; without it "
                             "the marker bits alias away and nothing is detected")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    spec = BoardSpec.load(args.board)
    board_size = spec.size_mm
    texture, px_per_mm, margin_px = make_texture(spec, 14.0, 12.0)
    target = np.array([board_size[0] / 2, board_size[1] / 2, 0.0])

    # ------------------------------------------------------------ true rig
    serials = [f"9000{i:04d}" for i in range(args.cameras)]
    azimuths = np.linspace(-62.0, 62.0, args.cameras)
    elevation = 52.0
    truth_cameras = []
    for ci, azimuth in enumerate(azimuths):
        az, el = np.radians(azimuth), np.radians(elevation)
        # The cameras sit on the board's NEGATIVE z side. That is not a quirk of
        # the simulation: a ChArUco board's own frame has x right and y down as
        # printed, so its +z axis points away from whoever is looking at the
        # printed face. Lay the board on the platform face-up with the cameras
        # above it and solvePnP puts them at negative z, exactly like this.
        # `calibrate.py --world-flip-z` is what turns that into the intuitive
        # "z = height above the array" frame.
        center = target + args.radius * np.array([np.sin(az) * np.cos(el),
                                                  -np.cos(az) * np.cos(el),
                                                  -np.sin(el)])
        R, t = look_at(center, target, up=(0.0, 0.0, -1.0))
        focal = args.focal * (1.0 + 0.02 * ci)
        K = np.array([[focal, 0.0, args.width / 2 - 6 + 3 * ci],
                      [0.0, focal * 1.001, args.height / 2 + 4 - 2 * ci],
                      [0.0, 0.0, 1.0]])
        dist = np.array([-0.14 + 0.01 * ci, 0.06, 3e-4, -2e-4, 0.0])
        truth_cameras.append({"serial": serials[ci], "K": K, "dist": dist, "R": R, "t": t})

    # Render at `supersample` times the sensor resolution, then area-average
    # down; that is what an image sensor does, and it keeps the marker bits
    # readable instead of aliasing them into noise.
    scale = max(1, args.supersample)
    big = (args.width * scale, args.height * scale)
    big_K = []
    for cam in truth_cameras:
        K = cam["K"].copy()
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 1] *= scale
        K[0, 2] = scale * K[0, 2] + (scale - 1) / 2.0
        K[1, 2] = scale * K[1, 2] + (scale - 1) / 2.0
        big_K.append(K)
    maps = [distortion_maps(big_K[ci], truth_cameras[ci]["dist"], big)
            for ci in range(args.cameras)]

    # ------------------------------------------------------------- frames
    poses = [(np.eye(3), np.zeros(3)) for _ in range(args.world_frames)]
    poses += [random_board_pose(rng, board_size) for _ in range(args.wave_frames)]

    # texture pixel -> board millimetre
    T = np.array([[1.0 / px_per_mm, 0.0, -margin_px / px_per_mm],
                  [0.0, 1.0 / px_per_mm, -margin_px / px_per_mm],
                  [0.0, 0.0, 1.0]])
    tint = np.array([0.92, 0.98, 1.00])  # warm white, so the Bayer path is exercised

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = {ci: [] for ci in range(args.cameras)}
    for ci in range(args.cameras):
        (out / f"camera_{ci}_{serials[ci]}").mkdir(exist_ok=True)

    for fi, (R_bw, t_bw) in enumerate(poses):
        for ci, cam in enumerate(truth_cameras):
            R_bc = cam["R"] @ R_bw
            t_bc = cam["R"] @ t_bw + cam["t"]
            H = big_K[ci] @ np.column_stack([R_bc[:, 0], R_bc[:, 1], t_bc]) @ T
            ideal = cv.warpPerspective(
                texture, H, big, flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT, borderValue=70)
            image = cv.remap(ideal, maps[ci][0], maps[ci][1], cv.INTER_LINEAR,
                             borderMode=cv.BORDER_CONSTANT, borderValue=70)
            image = cv.resize(image, (args.width, args.height), interpolation=cv.INTER_AREA)
            image = cv.GaussianBlur(image, (3, 3), 0.5)
            bgr = (image[:, :, None].astype(np.float32) * tint[None, None, :])
            if args.noise > 0:
                bgr += rng.normal(0.0, args.noise, bgr.shape)
            raw = mosaic_bayer_rg(np.clip(bgr, 0, 255).astype(np.uint8))

            name = f"capture_{fi:09d}_fid_{fi}.bmp"
            rel = f"camera_{ci}_{serials[ci]}/{name}"
            cv.imwrite(str(out / rel), raw)
            rows[ci].append({
                "host_received_ns": 1_000_000_000 + fi * 41_666_667,
                "capture_sequence_index": fi,
                "camera_frame_id": fi,
                "camera_timestamp": fi * 41_666_667,
                "complete": 1,
                "image_status": 0,
                "image_path": rel.replace("/", "\\"),
            })
        print(f"  rendered frame {fi + 1}/{len(poses)}", end="\r")
    print()

    fields = ["host_received_ns", "capture_sequence_index", "camera_frame_id",
              "camera_timestamp", "complete", "image_status", "image_path"]
    for ci in range(args.cameras):
        path = out / f"camera_{ci}_{serials[ci]}" / "frames.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[ci])

    (out / "session.json").write_text(json.dumps({
        "saved_utc": "2026-01-01T00:00:00+00:00",
        "camera_count": args.cameras,
        "camera_serials": serials,
        "image_format": "bmp",
        "color_requested": True,
        "demosaic_on_save": False,
        "trigger_source": "Line0",
        "camera_stats": [{"camera_index": ci, "serial": serials[ci],
                          "source_pixel_format": "BayerRG8",
                          "saved_pixel_format": "BayerRG8"}
                         for ci in range(args.cameras)],
        "synthetic": True,
    }, indent=2), encoding="utf-8")

    (out / "ground_truth.json").write_text(json.dumps({
        "world_frames": [0, args.world_frames],
        "image_size": [args.width, args.height],
        "cameras": [{"serial": cam["serial"],
                     "K": cam["K"].tolist(),
                     "dist": cam["dist"].tolist(),
                     "R_world_to_cam": cam["R"].tolist(),
                     "t_world_to_cam": cam["t"].tolist(),
                     "position_world": (-cam["R"].T @ cam["t"]).tolist()}
                    for cam in truth_cameras],
    }, indent=2), encoding="utf-8")

    print(f"wrote {len(poses)} frames x {args.cameras} cameras to {out}")
    print(f"world-frame segment: capture_sequence_index 0:{args.world_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
