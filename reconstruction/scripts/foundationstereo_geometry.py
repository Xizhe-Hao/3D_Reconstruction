"""Geometry utilities shared by the FoundationStereo backend and worker."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RectifiedPair:
    left: int
    right: int
    maps: dict[int, tuple[np.ndarray, np.ndarray]]
    intrinsics: dict[int, np.ndarray]
    extrinsics: dict[int, np.ndarray]
    rectification_rotations: dict[int, np.ndarray]
    baseline_m: float


def parse_pairs(value: str, views: int = 4) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    try:
        for item in value.split(","):
            a, b = (int(part) for part in item.split("-"))
            if a == b or a not in range(views) or b not in range(views):
                raise ValueError
            if a in used or b in used:
                raise ValueError
            pairs.append((a, b))
            used.update((a, b))
    except ValueError as exc:
        raise ValueError("pairs must be disjoint camera pairs such as 0-2,1-3") from exc
    if used != set(range(views)):
        raise ValueError(f"pairs must cover camera indices 0..{views - 1} exactly once")
    return pairs


def _relative_pose(extr_a: np.ndarray, extr_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ra, ta = extr_a[:3, :3], extr_a[:3, 3]
    rb, tb = extr_b[:3, :3], extr_b[:3, 3]
    rotation = rb @ ra.T
    translation = tb - rotation @ ta
    return rotation.astype(np.float64), translation.astype(np.float64).reshape(3, 1)


def build_rectified_pair(
    cameras, first: int, second: int, image_size: tuple[int, int], alpha: float = -1.0
) -> RectifiedPair:
    """Build a horizontal stereo pair, swapping order when needed for positive disparity."""
    for left, right in ((first, second), (second, first)):
        ca, cb = cameras[left], cameras[right]
        rotation, translation = _relative_pose(ca.extrinsic_w2c_m, cb.extrinsic_w2c_m)
        r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
            ca.intrinsic.astype(np.float64), ca.distortion.astype(np.float64),
            cb.intrinsic.astype(np.float64), cb.distortion.astype(np.float64),
            image_size, rotation, translation, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
        )
        if p2[0, 3] <= 0:
            break
    else:
        raise ValueError(f"Could not orient stereo pair {first}-{second} for positive disparity")
    ka, kb = p1[:3, :3].astype(np.float32), p2[:3, :3].astype(np.float32)
    maps = {
        left: cv2.initUndistortRectifyMap(ca.intrinsic, ca.distortion, r1, ka, image_size, cv2.CV_32FC1),
        right: cv2.initUndistortRectifyMap(cb.intrinsic, cb.distortion, r2, kb, image_size, cv2.CV_32FC1),
    }
    extrinsics = {}
    for index, rectification, camera in ((left, r1, ca), (right, r2, cb)):
        e = camera.extrinsic_w2c_m
        extrinsics[index] = np.concatenate(
            [rectification @ e[:3, :3], (rectification @ e[:3, 3])[:, None]], axis=1
        ).astype(np.float32)
    return RectifiedPair(
        left=left, right=right, maps=maps,
        intrinsics={left: ka, right: kb}, extrinsics=extrinsics,
        rectification_rotations={left: r1.astype(np.float32), right: r2.astype(np.float32)},
        baseline_m=float(abs(p2[0, 3] / p2[0, 0])),
    )


def unflip_reverse_disparity(disparity: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.fliplr(disparity))


def scale_intrinsics(intrinsic: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = np.asarray(intrinsic, dtype=np.float32).copy()
    scaled[0, :] *= scale_x
    scaled[1, :] *= scale_y
    return scaled


def disparity_to_depth(disparity: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 1e-6)
    depth[valid] = float(focal_px * baseline_m) / disparity[valid]
    return depth


def left_right_masks(left: np.ndarray, right: np.ndarray, threshold_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Return consistency masks for disparities defined on left and right images."""
    height, width = left.shape
    yy, xx = np.mgrid[:height, :width]
    xr = np.rint(xx - left).astype(np.int32)
    xl = np.rint(xx + right).astype(np.int32)
    left_ok = np.isfinite(left) & (left > 0) & (xr >= 0) & (xr < width)
    right_ok = np.isfinite(right) & (right > 0) & (xl >= 0) & (xl < width)
    sampled_right = np.zeros_like(left)
    sampled_left = np.zeros_like(right)
    sampled_right[left_ok] = right[yy[left_ok], xr[left_ok]]
    sampled_left[right_ok] = left[yy[right_ok], xl[right_ok]]
    left_ok &= np.isfinite(sampled_right) & (sampled_right > 0) & (np.abs(left - sampled_right) <= threshold_px)
    right_ok &= np.isfinite(sampled_left) & (sampled_left > 0) & (np.abs(right - sampled_left) <= threshold_px)
    return left_ok, right_ok


def downsample_depth_median(depth: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Validity-aware median downsampling for integer scale factors."""
    out_w, out_h = output_size
    height, width = depth.shape
    if width % out_w or height % out_h:
        raise ValueError(f"Depth size {(width, height)} is not divisible by {output_size}")
    sy, sx = height // out_h, width // out_w
    blocks = depth.reshape(out_h, sy, out_w, sx).transpose(0, 2, 1, 3).reshape(out_h, out_w, sy * sx)
    blocks = np.where(np.isfinite(blocks) & (blocks > 0), blocks, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        result = np.nanmedian(blocks, axis=-1)
    return np.nan_to_num(result, nan=0.0).astype(np.float32)
