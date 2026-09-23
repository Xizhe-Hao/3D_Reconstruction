"""Rigid-transform and triangulation helpers.

Pose convention used everywhere in this folder, matching OpenCV:

    X_cam = R @ X_world + t

so ``(R, t)`` is the *world -> camera* transform and the camera centre in world
coordinates is ``-R.T @ t``. `calibrateMultiview` returns ``Rs[c], Ts[c]`` that
map camera-0 coordinates into camera-c coordinates, and ``rvecs0/tvecs0`` that
map board coordinates into camera-0 coordinates; both follow the same rule.

Lengths are millimetres.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np


# --------------------------------------------------------------------- basics
def rodrigues(rvec: np.ndarray) -> np.ndarray:
    return cv.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]


def as_rotation_matrix(value: np.ndarray) -> np.ndarray:
    """Accept either a 3x3 matrix or a 3-vector and return a 3x3 matrix."""
    arr = np.asarray(value, np.float64)
    if arr.size == 9:
        return arr.reshape(3, 3)
    return rodrigues(arr)


def compose(R_ab: np.ndarray, t_ab: np.ndarray,
            R_bc: np.ndarray, t_bc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compose a->b then b->c into a->c."""
    R_ac = R_bc @ R_ab
    t_ac = R_bc @ np.asarray(t_ab, np.float64).reshape(3) + np.asarray(t_bc, np.float64).reshape(3)
    return R_ac, t_ac


def invert(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_inv = np.asarray(R, np.float64).T
    return R_inv, -R_inv @ np.asarray(t, np.float64).reshape(3)


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -np.asarray(R, np.float64).T @ np.asarray(t, np.float64).reshape(3)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    delta = np.asarray(R_a, np.float64).T @ np.asarray(R_b, np.float64)
    cos = (np.trace(delta) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def average_rigid(rotations, translations):
    """Average a cluster of rigid transforms and report how tight it is.

    The rotation mean is the closest orthonormal matrix to the arithmetic mean
    (SVD projection), which is exact enough for the tightly clustered poses we
    average here, and the spread is returned so a loose cluster -- meaning the
    'static' frames were not actually static -- is visible instead of silently
    averaged away.
    """
    rotations = [np.asarray(R, np.float64) for R in rotations]
    translations = [np.asarray(t, np.float64).reshape(3) for t in translations]
    if not rotations:
        raise ValueError("nothing to average")

    u, _, vt = np.linalg.svd(sum(rotations))
    R_mean = u @ vt
    if np.linalg.det(R_mean) < 0:
        u[:, -1] *= -1
        R_mean = u @ vt
    t_mean = np.mean(translations, axis=0)

    spread = {
        "count": len(rotations),
        "max_rotation_deg": max(rotation_angle_deg(R_mean, R) for R in rotations),
        "max_translation_mm": float(max(np.linalg.norm(t - t_mean) for t in translations)),
        "rms_translation_mm": float(np.sqrt(np.mean([np.sum((t - t_mean) ** 2)
                                                     for t in translations]))),
    }
    return R_mean, t_mean, spread


# ------------------------------------------------------------------ 2D <-> 3D
def undistort_to_pixels(points: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Remove lens distortion, keeping the result in pixel units of the same K."""
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 1e-8)
    out = cv.undistortPoints(pts, np.asarray(K, np.float64),
                             np.asarray(dist, np.float64).reshape(1, -1),
                             None, None, np.asarray(K, np.float64), criteria)
    return out.reshape(-1, 2)


def triangulate(observations, refine_iterations: int = 10) -> np.ndarray:
    """Triangulate one 3D point from N >= 2 views.

    ``observations`` is a sequence of ``(K, R, t, uv_undistorted)`` where ``uv``
    is already free of lens distortion and expressed in pixels. A linear DLT
    provides the seed and a few Gauss-Newton steps minimise the true pixel
    re-projection error, which is what the validation metrics are quoted in.
    """
    rows = []
    for K, R, t, uv in observations:
        P = np.asarray(K, np.float64) @ np.hstack([np.asarray(R, np.float64),
                                                   np.asarray(t, np.float64).reshape(3, 1)])
        u, v = float(uv[0]), float(uv[1])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-12:
        return np.full(3, np.nan)
    point = homogeneous[:3] / homogeneous[3]

    for _ in range(refine_iterations):
        JtJ = np.zeros((3, 3))
        Jtr = np.zeros(3)
        for K, R, t, uv in observations:
            K = np.asarray(K, np.float64)
            R = np.asarray(R, np.float64)
            cam = R @ point + np.asarray(t, np.float64).reshape(3)
            if cam[2] <= 1e-9:
                return point
            fx, fy = K[0, 0], K[1, 1]
            skew = K[0, 1]
            proj = np.array([
                (fx * cam[0] + skew * cam[1]) / cam[2] + K[0, 2],
                fy * cam[1] / cam[2] + K[1, 2],
            ])
            residual = proj - np.asarray(uv, np.float64)
            inv_z = 1.0 / cam[2]
            d_proj_d_cam = np.array([
                [fx * inv_z, skew * inv_z, -(fx * cam[0] + skew * cam[1]) * inv_z ** 2],
                [0.0, fy * inv_z, -fy * cam[1] * inv_z ** 2],
            ])
            J = d_proj_d_cam @ R
            JtJ += J.T @ J
            Jtr += J.T @ residual
        try:
            step = np.linalg.solve(JtJ + 1e-9 * np.eye(3), Jtr)
        except np.linalg.LinAlgError:
            break
        point = point - step
        if np.linalg.norm(step) < 1e-9:
            break
    return point


def kabsch(source: np.ndarray, target: np.ndarray):
    """Best rigid transform mapping ``source`` onto ``target`` (no scaling).

    Used to compare a triangulated point cloud against the known board
    geometry: the residual after alignment is a direct, metric statement of 3D
    accuracy in millimetres.
    """
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)
    H = (source - src_mean).T @ (target - dst_mean)
    u, _, vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    R = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = dst_mean - R @ src_mean
    residuals = np.linalg.norm((source @ R.T + t) - target, axis=1)
    return R, t, residuals
