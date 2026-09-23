"""Final refinement over EVERY detected corner, not just the complete ones.

`cv::calibrateMultiview` decides what to use with a `detectionMask` indexed by
(camera, frame). Visibility is therefore a property of a whole view: either a
camera contributed all of the pattern in that frame or none of it. A ChArUco
target seen at an angle is almost never complete, so `calibrate.py` keeps the
observations honest by shrinking the pattern until whole cells are valid --
correct, but it means most of what was measured never reaches the solver. On one
session here: 27252 corners detected, ~11000 used by the strict solve, and only
6520 under `--common-corners 40`.

Nothing about the geometry requires that. A corner seen by one camera in one
frame is a perfectly good constraint on that camera's pose and that frame's
board pose; it is only the API that cannot express it. So this takes the
multi-view result as a starting point and runs one joint Levenberg-Marquardt
over the whole set:

    parameters   6 per camera (all but camera 0, which fixes the gauge)
                 6 per frame  (the board's pose in that frame)
    residuals    2 per detected corner

The Jacobian is extremely sparse -- a corner in frame f seen by camera c touches
only those two blocks -- so the sparsity pattern is handed to the solver and a
few hundred parameters against fifty thousand residuals costs seconds.

What this does and does not buy:

* it uses every measurement, so cameras that rarely see the whole board stop
  being under-represented (one session had a camera contributing 11 frames to
  the solve while it had detected corners in 57);
* it cannot repair a rig that moved during the session. A single rigid set of
  extrinsics is the wrong model for that, and the residual it leaves is the
  honest size of the change. Fix the mounts; this is not a substitute.
"""

from __future__ import annotations

import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _rt_to_vec(R, t):
    rvec, _ = cv.Rodrigues(np.asarray(R, np.float64))
    return np.concatenate([rvec.ravel(), np.asarray(t, np.float64).ravel()])


def _vec_to_rt(vec):
    R, _ = cv.Rodrigues(np.asarray(vec[:3], np.float64))
    return R, np.asarray(vec[3:6], np.float64)


def _project(points_board, board_vec, cam_vec, K, dist):
    """Board points -> pixels, through the board pose then the camera pose."""
    R_b, t_b = _vec_to_rt(board_vec)
    R_c, t_c = _vec_to_rt(cam_vec)
    world = points_board @ R_b.T + t_b
    rvec, _ = cv.Rodrigues(R_c)
    uv, _ = cv.projectPoints(world.reshape(-1, 1, 3), rvec, t_c, K, dist)
    return uv.reshape(-1, 2)


def seed_board_poses(corners, object_points, Ks, dists, Rs, Ts):
    """One board pose per frame, from whichever camera saw the most of it.

    A per-frame pose has to exist before the joint solve can start, and the
    single best view gives a seed that is close enough for LM to take over.
    """
    num_cams, num_frames = corners.shape[0], corners.shape[1]
    poses, valid = np.zeros((num_frames, 6)), np.zeros(num_frames, bool)
    for fi in range(num_frames):
        best, best_count = None, 0
        for ci in range(num_cams):
            seen = corners[ci, fi, :, 0] >= 0
            if seen.sum() < 8 or seen.sum() <= best_count:
                continue
            ok, rvec, tvec = cv.solvePnP(
                object_points[seen].astype(np.float64),
                corners[ci, fi, seen].astype(np.float64),
                Ks[ci], dists[ci], flags=cv.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            # solvePnP gives the board in CAMERA coordinates; lift it to world.
            R_cb, _ = cv.Rodrigues(rvec)
            R_wc, t_wc = np.asarray(Rs[ci], np.float64), np.asarray(Ts[ci], np.float64).ravel()
            R_wb = R_wc.T @ R_cb
            t_wb = R_wc.T @ (tvec.ravel() - t_wc)
            best, best_count = _rt_to_vec(R_wb, t_wb), int(seen.sum())
        if best is not None:
            poses[fi], valid[fi] = best, True
    return poses, valid


def refine(corners, object_points, Ks, dists, Rs, Ts, max_iterations=60,
           verbose=True):
    """Joint LM over every detected corner. Returns refined (Rs, Ts, stats)."""
    num_cams, num_frames = corners.shape[0], corners.shape[1]
    Ks = [np.asarray(K, np.float64) for K in Ks]
    dists = [np.asarray(d, np.float64).reshape(1, -1) for d in dists]

    board_poses, valid = seed_board_poses(corners, object_points, Ks, dists, Rs, Ts)
    frames = np.flatnonzero(valid)
    frame_slot = {int(f): i for i, f in enumerate(frames)}

    # Every (camera, frame, corner) that was really measured.
    obs = []
    for ci in range(num_cams):
        for fi in frames:
            seen = np.flatnonzero(corners[ci, fi, :, 0] >= 0)
            if seen.size:
                obs.append((ci, int(fi), seen))
    total = sum(len(s) for _, _, s in obs)
    if verbose:
        print(f"   refining on {total} detected corners "
              f"({len(obs)} views, {len(frames)} frames)")

    cam_vecs = np.array([_rt_to_vec(Rs[ci], Ts[ci]) for ci in range(num_cams)])
    # Camera 0 is held fixed: the rig's absolute pose is not observable, and
    # leaving it free lets the whole solution drift along that null space.
    x0 = np.concatenate([cam_vecs[1:].ravel(), board_poses[frames].ravel()])
    cam_block = (num_cams - 1) * 6

    def unpack(x):
        cams = np.vstack([cam_vecs[0], x[:cam_block].reshape(num_cams - 1, 6)])
        return cams, x[cam_block:].reshape(len(frames), 6)

    def residuals(x):
        cams, boards = unpack(x)
        out = np.empty(total * 2)
        at = 0
        for ci, fi, seen in obs:
            predicted = _project(object_points[seen], boards[frame_slot[fi]],
                                 cams[ci], Ks[ci], dists[ci])
            delta = (predicted - corners[ci, fi, seen]).ravel()
            out[at:at + delta.size] = delta
            at += delta.size
        return out

    sparsity = lil_matrix((total * 2, x0.size), dtype=np.uint8)
    at = 0
    for ci, fi, seen in obs:
        rows = slice(at, at + len(seen) * 2)
        if ci > 0:
            sparsity[rows, (ci - 1) * 6:ci * 6] = 1
        slot = frame_slot[fi]
        sparsity[rows, cam_block + slot * 6:cam_block + (slot + 1) * 6] = 1
        at += len(seen) * 2

    before = float(np.sqrt(np.mean(residuals(x0) ** 2)))
    # x_scale="jac" is not optional here: translations are in millimetres
    # (order 200) and rotations in radians (order 1), so an unscaled trust
    # region takes steps that are enormous for one and negligible for the
    # other -- the translations simply never move.
    result = least_squares(residuals, x0, jac_sparsity=sparsity.tocsr(),
                           method="trf", loss="huber", f_scale=2.0,
                           x_scale="jac", max_nfev=max_iterations, verbose=0)
    after = float(np.sqrt(np.mean(result.fun ** 2)))

    cams, boards = unpack(result.x)
    Rs_out, Ts_out = [], []
    for ci in range(num_cams):
        R, t = _vec_to_rt(cams[ci])
        Rs_out.append(R)
        Ts_out.append(t)
    if verbose:
        print(f"   reprojection rms over ALL corners: {before:.4f} -> {after:.4f} px")
    return Rs_out, Ts_out, {"observations": total, "rms_before": before,
                            "rms_after": after, "frames": len(frames)}
