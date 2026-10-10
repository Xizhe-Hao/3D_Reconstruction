"""Where the cameras stand around the board, and which of them are neighbours.

Used by `capture_gui.py` to plan the capture from the rig in front of it rather
than from camera numbering. The two can disagree completely: on this rig
cameras 0 and 1 face each other across the array (180 deg apart), as do 2 and
3, while the genuinely adjacent pairs are 0+2 (50 deg) and 1+3 (44 deg). The
old fixed plan asked for 0+1, 1+2 and 2+3 together, so two of its three pair
steps requested a pose no flat board can give both cameras -- which is why the
2026-09-30 session ended up solving on an 80-corner sub-pattern.

No PySpin here, so the planning can be tested in the calibration environment.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

# Two cameras further apart than this, around the board, cannot both see one
# flat board well; a pose that splits the difference is seen edge-on by both.
MAX_PAIR_GAP_DEG = 100.0


def nominal_K(width: int, height: int, focal_px: float | None = None) -> np.ndarray:
    """A camera matrix good enough to tell which side of the board a camera is on.

    Azimuth barely depends on the focal length, so before any calibration a
    rough value is fine; 0.85 x the image width is what this rig's lenses give.
    """
    f = focal_px or 0.85 * width
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def camera_azimuth(object_points: np.ndarray, ids: np.ndarray, corners: np.ndarray,
                   K: np.ndarray, dist: np.ndarray | None = None) -> float | None:
    """Azimuth of the camera around the board centre, degrees, in the board plane."""
    if ids is None or len(ids) < 8:
        return None
    obj = np.ascontiguousarray(object_points[ids], np.float64)
    img = np.ascontiguousarray(corners, np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv.solvePnP(obj, img, K, dist if dist is not None else np.zeros(5))
    if not ok:
        return None
    R, _ = cv.Rodrigues(rvec)
    centre = -R.T @ tvec.reshape(3)                  # camera centre in board frame
    middle = object_points.mean(axis=0)
    return float(np.degrees(np.arctan2(centre[1] - middle[1], centre[0] - middle[0])) % 360.0)


def neighbour_pairs(azimuths: dict[int, float],
                    max_gap: float = MAX_PAIR_GAP_DEG) -> list[tuple[int, int]]:
    """Pairs of cameras that are next to each other around the board.

    Cameras are sorted by azimuth and each is paired with its neighbour on the
    ring when the gap is small enough for one board pose to face both.
    """
    order = sorted(azimuths, key=lambda index: azimuths[index])
    if len(order) < 2:
        return []
    pairs = []
    ring = list(zip(order, order[1:] + order[:1])) if len(order) > 2 else [tuple(order)]
    for a, b in ring:
        gap = (azimuths[b] - azimuths[a]) % 360.0
        gap = min(gap, 360.0 - gap)
        if gap <= max_gap:
            pairs.append(tuple(sorted((a, b))))
    return sorted(set(pairs))


def unpaired(cameras: list[int], pairs: list[tuple[int, int]]) -> list[int]:
    """Cameras that no pair step will capture square-on; they get a face step."""
    covered = {index for pair in pairs for index in pair}
    return [index for index in cameras if index not in covered]


# ------------------------------------------------------------------ coverage
GRID = 3            # the image is judged on a 3 x 3 grid
COVERAGE_GOAL = 8   # of 9 cells; the corners of the image are where distortion lives


def coverage_cells(corners: np.ndarray, width: int, height: int,
                   min_corners: int = 4) -> set[tuple[int, int]]:
    """Grid cells that this view puts at least `min_corners` board corners in."""
    if corners is None or len(corners) == 0:
        return set()
    col = np.clip((corners[:, 0] * GRID // width).astype(int), 0, GRID - 1)
    row = np.clip((corners[:, 1] * GRID // height).astype(int), 0, GRID - 1)
    cells, counts = np.unique(np.stack([row, col], axis=1), axis=0, return_counts=True)
    return {(int(r), int(c)) for (r, c), n in zip(cells, counts) if n >= min_corners}


CELL_NAMES = {(0, 0): "top-left", (0, 1): "top", (0, 2): "top-right",
              (1, 0): "left", (1, 1): "centre", (1, 2): "right",
              (2, 0): "bottom-left", (2, 1): "bottom", (2, 2): "bottom-right"}


def missing_cells(covered: set[tuple[int, int]]) -> list[str]:
    """Names of the image regions the board has not reached yet, corners first."""
    order = [(0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1), (1, 1)]
    return [CELL_NAMES[cell] for cell in order if cell not in covered]
