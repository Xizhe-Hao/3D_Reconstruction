"""ChArUco board definition shared by every script in this folder.

One JSON file describes the target so that the *same* geometry is used to
print it, to detect its corners, and to solve the calibration. If the printed
board and the JSON disagree, every downstream number is silently wrong, so
this is deliberately the single source of truth.

ALL LENGTHS ARE IN MILLIMETRES. The solved translations therefore come out in
millimetres too, which is the natural unit for this rig (soft-body deformation
is a millimetre-scale quantity).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

# Predefined ArUco dictionaries, by the name written into board.json.
# 4X4 markers carry the least data and are the easiest to detect at an angle,
# which is what matters on a rig where the board is seen very obliquely.
ARUCO_DICTS = {
    "DICT_4X4_50": cv.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv.aruco.DICT_6X6_50,
    "DICT_6X6_250": cv.aruco.DICT_6X6_250,
}

DICT_CAPACITY = {
    "DICT_4X4_50": 50, "DICT_4X4_100": 100, "DICT_4X4_250": 250,
    "DICT_5X5_50": 50, "DICT_5X5_100": 100, "DICT_5X5_250": 250,
    "DICT_6X6_50": 50, "DICT_6X6_250": 250,
}


@dataclass
class BoardSpec:
    """Geometry of the printed ChArUco target (millimetres)."""

    squares_x: int
    squares_y: int
    square_mm: float
    marker_mm: float
    dictionary: str = "DICT_4X4_50"
    legacy_pattern: bool = False  # only for boards printed with OpenCV < 4.6
    note: str = ""

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("a ChArUco board needs at least 3x3 squares")
        if self.squares_x == self.squares_y:
            raise ValueError(
                "square boards are rotationally ambiguous for pose averaging; "
                "use a non-square layout (e.g. 8x11)"
            )
        if not 0 < self.marker_mm < self.square_mm:
            raise ValueError("marker_mm must be > 0 and smaller than square_mm")
        if self.marker_mm / self.square_mm > 0.85:
            raise ValueError(
                "marker_mm / square_mm > 0.85 leaves too little white border; "
                "0.7-0.8 is the usable range"
            )
        if self.dictionary not in ARUCO_DICTS:
            raise ValueError(
                f"unknown dictionary {self.dictionary!r}; "
                f"choose one of {sorted(ARUCO_DICTS)}"
            )
        needed = (self.squares_x * self.squares_y) // 2
        capacity = DICT_CAPACITY[self.dictionary]
        if needed > capacity:
            raise ValueError(
                f"{self.squares_x}x{self.squares_y} needs {needed} markers but "
                f"{self.dictionary} only holds {capacity}"
            )

    # ------------------------------------------------------------------
    @property
    def num_corners(self) -> int:
        """Number of interior chessboard corners (= the calibration points)."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def size_mm(self) -> tuple[float, float]:
        return (self.squares_x * self.square_mm, self.squares_y * self.square_mm)

    def build(self) -> cv.aruco.CharucoBoard:
        self.validate()
        board = cv.aruco.CharucoBoard(
            size=(self.squares_x, self.squares_y),
            squareLength=float(self.square_mm),
            markerLength=float(self.marker_mm),
            dictionary=cv.aruco.getPredefinedDictionary(ARUCO_DICTS[self.dictionary]),
        )
        board.setLegacyPattern(bool(self.legacy_pattern))
        return board

    def object_points(self) -> np.ndarray:
        """(num_corners, 3) float32 corner positions in the board frame.

        Taken straight from the board object so that row i always corresponds
        to ChArUco corner id i. Deriving this grid by hand is the classic way
        to get a calibration that converges to a plausible-looking but wrong
        answer, so we never do it.
        """
        pts = np.asarray(self.build().getChessboardCorners(), dtype=np.float32)
        return pts.reshape(-1, 3)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "BoardSpec":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        spec = BoardSpec(**{k: v for k, v in data.items() if k in BoardSpec.__annotations__})
        spec.validate()
        return spec

    def describe(self) -> str:
        w, h = self.size_mm
        return (
            f"{self.squares_x}x{self.squares_y} squares, {self.square_mm:g} mm square / "
            f"{self.marker_mm:g} mm marker, {self.dictionary}, "
            f"{self.num_corners} corners, board {w:g}x{h:g} mm"
        )


def make_detector(board: cv.aruco.CharucoBoard) -> cv.aruco.CharucoDetector:
    """Detector configured the way OpenCV's own multiview sample configures it.

    Contour-based marker refinement plus `tryRefineMarkers` recovers markers
    that a plain threshold pass drops at steep viewing angles, which is exactly
    the regime a ring of cameras around a small object operates in.
    """
    detector_params = cv.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR

    charuco_params = cv.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True

    return cv.aruco.CharucoDetector(
        board, charuco_params, detector_params, cv.aruco.RefineParameters()
    )
