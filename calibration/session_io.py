"""Reading a capture session produced by `capture_3blackfly_sensor_force.py`.

The rig triggers every camera from the same Arduino D9 pulse, so frames with
equal `capture_sequence_index` in `frames.csv` were exposed at the same
instant. That index -- not the file order -- is what makes a frame a "frame"
across cameras, and multi-view extrinsics calibration is only valid because of
it.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

# ---------------------------------------------------------------------------
# Bayer naming: GenICam names a pattern after the first two pixels of the FIRST
# row (BayerRG8 -> row0 = R G R G, row1 = G B G B). OpenCV names it after the
# second and third pixel of the SECOND row (that same pattern is "BG" to
# OpenCV). The two conventions are shifted by one pixel, so the letters swap.
# Getting this wrong does not throw -- it just biases interpolated intensities
# and quietly costs you sub-pixel corner accuracy.
GENICAM_TO_OPENCV_BAYER = {
    "BayerRG": "BG",
    "BayerBG": "RG",
    "BayerGR": "GB",
    "BayerGB": "GR",
}

_BAYER_TO_GRAY = {
    "BG": cv.COLOR_BayerBG2GRAY,
    "RG": cv.COLOR_BayerRG2GRAY,
    "GB": cv.COLOR_BayerGB2GRAY,
    "GR": cv.COLOR_BayerGR2GRAY,
}

_BAYER_TO_BGR = {
    "BG": cv.COLOR_BayerBG2BGR,
    "RG": cv.COLOR_BayerRG2BGR,
    "GB": cv.COLOR_BayerGB2BGR,
    "GR": cv.COLOR_BayerGR2BGR,
}


@dataclass
class CameraStream:
    index: int
    serial: str
    directory: Path
    # capture_sequence_index -> image path
    images: dict[int, Path] = field(default_factory=dict)


@dataclass
class Session:
    root: Path
    meta: dict
    cameras: list[CameraStream]
    frame_indices: list[int]
    bayer: str | None  # OpenCV two-letter code, or None if already demosaiced

    @property
    def serials(self) -> list[str]:
        return [c.serial for c in self.cameras]

    def image_path(self, camera: int, frame_index: int) -> Path | None:
        return self.cameras[camera].images.get(frame_index)


def _resolve_bayer(meta: dict, override: str | None) -> str | None:
    """Decide how a saved BMP has to be converted to grayscale."""
    if override is not None:
        override = override.strip()
        if override.lower() in ("none", "off", "mono", "gray"):
            return None
        code = override.upper()
        if code in _BAYER_TO_GRAY:
            return code
        raise ValueError(f"--bayer must be one of BG/RG/GB/GR/none, got {override!r}")

    if meta.get("demosaic_on_save"):
        return None  # already BGR8 on disk

    fmt = ""
    for stat in meta.get("camera_stats") or []:
        fmt = str(stat.get("saved_pixel_format") or stat.get("source_pixel_format") or "")
        if fmt:
            break
    match = re.match(r"(Bayer(?:RG|BG|GR|GB))", fmt)
    if match:
        return GENICAM_TO_OPENCV_BAYER[match.group(1)]
    if fmt.startswith("Mono") or fmt.startswith("BGR") or fmt.startswith("RGB"):
        return None
    return None


def load_session(session_dir: str | Path, bayer_override: str | None = None) -> Session:
    root = Path(session_dir)
    meta_path = root / "session.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found -- is this a capture session?")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    camera_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and re.match(r"camera_\d+_", p.name)),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not camera_dirs:
        raise FileNotFoundError(f"no camera_<i>_<serial> directories under {root}")

    cameras: list[CameraStream] = []
    for directory in camera_dirs:
        index = int(directory.name.split("_")[1])
        serial = directory.name.split("_", 2)[2]
        frames_csv = directory / "frames.csv"
        if not frames_csv.exists():
            raise FileNotFoundError(f"{frames_csv} not found")

        images: dict[int, Path] = {}
        with frames_csv.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("complete") not in (None, "", "1"):
                    continue
                rel = (row.get("image_path") or "").strip()
                if not rel:
                    continue
                path = root / rel.replace("\\", "/")
                if path.exists():
                    images[int(row["capture_sequence_index"])] = path

        if not images:
            raise FileNotFoundError(
                f"{directory.name}: frames.csv lists no readable images. "
                "A calibration session must be exported WITHOUT --delete-images."
            )
        cameras.append(CameraStream(index, serial, directory, images))

    # A frame is usable only if at least two cameras recorded it; anything seen
    # by one camera alone can still help intrinsics, so keep the union here and
    # let the caller decide.
    all_indices = sorted(set().union(*[set(c.images) for c in cameras]))

    return Session(
        root=root,
        meta=meta,
        cameras=cameras,
        frame_indices=all_indices,
        bayer=_resolve_bayer(meta, bayer_override),
    )


def read_gray(path: str | Path, bayer: str | None) -> np.ndarray:
    """Load one saved frame as 8-bit grayscale, demosaicing when needed."""
    if bayer is None:
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"cannot read {path}")
        return img

    raw = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"cannot read {path}")
    if raw.ndim == 3:
        # Already demosaiced on disk despite the session metadata.
        return cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
    return cv.cvtColor(raw, _BAYER_TO_GRAY[bayer])


def read_bgr(path: str | Path, bayer: str | None) -> np.ndarray:
    """Colour version of :func:`read_gray`, used only for debug overlays."""
    raw = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"cannot read {path}")
    if raw.ndim == 3:
        return raw
    if bayer is None:
        return cv.cvtColor(raw, cv.COLOR_GRAY2BGR)
    return cv.cvtColor(raw, _BAYER_TO_BGR[bayer])
