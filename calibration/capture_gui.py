#!/usr/bin/env python3
r"""One window for the whole calibration: pre-flight, capture, verdict.

RUN WITH THE **CAPTURE** ENVIRONMENT (needs PySpin):

    cd "sensor-data-collection\Python files"
    .\.venv\Scripts\Activate.ps1
    python ..\..\calibration\capture_gui.py ^
        --board ..\..\calibration\board\board.json ^
        --output C:\tactile_calib\guided

That is the only command. The solve runs itself afterwards under the
calibration interpreter, so there is no second environment to activate and no
path to retype.

Three phases
------------
**Pre-flight.** Nothing is recorded until every camera passes. It checks the
four faults that have each cost a whole session here and none of which announce
themselves during capture: a lens that is not focused, a board held outside the
depth of field, clipped or dark exposure, and a camera that has moved since the
window opened. Focus is the 10-90% rise width of a black/white edge -- under
3.2 px the corner refiner sees an edge, above it a ramp. Drift is measured
against the static scene and keeps running through the capture, so by the time
the session ends it covers the whole half hour.

**Capture.** The protocol is named a step at a time and a frame set is recorded
only when the pose is genuinely right: the board still, square-on to the camera
that needs it, sharing enough corners with its partner, and -- for the steps
that feed the multi-view solve -- entirely inside the frame. Detecting *the
board* is not the same as seeing *all of* it, and the strict solve uses only
frames where every corner was measured.

It deliberately does not use the Arduino trigger. Hardware sync matters when the
target moves between cameras' exposures; this protocol holds the board still and
verifies stationarity in every contributing view before saving, which is the
stricter condition. Calibration therefore does not depend on the sensor rig
running at all.

**Verdict.** On "finish and save" the window closes and the session is graded:
drift per camera, then detect_corners -> calibrate -> validate, then GOOD /
MARGINAL / NOT USABLE / UNUSABLE. Drift outranks the rest -- if the rig changed
halfway, no calibration of it exists and the pixel numbers describe nothing.

Output is an ordinary capture session (`camera_<i>_<serial>/` + `frames.csv` +
`session.json`) so `detect_corners.py` reads it with no changes -- roughly 1 GB
instead of 27 GB, because only good poses land on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The Windows console defaults to cp1252, which cannot encode the degree sign
# and arrows in the prompts; without this a stray print kills the UI loop.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    import cv2 as cv
except ImportError:
    sys.exit("OpenCV missing in the capture environment:\n"
             "  python -m pip install opencv-python==4.9.0.80")
try:
    import PySpin
except ImportError:
    sys.exit("PySpin not found -- activate the capture environment (.venv).")
from PIL import Image, ImageTk  # noqa: E402

from charuco_board import BoardSpec  # noqa: E402
import focus_check  # noqa: E402
import session_report  # noqa: E402
from tune_exposure import prepare_for_capture, set_float, throttle  # noqa: E402

# GenICam BayerRG8 is OpenCV's BayerBG -- the two conventions are one pixel
# apart. session_io.py documents this; keep the two in step.
BAYER_TO_GRAY = cv.COLOR_BayerBG2GRAY
BAYER_TO_BGR = cv.COLOR_BayerBG2BGR

FACING_GOOD = 0.55
STILL_PIXELS = 1.5          # per-corner movement between frames to count as still
MIN_CORNERS = 12
FULL_BOARD_MARGIN = 0.6     # keep-out zone at the image border, in squares
SHARP_PX = 3.2              # 10-90% edge width above which corners are mush
DRIFT_PX = 3.0              # rig movement above which no calibration exists
P99_DARK, P99_CLIPPED = 120, 245

# One palette, used for the card borders, the headline and the buttons, so a
# colour always means the same thing wherever it appears.
BG, CARD, LINE = "#f2f2f4", "#ffffff", "#dcdce2"
INK, MUTED = "#1a1a1e", "#63636e"
GOOD, WARN, BAD, IDLE = "#1f9d55", "#c77400", "#c0392b", "#b4b4bc"

# What to do about each pre-flight fault. Keyed on the first word of the fault
# so the card headline stays short enough to read at a glance.
FIX = {
    "MEASURING": "Waiting for the first focus reading -- a few seconds.",
    "LENS": "Turn that camera's focus ring until its focus number bottoms out. "
            "Focus on the TACTILE ARRAY, not on the background.",
    "NO": "That camera cannot see the board. Put the board flat on the array "
          "where all four cameras can see it.",
    "BOARD": "The lens is focused, but not at the board. Turn that camera's "
             "focus ring while watching the second focus number.",
    "TOO": "Press auto exposure, or open that lens's iris.",
    "OVEREXPOSED": "Highlights are clipped, which destroys marker decoding just "
                   "like blur. Press auto exposure, or stop the iris down.",
    "MOVED": "That camera has shifted since this window opened. Secure the mount "
             "and the cable before capturing -- a rig that moves cannot be calibrated.",
}


# The widest readout the cards ever show. The monospace size is chosen so this
# fits the preview width exactly once -- wrapping it instead makes the card
# taller every time the preview gets narrower, which never converges.
READOUT_SAMPLE = "focus 2.00/2.10  drift 12.3px  p99 200 CLIPPED"


def mono_size(width_px, ceiling):
    """Largest Consolas size at which READOUT_SAMPLE fits `width_px`."""
    for size in range(ceiling, 6, -1):
        if tkfont.Font(family="Consolas", size=size).measure(READOUT_SAMPLE) <= width_px:
            return size
    return 7


def ui_scale(root, cameras=4):
    """Size the whole interface to the display it is actually on.

    Hard-coded point sizes are unreadable on a 4K panel and overflow a laptop,
    and this window is read at arm's length while both hands hold the board --
    so the previews take whatever room is left after the text, rather than the
    text being squeezed around a fixed preview.
    """
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    scale = min(max(screen_h / 1080.0, 0.85), 1.7)
    chrome = int(430 * scale)                     # header, progress, status, buttons
    card_extra = int(130 * scale)                 # title bar + readouts on each card
    avail_h, avail_w = screen_h * 0.86, screen_w * 0.95

    # Screens are wide and this window is tall, so the row/column split is worth
    # choosing rather than assuming: four cameras in one row make each preview
    # more than twice the size that a 2x2 grid allows on a 16:10 display.
    best = None
    for columns in (4, 3, 2, 1):
        rows = -(-cameras // columns)
        by_height = (avail_h - chrome - rows * card_extra) / rows
        by_width = (avail_w - columns * 22) / columns * 3 / 4
        height_px = min(by_height, by_width)
        if height_px >= 150 and (best is None or height_px > best[1]):
            best = (columns, height_px)
    columns, preview_h = best if best else (2, 150)
    preview_h = int(min(preview_h, 430))
    preview_w = int(preview_h * 4 / 3)
    rows = -(-cameras // columns)
    width = preview_w * columns + int(30 * scale) * columns
    height = min(int(avail_h), preview_h * rows + rows * card_extra + chrome)

    def font(size, weight="normal", family="Segoe UI"):
        return (family, max(9, int(round(size * scale))), weight)

    return {"font": font, "preview": (preview_w, preview_h), "columns": columns,
            "geometry": f"{width}x{height}+{max(0, (screen_w - width) // 2)}+20",
            "wrap": width - int(60 * scale), "border": max(3, int(4 * scale))}
SHARED_CORNERS = 20         # corner ids two cameras must BOTH see to be useful
GROUP_CAMERAS = 3           # cameras that must agree in the all-cameras step


# --------------------------------------------------------------------- protocol
@dataclass
class Step:
    key: str
    title: str
    detail: str
    needed: int
    target: int | None = None          # camera index that must face the board
    pair: tuple[int, int] | None = None
    group: list[int] | None = None     # cameras that must share a common region
    min_shared: int = 0                # corner ids every watched camera must see
    # The world step deliberately records the SAME static pose many times, so
    # averaging can beat down noise in the origin; every other step wants
    # distinct poses and rejects repeats.
    want_distinct_poses: bool = True
    captured: int = 0
    signatures: list = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.captured >= self.needed


def build_plan(serials: list[str]) -> list[Step]:
    total = 2 * len(serials) + 1
    steps = [Step(
        key="world",
        title=f"Step 1 of {total}  --  world frame",
        detail=("Lay the board FLAT on the tactile array, printed side up, then\n"
                "LET GO. This segment defines the world origin: the same pose is\n"
                "recorded 12x for averaging, so do not move the board at all."),
        needed=12,
        want_distinct_poses=False,
    )]
    for index, serial in enumerate(serials):
        steps.append(Step(
            key=f"face{index}",
            title=f"Step {index + 2} of {total}  --  face camera {index} ({serial})",
            detail=(f"Pick the board up and aim its printed face at {serial}." + " " +
                    f"Turn it until that camera reads facing > {FACING_GOOD:.2f}," + " " +
                    "then hold still." + " " +
                    "Change the pose between shots: 30 deg left and right, 30 deg" + " " +
                    "up and down, nearer, further."),
            needed=8,
            target=index,
        ))
    # Extrinsics live entirely in these steps: the "face" steps above point the
    # board at one camera, which by construction hides it from the others. Every
    # adjacent pair needs its own healthy stack of shared observations, plus a
    # final all-cameras block that ties the whole graph together.
    number = len(serials) + 2
    for first in range(len(serials) - 1):
        second = first + 1
        steps.append(Step(
            key=f"pair{first}{second}",
            title=f"Step {number + first} of {total}  --  cameras {first} + {second} together",
            detail=(f"Hold the board between {serials[first]} and {serials[second]}," + " " +
                    "at an angle that splits the difference, so BOTH cameras see" + " " +
                    "the SAME part of it -- the shared-corner count is under each" + " " +
                    "panel. Change the angle or the distance after every shot."),
            needed=12,
            pair=(first, second),
            min_shared=SHARED_CORNERS,
        ))
    steps.append(Step(
        key="all",
        title=f"Step {number + len(serials) - 1} of {total}  --  all cameras together",
        detail=("Put the board near the middle of the working volume and angle it" + " " +
                "so AS MANY CAMERAS AS POSSIBLE see the same part of it at once." + " " +
                "This step is what ties the whole camera graph together."),
        needed=12,
        group=list(range(len(serials))),
        min_shared=SHARED_CORNERS,
    ))
    return steps


# ------------------------------------------------------------------ camera rig
@dataclass
class CameraState:
    serial: str
    raw: np.ndarray | None = None
    preview: np.ndarray | None = None
    corners: np.ndarray | None = None
    ids: np.ndarray | None = None
    facing: float = 0.0
    p99: float = 0.0
    moved: float = 1e9
    stamp_ns: int = 0
    sharp_scene: float = 0.0    # edge width of the sharpest part of the frame
    sharp_board: float = 0.0    # edge width measured on the printed target
    drift: float = 0.0          # how far the static scene has moved since start
    in_frame: int = 0           # board corners whose projection lands in the image
    nudge: str = ""             # which way to move the board to bring it all in


class Rig:
    """Owns the cameras and a background grab/detect loop."""

    def __init__(self, board, object_points, fps: float, timeout_ms: int,
                 monitor_period: float = 4.0, preview_size=(320, 240)):
        self.object_points = object_points
        self.timeout_ms = timeout_ms
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.system = PySpin.System.GetInstance()
        self.camera_list = self.system.GetCameras()
        if self.camera_list.GetSize() == 0:
            self.camera_list.Clear()
            self.system.ReleaseInstance()
            raise SystemExit("no cameras found (is SpinView open?)")

        self.handles = []
        for index in range(self.camera_list.GetSize()):
            camera = self.camera_list[index]
            camera.Init()
            node = PySpin.CStringPtr(
                camera.GetTLDeviceNodeMap().GetNode("DeviceSerialNumber"))
            self.handles.append(
                (node.GetValue() if PySpin.IsReadable(node) else "?", camera))
        self.handles.sort(key=lambda item: item[0])
        self.serials = [serial for serial, _ in self.handles]

        detector_params = cv.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
        charuco_params = cv.aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = True
        self.detectors = {
            serial: cv.aruco.CharucoDetector(board, charuco_params, detector_params,
                                             cv.aruco.RefineParameters())
            for serial in self.serials}

        self.state = {serial: CameraState(serial) for serial in self.serials}
        self.previous = {serial: None for serial in self.serials}
        self.orb = cv.ORB_create(3000)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.drift_reference = {}
        self.monitor_period = monitor_period
        self.preview_size = preview_size
        self.last_monitor = {serial: 0.0 for serial in self.serials}

        for _, camera in self.handles:
            throttle(camera, fps)
            camera.BeginAcquisition()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    def _detect(self, serial, gray):
        corners, ids, _, _ = self.detectors[serial].detectBoard(gray)
        if ids is None or len(ids) < MIN_CORNERS:
            return None, None, 0.0, 0, ""
        ids = np.asarray(ids).reshape(-1)
        corners = np.asarray(corners).reshape(-1, 2)
        homography, _ = cv.findHomography(
            np.ascontiguousarray(self.object_points[ids][:, :2], np.float64),
            np.ascontiguousarray(corners, np.float64), 0)
        facing = 0.0
        if homography is not None:
            singular = np.linalg.svd(homography[:2, :2], compute_uv=False)
            if singular[0] > 1e-9:
                facing = float(singular[1] / singular[0])
            in_frame, nudge = self._framing(homography, gray.shape)
        return corners, ids, facing, in_frame, nudge

    def _framing(self, homography, shape):
        """How much of the WHOLE board lands in the frame, and where to move it.

        Detecting *the board* says nothing about seeing *all of* it. The strict
        multi-view solve only uses frames in which a camera measured every
        corner, and a target held near the edge of the field of view silently
        yields none of those -- the session finishes looking healthy and the
        co-visibility matrix comes back empty two steps later. The homography
        extrapolates the corners the detector missed, so this counts the ones
        that are really in view rather than the ones that happened to decode.
        """
        height, width = shape[:2]
        projected = cv.perspectiveTransform(
            np.ascontiguousarray(self.object_points[:, :2], np.float64).reshape(-1, 1, 2),
            homography).reshape(-1, 2)
        # A ChArUco corner needs its neighbouring squares to be visible too, so
        # the usable region stops short of the border by about one square.
        diff = projected[:, None, :] - projected[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        square_px = float(np.median(dist.min(axis=1)))
        margin = FULL_BOARD_MARGIN * square_px
        inside = ((projected[:, 0] >= margin) & (projected[:, 0] < width - margin)
                  & (projected[:, 1] >= margin) & (projected[:, 1] < height - margin))
        if inside.all():
            return int(inside.sum()), ""
        # Name the direction to move, not the number to interpret.
        out = projected[~inside]
        hints = []
        if (out[:, 0] < margin).any(): hints.append("right")
        if (out[:, 0] >= width - margin).any(): hints.append("left")
        if (out[:, 1] < margin).any(): hints.append("down")
        if (out[:, 1] >= height - margin).any(): hints.append("up")
        if len(hints) >= 3:
            hints = ["further away"]
        return int(inside.sum()), "".join(hints)

    def _monitor(self, serial, gray, corners):
        """Focus and rig-drift, both sampled slowly because both change slowly.

        These are the two faults that do not announce themselves. A defocused
        camera still detects corners, and a camera that creeps during the
        session still produces a session -- the damage only appears as a
        residual nobody can attribute. Watching them continuously costs a
        fraction of a second every few seconds and turns both into a number on
        screen while there is still time to act.
        """
        now = time.monotonic()
        if now - self.last_monitor[serial] < self.monitor_period:
            return
        self.last_monitor[serial] = now

        scene = focus_check.scene_sharpness(gray)
        board = (focus_check.board_sharpness(gray, corners)
                 if corners is not None and len(corners) >= 8 else None)

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        drift = None
        reference = self.drift_reference.get(serial)
        if reference is None:
            if descriptors is not None and len(descriptors) >= 60:
                self.drift_reference[serial] = (keypoints, descriptors)
        elif descriptors is not None and len(descriptors) >= 30:
            drift = self._displacement(reference, (keypoints, descriptors))

        with self.lock:
            state = self.state[serial]
            if scene is not None:
                state.sharp_scene = scene
            if board is not None:
                state.sharp_board = board
            if drift is not None:
                state.drift = drift

    def _displacement(self, reference, current):
        """Median motion of the static scene between two frames of one camera.

        RANSAC keeps this honest while the board and a pair of hands move
        through the frame: they are a minority of the matches, so the dominant
        transform is the one that describes the rig.
        """
        ka, da = reference
        kb, db = current
        matches = sorted(self.matcher.match(da, db), key=lambda m: m.distance)[:600]
        if len(matches) < 30:
            return None
        src = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        _, inliers = cv.findHomography(src, dst, cv.RANSAC, 3.0)
        if inliers is None or inliers.sum() < 20:
            return None
        keep = inliers.ravel().astype(bool)
        return float(np.median(np.linalg.norm(
            dst[keep].reshape(-1, 2) - src[keep].reshape(-1, 2), axis=1)))

    def _loop(self):
        while not self.stop_event.is_set():
            for serial, camera in self.handles:
                try:
                    image = camera.GetNextImage(self.timeout_ms)
                except PySpin.SpinnakerException:
                    continue
                try:
                    if image.IsIncomplete():
                        continue
                    raw = np.array(image.GetNDArray(), copy=True)
                    stamp = time.time_ns()
                finally:
                    image.Release()

                gray = cv.cvtColor(raw, BAYER_TO_GRAY) if raw.ndim == 2 else raw
                corners, ids, facing, in_frame, nudge = self._detect(serial, gray)
                self._monitor(serial, gray, corners)

                moved = 1e9
                earlier = self.previous[serial]
                if corners is not None and earlier is not None:
                    old_ids, old_corners = earlier
                    shared = np.intersect1d(ids, old_ids)
                    if len(shared) >= 6:
                        a = corners[np.searchsorted(ids, shared)]
                        b = old_corners[np.searchsorted(old_ids, shared)]
                        moved = float(np.linalg.norm(a - b, axis=1).mean())
                if corners is not None:
                    order = np.argsort(ids)
                    self.previous[serial] = (ids[order], corners[order])
                else:
                    self.previous[serial] = None

                preview = cv.cvtColor(raw, BAYER_TO_BGR) if raw.ndim == 2 else raw
                preview = cv.resize(preview, self.preview_size,
                                    interpolation=cv.INTER_AREA)
                if corners is not None:
                    scale = np.array([self.preview_size[0] / raw.shape[1],
                                      self.preview_size[1] / raw.shape[0]])
                    dot = max(2, self.preview_size[0] // 200)
                    for point in (corners * scale).astype(int):
                        cv.circle(preview, tuple(point), dot, (0, 255, 255), -1)

                with self.lock:
                    state = self.state[serial]
                    state.raw = raw
                    state.preview = preview
                    state.corners = corners
                    state.ids = ids
                    state.facing = facing
                    state.in_frame = in_frame
                    state.nudge = nudge
                    state.p99 = float(np.percentile(raw, 99))
                    state.moved = moved
                    state.stamp_ns = stamp

    def snapshot(self) -> dict[str, CameraState]:
        with self.lock:
            return {serial: CameraState(
                serial=state.serial, raw=state.raw, preview=state.preview,
                corners=state.corners, ids=state.ids, facing=state.facing,
                p99=state.p99, moved=state.moved, stamp_ns=state.stamp_ns,
                in_frame=state.in_frame, nudge=state.nudge,
                sharp_scene=state.sharp_scene, sharp_board=state.sharp_board,
                drift=state.drift)
                for serial, state in self.state.items()}

    def auto_exposure(self, target_p99: float = 205.0, max_exposure_us: float = 30000.0,
                      max_gain_db: float = 18.0, iterations: int = 5) -> str:
        """Bring every camera to the same brightness, each with its own settings.

        Exposure first and gain only for the shortfall: gain is amplification,
        so it buys brightness at the price of the noise that limits sub-pixel
        corner localisation. The live p99 the monitor already maintains is the
        measurement, so this needs no extra grabs of its own.
        """
        for _ in range(iterations):
            with self.lock:
                measured = {s: self.state[s].p99 for s in self.serials}
            if all(abs(v - target_p99) <= 12 for v in measured.values() if v > 0):
                break
            for serial, camera in self.handles:
                p99 = measured.get(serial, 0.0)
                if p99 <= 0:
                    continue
                nodemap = camera.GetNodeMap()
                exposure = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime")).GetValue()
                gain = PySpin.CFloatPtr(nodemap.GetNode("Gain")).GetValue()
                wanted = target_p99 / max(p99, 1.0)
                headroom = max_exposure_us / max(exposure, 1.0)
                factor = min(wanted, headroom)
                exposure = set_float(nodemap, "ExposureTime", exposure * factor)
                remaining = wanted / max(factor, 1e-6)
                if remaining > 1.02 or (remaining < 0.98 and gain > 0):
                    set_float(nodemap, "Gain",
                              float(np.clip(gain + 20.0 * np.log10(remaining),
                                            0.0, max_gain_db)))
            time.sleep(max(0.6, self.monitor_period * 0.4))
        report = []
        for serial, camera in self.handles:
            nodemap = camera.GetNodeMap()
            report.append(
                f"{serial[-4:]} {PySpin.CFloatPtr(nodemap.GetNode('ExposureTime')).GetValue():.0f}us"
                f"/{PySpin.CFloatPtr(nodemap.GetNode('Gain')).GetValue():.1f}dB")
        return "exposure set -- " + "  ".join(report)

    def scale_exposure(self, factor: float) -> str:
        values = []
        for _, camera in self.handles:
            nodemap = camera.GetNodeMap()
            current = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime")).GetValue()
            values.append(set_float(nodemap, "ExposureTime", current * factor))
        return ", ".join(f"{v:.0f}" for v in values)

    def close(self, trigger_hz: float = 24.2):
        self.stop_event.set()
        self.thread.join(timeout=3.0)
        for _, camera in self.handles:
            try:
                camera.EndAcquisition()
            except PySpin.SpinnakerException:
                pass
            try:
                prepare_for_capture(camera, trigger_hz)
            except PySpin.SpinnakerException:
                pass
        while self.handles:
            _, camera = self.handles.pop()
            try:
                camera.DeInit()
            except PySpin.SpinnakerException:
                pass
            del camera
        self.camera_list.Clear()
        self.system.ReleaseInstance()


# --------------------------------------------------------------------- writing
class SessionWriter:
    def __init__(self, root: Path, serials: list[str], board_path: Path):
        stamp = time.strftime("capture_%Y%m%d_%H%M%S")
        self.root = root / stamp
        self.serials = serials
        self.board_path = board_path
        self.rows = {serial: [] for serial in serials}
        self.index = 0
        for position, serial in enumerate(serials):
            (self.root / f"camera_{position}_{serial}").mkdir(parents=True, exist_ok=True)

    def save_set(self, snapshot: dict[str, CameraState]) -> int:
        index = self.index
        for position, serial in enumerate(self.serials):
            state = snapshot[serial]
            if state.raw is None:
                continue
            name = f"capture_{index:09d}_fid_{index}.bmp"
            relative = f"camera_{position}_{serial}/{name}"
            cv.imwrite(str(self.root / relative), state.raw)
            self.rows[serial].append({
                "host_received_ns": state.stamp_ns,
                "capture_sequence_index": index,
                "camera_frame_id": index,
                "camera_timestamp": state.stamp_ns,
                "complete": 1,
                "image_status": 0,
                "image_path": relative.replace("/", "\\"),
            })
        self.index += 1
        return index

    def finish(self, steps: list[Step]) -> Path:
        fields = ["host_received_ns", "capture_sequence_index", "camera_frame_id",
                  "camera_timestamp", "complete", "image_status", "image_path"]
        for position, serial in enumerate(self.serials):
            path = self.root / f"camera_{position}_{serial}" / "frames.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.rows[serial])
        world = [s for s in steps if s.key == "world"]
        (self.root / "session.json").write_text(json.dumps({
            "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "camera_count": len(self.serials),
            "camera_serials": self.serials,
            "image_format": "bmp",
            "color_requested": True,
            "demosaic_on_save": False,
            "trigger_source": "none (guided free-run, board verified stationary)",
            "camera_stats": [{"camera_index": i, "serial": s,
                              "source_pixel_format": "BayerRG8",
                              "saved_pixel_format": "BayerRG8"}
                             for i, s in enumerate(self.serials)],
            "guided_capture": True,
            "board": str(self.board_path),
            "world_frames": [0, world[0].captured if world else 0],
            "steps": [{"key": s.key, "captured": s.captured, "needed": s.needed}
                      for s in steps],
        }, indent=2), encoding="utf-8")
        return self.root


# ------------------------------------------------------------------------- gui
class App:
    def __init__(self, root: tk.Tk, rig: Rig, steps: list[Step],
                 writer: SessionWriter, args):
        self.root = root
        self.rig = rig
        self.steps = steps
        self.writer = writer
        self.args = args
        self.current = 0
        self.phase = "preflight"
        self.drift_at_start = {}
        self.bumped = {}
        self.running = True
        self.saved_path = None
        self.message = ""
        self.message_until = 0.0

        self.ui = ui_scale(root)
        f = self.ui["font"]
        root.title("Calibration capture")
        root.configure(bg=BG)
        root.geometry(self.ui["geometry"])
        root.minsize(900, 640)

        self.title_var = tk.StringVar()
        self.detail_var = tk.StringVar()
        self.progress_var = tk.StringVar()
        self.status_var = tk.StringVar()

        head = tk.Frame(root, bg=BG)
        head.pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(head, textvariable=self.title_var, font=f(20, "bold"),
                 anchor="w", bg=BG, fg=INK).pack(fill="x")
        tk.Label(head, textvariable=self.detail_var, font=f(12), anchor="w",
                 justify="left", bg=BG, fg=MUTED, wraplength=self.ui["wrap"]
                 ).pack(fill="x", pady=(4, 0))

        # The one line that says whether to act: kept large, on its own, and
        # coloured, because during capture the operator is looking at the board
        # and glances at the screen rather than reading it.
        self.progress_label = tk.Label(root, textvariable=self.progress_var,
                                       font=f(16, "bold"), anchor="w", bg=BG, fg=GOOD)
        self.progress_label.pack(fill="x", padx=18, pady=(10, 2))
        self.bar = ttk.Progressbar(root, mode="determinate", maximum=100)
        self.bar.pack(fill="x", padx=18, pady=(0, 8))

        grid = tk.Frame(root, bg=BG)
        grid.pack(padx=12, pady=2)
        self.panels = {}
        pw, ph = self.ui["preview"]
        columns = self.ui["columns"]
        for position, serial in enumerate(rig.serials):
            card = tk.Frame(grid, bd=0, bg=CARD, highlightthickness=2,
                            highlightbackground=LINE, highlightcolor=LINE)
            card.grid(row=position // columns, column=position % columns,
                      padx=8, pady=8)
            rule = tk.Frame(card, bg=IDLE, height=self.ui["border"])
            rule.pack(fill="x")
            header = tk.StringVar()
            header_label = tk.Label(card, textvariable=header, font=f(12, "bold"),
                                    anchor="w", bg=CARD, fg=INK, padx=12, pady=6)
            header_label.pack(fill="x")
            # A black image of the right size, not width=/height= -- those are
            # character cells on a Label with no image, so 492x369 asks for a
            # 3942 x 7386 px widget and the other cards get pushed off-screen.
            blank = ImageTk.PhotoImage(Image.new("RGB", (pw, ph), "#101014"))
            image_label = tk.Label(card, image=blank, bd=0)
            image_label.image = blank
            image_label.pack()
            text = tk.StringVar()
            text_label = tk.Label(card, textvariable=text,
                                  font=("Consolas", mono_size(pw - 24, f(11)[1])),
                                  anchor="w", justify="left", bg=CARD, fg=MUTED,
                                  padx=12, pady=8)
            text_label.pack(fill="x")
            self.panels[serial] = {"card": card, "header": header, "rule": rule,
                                   "header_label": header_label, "text_label": text_label,
                                   "image": image_label, "text": text}

        tk.Label(root, textvariable=self.status_var, font=f(12), anchor="w",
                 bg=BG, fg=MUTED).pack(fill="x", padx=18, pady=(6, 0))

        def button(parent, text, command, accent=False, big=False):
            return tk.Button(parent, text=text, command=command,
                             font=f(13 if big else 11, "bold" if accent else "normal"),
                             bg=GOOD if accent else "#e9e9ec",
                             fg="white" if accent else INK,
                             activebackground=GOOD if accent else "#dcdce0",
                             relief="flat", padx=16 if big else 12, pady=8 if big else 6,
                             cursor="hand2")

        self.preflight_bar = tk.Frame(root, bg=BG)
        self.preflight_bar.pack(fill="x", padx=18, pady=12)
        button(self.preflight_bar, "auto exposure", self.auto_exposure).pack(side="left")
        button(self.preflight_bar, "exposure -20%",
               lambda: self.nudge(0.8)).pack(side="left", padx=8)
        button(self.preflight_bar, "exposure +25%",
               lambda: self.nudge(1.25)).pack(side="left")
        self.start_button = button(self.preflight_bar, "START CALIBRATION",
                                   self.start_capture, accent=True, big=True)
        self.start_button.pack(side="right")
        button(self.preflight_bar, "start anyway",
               lambda: self.start_capture(force=True)).pack(side="right", padx=10)

        self.capture_bar = tk.Frame(root, bg=BG)
        button(self.capture_bar, "skip this step", self.skip).pack(side="left")
        button(self.capture_bar, "exposure -20%",
               lambda: self.nudge(0.8)).pack(side="left", padx=8)
        button(self.capture_bar, "exposure +25%",
               lambda: self.nudge(1.25)).pack(side="left")
        button(self.capture_bar, "FINISH AND SAVE", self.finish,
               accent=True, big=True).pack(side="right")

        self.photo_refs = {}
        # Render once so the readouts exist, THEN measure: an empty label is not
        # the size of a full one, and fitting to the empty version is how the
        # window ends up taller than the screen.
        self.tick()
        self.fit_to_screen()

    def fit_to_screen(self):
        """Shrink the previews until the whole window fits the display.

        The space the text needs cannot be predicted -- it depends on the font
        the platform actually resolves, the DPI, and how long the readout
        strings are -- so rather than guessing an allowance, build the layout,
        ask Tk what it came to, and give the previews whatever is left. Getting
        this wrong is not cosmetic: a widget that demands more than the screen
        pushes the other cards out of the window entirely.
        """
        columns = self.ui["columns"]
        rows = -(-len(self.rig.serials) // columns)
        avail_h = int(self.root.winfo_screenheight() * 0.88)
        avail_w = int(self.root.winfo_screenwidth() * 0.95)
        for _ in range(4):                       # shrinking rewraps text, so iterate
            self.root.update_idletasks()
            pw, ph = self.rig.preview_size
            need_h, need_w = self.root.winfo_reqheight(), self.root.winfo_reqwidth()
            if need_h <= avail_h and need_w <= avail_w:
                break
            scale = min((avail_h - (need_h - rows * ph)) / max(rows * ph, 1),
                        (avail_w - (need_w - columns * pw)) / max(columns * pw, 1),
                        0.98)
            pw, ph = max(200, int(pw * scale)), max(150, int(ph * scale))
            if (pw, ph) == self.rig.preview_size:
                break
            self.rig.preview_size = (pw, ph)
            for serial in self.rig.serials:
                blank = ImageTk.PhotoImage(Image.new("RGB", (pw, ph), "#101014"))
                panel = self.panels[serial]
                panel["image"].configure(image=blank)
                panel["image"].image = blank
                panel["text_label"].configure(
                    font=("Consolas", mono_size(pw - 24, self.ui["font"](11)[1])))
        self.root.geometry(f"{self.root.winfo_reqwidth()}x{self.root.winfo_reqheight()}"
                           f"+{max(0, (self.root.winfo_screenwidth() - self.root.winfo_reqwidth()) // 2)}+20")

    # ------------------------------------------------------------------
    def step(self) -> Step | None:
        return self.steps[self.current] if self.current < len(self.steps) else None

    def notify(self, text: str, seconds: float = 2.0):
        self.message = text
        self.message_until = time.monotonic() + seconds

    def skip(self):
        if self.step() is not None:
            self.current += 1
            self.notify("step skipped")

    def nudge(self, factor: float):
        self.notify("exposure -> " + self.rig.scale_exposure(factor) + " us", 3.0)

    def finish(self):
        self.running = False
        self.saved_path = self.writer.finish(self.steps)
        self.rig.close(self.args.trigger_hz)
        print()
        print(f"saved {self.writer.index} frame sets to {self.saved_path}")
        # The window goes away before the solve starts: grading takes
        # minutes, and a frozen GUI reads as a crash.
        self.root.destroy()

    # ------------------------------------------------------------------
    def accept(self, step: Step, snapshot: dict[str, CameraState]) -> str | None:
        """Return None when the pose qualifies, otherwise why it does not."""
        seen = [s for s in snapshot.values() if s.corners is not None]
        if len(seen) < 2:
            return "at least two cameras must see the board at once"

        if step.target is not None:
            watched = [self.rig.serials[step.target]]
        elif step.pair is not None:
            watched = [self.rig.serials[i] for i in step.pair]
        elif step.group is not None:
            # Requiring all four to agree is often geometrically impossible, so
            # take whichever of them currently see the board and insist on a
            # quorum instead.
            watched = [self.rig.serials[i] for i in step.group
                       if snapshot[self.rig.serials[i]].corners is not None]
            if len(watched) < GROUP_CAMERAS:
                return (f"only {len(watched)} cameras see the board; this step needs "
                        f"{GROUP_CAMERAS}")
        else:
            watched = [s.serial for s in seen]

        for serial in watched:
            state = snapshot[serial]
            if state.corners is None:
                return f"{serial} cannot see the board yet"
            if step.target is not None and state.facing < FACING_GOOD:
                return (f"{serial} reads facing {state.facing:.2f}; turn the board to "
                        f"square up with it (needs > {FACING_GOOD:.2f})")

        # Extrinsics come only from corners that two cameras BOTH measured. It is
        # not enough for each to see "the board" -- with a target larger than the
        # field of view they can easily be looking at opposite halves of it, which
        # contributes nothing to the multi-view solve.
        # Only the pair/group steps feed the multi-view solve, and only there is a
        # complete board reachable: with the target flat on the array, three of the
        # four views cannot contain all of it however it is placed, so gating the
        # world and face steps on this would deadlock the session.
        if step.min_shared:
            total = len(self.rig.object_points)
            # A 99 x 135 mm board on a 4:3 sensor barely fits with its long axis
            # upright, so demanding every corner can leave too few reachable poses.
            # What the solve needs is a large SHARED region, and calibrate.py
            # --common-corners already solves on one -- so this is a knob, not a law.
            need = int(round(self.args.in_frame_frac * total))
            for serial in watched:
                state = snapshot[serial]
                if state.in_frame < need:
                    return (f"{serial} has only {state.in_frame}/{total} corners inside "
                            f"the frame (need {need}) -- move the board "
                            f"{state.nudge or 'towards the centre'}")

        if step.min_shared:
            shared = None
            for serial in watched:
                ids = set(snapshot[serial].ids.tolist())
                shared = ids if shared is None else (shared & ids)
            if len(shared) < step.min_shared:
                names = " and ".join(watched)
                return (f"{names} share only {len(shared)} corners (needs "
                        f">= {step.min_shared}) -- angle the board so both see the "
                        f"same part of it")

        for state in seen:
            if state.moved > STILL_PIXELS:
                return "board still moving -- hold it steady for a second"

        if step.want_distinct_poses:
            state = snapshot[watched[0]]
            centre = state.corners.mean(axis=0)
            signature = np.array([centre[0], centre[1], state.facing * 600.0])
            for previous in step.signatures:
                if np.linalg.norm(signature - previous) < self.args.novelty:
                    return "this pose is already recorded -- change the angle or the distance"
            step.signatures.append(signature)
        return None

    def shared_with_watched(self, serial, snapshot, step) -> str:
        """How many corners this camera shares with the others this step needs.

        Two cameras each seeing "the board" is not enough: with a target larger
        than the field of view they can be looking at opposite halves, which
        gives the multi-view solve nothing. This is the number to steer by.
        """
        if step is None or not step.min_shared:
            return ""
        if step.pair is not None:
            others = [self.rig.serials[i] for i in step.pair]
        elif step.group is not None:
            others = [self.rig.serials[i] for i in step.group]
        else:
            return ""
        if serial not in others:
            return ""
        sets = [snapshot[s].ids for s in others if snapshot[s].ids is not None]
        if len(sets) < 2:
            return "  shared --"
        common = set(sets[0].tolist())
        for ids in sets[1:]:
            common &= set(ids.tolist())
        return f"  shared {len(common):2d}/{step.min_shared}"

    # ------------------------------------------------------------- preflight
    def preflight(self, snapshot):
        """Per-camera go/no-go before a minute of anyone's time is spent.

        These four are checked because each has already cost a whole session and
        none of them announces itself during capture: a defocused lens still
        detects corners, a clipped highlight still detects corners, a camera
        that has not seen the board contributes nothing to a step it is not
        watched in, and a camera that creeps makes the whole solve unsolvable
        while every individual step still looks fine.
        """
        rows, blocking = {}, []
        for serial in self.rig.serials:
            state = snapshot[serial]
            # Short names, because they are the card headline. What to do about
            # each one lives in FIX and is shown once, for the fault in hand.
            faults = []
            if state.sharp_scene <= 0:
                faults.append("MEASURING")
            elif state.sharp_scene >= SHARP_PX:
                faults.append("LENS OUT OF FOCUS")
            if state.corners is None:
                faults.append("NO BOARD")
            elif 0 < state.sharp_board < 1e6 and state.sharp_board >= SHARP_PX:
                faults.append("BOARD OUT OF FOCUS")
            if state.p99 < P99_DARK:
                faults.append("TOO DARK")
            elif state.p99 > P99_CLIPPED:
                faults.append("OVEREXPOSED")
            if state.drift >= DRIFT_PX:
                faults.append(f"MOVED {state.drift:.1f} px")
            rows[serial] = faults
            blocking += faults
        return rows, not blocking

    def start_capture(self, force=False):
        if self.phase != "preflight":
            return
        # Remember where every camera was standing when recording began, so a
        # knock during the session shows up as a change rather than as a level.
        self.drift_at_start = {s: self.rig.snapshot()[s].drift
                               for s in self.rig.serials}
        self.bumped = {}
        _, ready = self.preflight(self.rig.snapshot())
        if not ready and not force:
            self.notify("pre-flight not clear -- fix the red rows, or 'start anyway'", 4.0)
            return
        self.phase = "capture"
        self.preflight_bar.pack_forget()
        self.capture_bar.pack(fill="x", padx=12, pady=8)
        self.notify("recording -- follow the step above", 3.0)

    def auto_exposure(self):
        self.notify("balancing exposure...", 6.0)
        self.root.update_idletasks()
        self.notify(self.rig.auto_exposure(), 6.0)

    def set_card(self, serial, state, colour, headline, body):
        """Paint one camera card: border, coloured title bar, readouts."""
        panel = self.panels[serial]
        if state.preview is not None:
            photo = ImageTk.PhotoImage(
                Image.fromarray(cv.cvtColor(state.preview, cv.COLOR_BGR2RGB)))
            panel["image"].configure(image=photo)
            self.photo_refs[serial] = photo
        panel["rule"].configure(bg=colour)
        panel["header_label"].configure(fg=colour if colour != IDLE else MUTED)
        panel["header"].set(headline)
        panel["text"].set(body)

    def readouts(self, state, snapshot, step, serial):
        """The two lines under each preview, in the order they get acted on."""
        total = len(self.rig.object_points)
        count = 0 if state.ids is None else len(state.ids)
        shared = self.shared_with_watched(serial, snapshot, step)
        if state.corners is None:
            framing = "board not seen"
        elif state.in_frame >= total:
            framing = "whole board in frame"
        else:
            framing = f"in frame {state.in_frame}/{total} -> move {state.nudge}"
        bright = ("DARK" if state.p99 < P99_DARK else
                  "CLIPPED" if state.p99 > P99_CLIPPED else "ok")
        focus = ("--" if state.sharp_scene <= 0 else
                 f"{state.sharp_scene:.2f}"
                 + (f"/{state.sharp_board:.2f}" if state.sharp_board > 0 else ""))
        return chr(10).join([
            f"facing {state.facing:4.2f}  corners {count:3d}{shared}  "
            + ("still" if state.moved <= STILL_PIXELS else "MOVING"),
            framing,
            f"focus {focus}  drift {state.drift:.1f}px  p99 {state.p99:3.0f} {bright}",
        ])


    def render_preflight(self, snapshot):
        rows, ready = self.preflight(snapshot)
        self.title_var.set("Pre-flight  --  check the rig before spending half an hour")
        self.bar.configure(maximum=100)
        self.detail_var.set(
            "Put the board FLAT on the tactile array so every camera can see it, "
            "then leave the rig alone." + chr(10) +
            "Focus is judged on the 10-90% edge width: under "
            f"{SHARP_PX:g} px the corner refiner sees an edge, above it a ramp." + chr(10) +
            "Drift is how far the static scene has moved since this window opened "
            "-- let it run a few minutes before trusting it.")
        for serial in self.rig.serials:
            state = snapshot[serial]
            faults = rows[serial]
            colour = GOOD if not faults else BAD
            headline = f"{serial}   " + ("READY" if not faults else faults[0])
            self.set_card(serial, state, colour, headline,
                          self.readouts(state, snapshot, None, serial))
        self.progress_var.set("READY  --  press START CALIBRATION" if ready else
                              "NOT READY  --  " + ";   ".join(
                                  f"{s[-4:]} {rows[s][0]}" for s in rows if rows[s]))
        self.progress_label.configure(fg=GOOD if ready else BAD)
        first = next((v[0] for v in rows.values() if v), None)
        self.status_var.set(FIX.get(first.split()[0], "") if first else
                            "All four cameras pass. Nothing is recorded until you press "
                            "START CALIBRATION.")
        self.bar.configure(value=100.0 * sum(not v for v in rows.values()) / max(len(rows), 1))
        self.start_button.configure(state="normal" if ready else "disabled",
                                    bg=GOOD if ready else IDLE)

    def check_for_knocks(self, snapshot, step):
        """Catch a camera being knocked WHILE recording, not half an hour later.

        Every bump so far has happened during the hand-held steps, when an arm
        or the board itself passes close to a camera. It is worth interrupting
        for: poses banked before the knock and poses banked after describe two
        different rigs, and no single set of extrinsics fits both -- so the rest
        of the session is being spent on data that cannot be combined with what
        is already on disk.
        """
        for serial in self.rig.serials:
            moved = snapshot[serial].drift - self.drift_at_start.get(serial, 0.0)
            if moved >= DRIFT_PX and serial not in self.bumped:
                self.bumped[serial] = (moved, step.key if step else "?")
        return self.bumped

    def tick(self):
        if not self.running:
            return
        snapshot = self.rig.snapshot()
        step = self.step() if self.phase == "capture" else None
        if self.phase == "capture":
            self.check_for_knocks(snapshot, step)

        for serial in self.rig.serials:
            state = snapshot[serial]
            # Grey means "this camera is not part of the current step" -- the
            # operator should not be trying to satisfy four cards at once.
            colour, tag = IDLE, "not needed this step"
            if step is not None:
                watched = (step.target is not None
                           and serial == self.rig.serials[step.target])
                in_pair = step.pair is not None and serial in (
                    self.rig.serials[step.pair[0]], self.rig.serials[step.pair[1]])
                in_group = step.group is not None and serial in [
                    self.rig.serials[k] for k in step.group]
                if watched:
                    ok = state.facing >= FACING_GOOD
                    colour = GOOD if ok else WARN
                    tag = "SQUARE ON" if ok else f"TURN TOWARDS ME ({state.facing:.2f})"
                elif in_pair or in_group:
                    need = self.args.in_frame_frac * len(self.rig.object_points)
                    ok = state.corners is not None and state.in_frame >= need
                    colour = GOOD if ok else WARN
                    tag = "IN VIEW" if ok else (
                        "BOARD NOT SEEN" if state.corners is None
                        else f"MOVE {state.nudge.upper() or 'TO CENTRE'}")
            self.set_card(serial, state, colour, f"{serial}   {tag}",
                          self.readouts(state, snapshot, step, serial))

        if self.phase == "preflight":
            self.render_preflight(snapshot)
            self.root.after(self.args.interval_ms, self.tick)
            return

        if step is None:
            self.title_var.set("All steps complete")
            self.detail_var.set("Press FINISH AND SAVE to write the session. The window "
                                "then closes and the calibration is solved and graded "
                                "for you -- that takes a few minutes.")
            self.progress_var.set("Done -- press FINISH AND SAVE")
            self.progress_label.configure(fg=GOOD)
            self.bar.configure(value=100)
            self.status_var.set("")
        else:
            self.title_var.set(step.title)
            self.detail_var.set(step.detail)
            self.progress_var.set(
                f"{step.captured} of {step.needed} poses recorded"
                f"        step {self.current + 1}/{len(self.steps)}"
                f"        {self.writer.index} frame sets on disk")
            if self.bumped:
                names = ", ".join(f"{s[-4:]} +{v:.0f}px during '{k}'"
                                  for s, (v, k) in self.bumped.items())
                self.progress_var.set(f"KNOCKED: {names}  --  poses before and after "
                                      f"this do not describe the same rig")
                self.progress_label.configure(fg=BAD)
            else:
                self.progress_label.configure(fg=GOOD if step.captured else MUTED)
            done = sum(min(t.captured, t.needed) for t in self.steps)
            self.bar.configure(value=100.0 * done / sum(t.needed for t in self.steps))
            reason = self.accept(step, snapshot)
            if self.args.debug:
                print(f"[tick] step={step.key} captured={step.captured} "
                      f"reason={reason!r}", flush=True)
            if reason is None:
                index = self.writer.save_set(snapshot)
                step.captured += 1
                self.notify(f"recorded set {step.captured} (frame {index})", 1.2)
                if step.done:
                    self.current += 1
                    self.notify("step complete -- read the next instruction above", 2.5)
            elif time.monotonic() > self.message_until:
                self.message = reason
            self.status_var.set(self.message)

        self.root.after(self.args.interval_ms, self.tick)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--interval-ms", type=int, default=120)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--novelty", type=float, default=90.0,
                        help="how different a pose must be from ones already kept")
    parser.add_argument("--trigger-hz", type=float, default=24.2)
    parser.add_argument("--debug", action="store_true",
                        help="print the accept/reject decision every tick")
    parser.add_argument("--in-frame-frac", type=float, default=0.85,
                        help="fraction of the board that must be inside the frame "
                             "before a pair/group set is recorded. 1.0 is what the "
                             "strict solve wants; below that the solve falls back "
                             "to --common-corners on the shared sub-pattern")
    parser.add_argument("--no-solve", action="store_true",
                        help="just save; skip the solve and the quality verdict")
    parser.add_argument("--calib-python", type=Path, default=None,
                        help="interpreter of the calibration venv; found "
                             "automatically at <repo>/.venv-calib")
    parser.add_argument("--autoclose", type=float, default=0.0,
                        help="close and save after N seconds; for smoke-testing only")
    args = parser.parse_args()

    spec = BoardSpec.load(args.board)
    dictionary = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, spec.dictionary))
    board = cv.aruco.CharucoBoard((spec.squares_x, spec.squares_y),
                                  float(spec.square_mm), float(spec.marker_mm),
                                  dictionary)
    board.setLegacyPattern(bool(spec.legacy_pattern))
    object_points = np.asarray(board.getChessboardCorners(), np.float32).reshape(-1, 3)
    print(f"board: {spec.describe()}")

    # The window comes first so the preview resolution can be chosen from the
    # display this is actually running on, before the grab loop starts filling
    # the buffers with the wrong size.
    root = tk.Tk()
    root.withdraw()

    rig = Rig(board, object_points, args.fps, args.timeout_ms)
    print(f"cameras: {', '.join(rig.serials)}")
    # The layout depends on how many cameras there are, which only the rig knows;
    # preview_size is read afresh every grab, so assigning it now is enough.
    rig.preview_size = ui_scale(root, cameras=len(rig.serials))["preview"]
    writer = SessionWriter(args.output, rig.serials, args.board)
    print(f"session: {writer.root}")

    root.deiconify()
    app = App(root, rig, build_plan(rig.serials), writer, args)
    if args.autoclose > 0:
        root.after(int(args.autoclose * 1000), app.finish)
    try:
        root.mainloop()
    finally:
        if app.running:
            rig.close(args.trigger_hz)

    if app.saved_path is None:
        return 0
    if args.no_solve:
        print(f"\nnot graded (--no-solve). Session: {app.saved_path}")
        return 0
    calib_python = session_report.find_calib_python(args.calib_python)
    return session_report.report(
        app.saved_path, args.board.resolve(), calib_python,
        [(s.key, s.captured, s.needed) for s in app.steps])


if __name__ == "__main__":
    raise SystemExit(main())
