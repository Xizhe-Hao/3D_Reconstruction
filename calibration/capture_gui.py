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
only when the pose is genuinely right: the board still, square-on to the cameras
that need it, sharing enough corners with its partner, and -- for the steps
that feed the multi-view solve -- entirely inside the frame. Detecting *the
board* is not the same as seeing *all of* it, and the strict solve uses only
frames where every corner was measured.

The plan is not fixed. Step 1 (board flat on the platform, the world frame) is
also where every camera sees the same board, so the window works out from it
where each camera stands around the board and plans the rest from that: one
step per pair of cameras that really are neighbours (both see the board
square-on at once, which feeds both intrinsics and their link), a face step
only for a camera without a neighbour, and one all-cameras step. On this rig
that is 5 steps instead of 9. The old fixed plan paired cameras by number, and
two of its pairs face each other across the array, which no flat board serves.
Pair steps also walk the board into the corners of each image, where lens
distortion lives.

The cameras are fired together by software trigger, so the images of one set
are exposed within ~3 ms of each other. Free-running, they were up to a few
hundred milliseconds apart, and a hand-held board that drifted in between
became a different board in each view (18 of 92 frames on the 2026-09-30
session). It still does not need the Arduino: calibration does not depend on
the sensor rig running at all.

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
import rig_layout  # noqa: E402
import session_report  # noqa: E402
from tune_exposure import (prepare_for_capture, set_bool, set_enum, set_float,  # noqa: E402
                           throttle)

# GenICam BayerRG8 is OpenCV's BayerBG -- the two conventions are one pixel
# apart. session_io.py documents this; keep the two in step.
BAYER_TO_GRAY = cv.COLOR_BayerBG2GRAY
BAYER_TO_BGR = cv.COLOR_BayerBG2BGR

FACING_GOOD = 0.55
# Both cameras of a pair step must see the board at least this square. 0.45 was
# too much for this rig on 2026-10-08: with the cameras 15-25 cm from the
# board, turning it further towards one of them pushed it out of the other.
PAIR_FACING = 0.35
# Per-corner movement between two consecutive frame sets to count as still, and
# how many consecutive still sets are needed. The cameras are fired together by
# software trigger, so the four images of one set are taken within ~3 ms of each
# other; what is left to guard against is motion blur within the exposure.
STILL_PIXELS = 1.0
STILL_SETS = 2
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
            "Focus on the OBJECT PLATFORM, not on the background.",
    "NO": "That camera cannot see the board. Put the board flat on the platform "
          "(soft object removed) where all four cameras can see it.",
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
    # Steps that feed intrinsics also have to walk the board into the corners
    # of each watched image: lens distortion is largest there, and on the
    # 2026-09-30 session no camera ever saw the board's centre outside the
    # middle half of its frame, so the edges were extrapolated.
    want_coverage: bool = False
    # Fraction of the board that must be inside every watched frame; None uses
    # --in-frame-frac. (Pair steps briefly demanded all of it; in the confined
    # space of this rig that let three poses through in a whole step, and the
    # bundle adjustment uses partial views anyway.)
    frame_frac: float | None = None
    pair_facing: float = PAIR_FACING     # both cameras of a pair step, at least
    captured: int = 0
    signatures: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)   # camera index -> covered cells

    def coverage_short(self) -> list[int]:
        """Watched cameras whose image is not yet covered well enough."""
        if not self.want_coverage:
            return []
        cameras = list(self.pair) if self.pair else [self.target]
        return [c for c in cameras
                if len(self.coverage.get(c, ())) < rig_layout.COVERAGE_GOAL]

    @property
    def done(self) -> bool:
        if self.captured < self.needed:
            return False
        # Coverage may be geometrically out of reach for one camera of a pair,
        # so it extends a step by a few poses at most instead of blocking it.
        return not self.coverage_short() or self.captured >= self.needed + 6


WORLD_STEP_NEEDED = 12


def build_plan(serials: list[str]) -> list[Step]:
    """Only the world step is fixed; the rest is planned from what it shows.

    The world step is where every camera sees the same board lying flat, which
    is exactly what is needed to work out where the cameras stand around it --
    see `plan_after_world`.
    """
    return [Step(
        key="world",
        title="Step 1  --  world frame",
        detail=("Take the soft object (sponge) off and lay the board FLAT on the "
                "platform where it sat, printed side up. Then LET GO. This defines "
                "the world origin (z = 0 is this surface); the same pose is "
                f"recorded {WORLD_STEP_NEEDED}x for averaging, so do not touch it."),
        needed=WORLD_STEP_NEEDED,
        want_distinct_poses=False,
    )]


EXTRINSICS_SHARED = 12      # shared corners per frame when only extrinsics are solved
EXTRINSICS_FRAME_FRAC = 0.6
EXTRINSICS_MAX_PAIR_GAP = 70.0


def plan_extrinsics_only(serials: list[str], azimuths: dict[int, float]) -> list[Step]:
    """The plan when the intrinsics are reused (--intrinsics-from).

    Facing and image-corner coverage exist to pin down focal length, principal
    point and distortion. With those fixed, a frame only has to tell where the
    cameras are relative to each other, which any shared set of corners does,
    seen square-on or not -- so those requirements are dropped, and partial
    boards are accepted because the bundle adjustment uses every detected
    corner. The cameras only have to form a connected graph: one all-cameras
    block (the board nearly flat is seen by three or four downward-looking
    cameras at once, the easiest pose on this rig) plus the clearly adjacent
    pairs. Wide pairs such as 1+2 (~95 deg) are left to the all-cameras block.
    """
    cameras = list(range(len(serials)))
    pairs = (rig_layout.neighbour_pairs(azimuths, EXTRINSICS_MAX_PAIR_GAP)
             if len(azimuths) == len(serials) else [])
    total = 2 + len(pairs)
    steps = [Step(
        key="all",
        title=f"Step 2 of {total}  --  all cameras together (extrinsics only)",
        detail=("Intrinsics are reused, so only shared views count -- no facing or "
                "corner requirements. Lay or prop the board nearly flat where three "
                "or more cameras see it; most of the board is enough. Move it after "
                "every shot: different places on and around the object, flat and "
                "tilted up to ~25 deg, on the platform and a few cm up."),
        needed=14,
        group=cameras,
        min_shared=EXTRINSICS_SHARED,
        frame_frac=EXTRINSICS_FRAME_FRAC,
    )]
    for number, (first, second) in enumerate(pairs, start=3):
        steps.append(Step(
            key=f"pair{first}{second}",
            title=(f"Step {number} of {total}  --  cameras {first} + {second} "
                   f"({serials[first][-4:]} + {serials[second][-4:]}), extrinsics only"),
            detail=("Any pose both cameras see most of the board in, still. Angle "
                    "does not matter now. A few varied poses are enough; "
                    "'skip this step' keeps what you have."),
            needed=6,
            pair=(first, second),
            min_shared=EXTRINSICS_SHARED,
            frame_frac=EXTRINSICS_FRAME_FRAC,
            pair_facing=0.0,
        ))
    return steps


def plan_after_world(serials: list[str], azimuths: dict[int, float]) -> list[Step]:
    """Pair steps for cameras that are really next to each other, then one
    all-cameras step.

    A pair step gives both of its cameras a square-on view of the whole board
    at once, so it feeds both intrinsics AND the link between them -- which is
    why the separate per-camera "face" steps are gone except for a camera that
    has no neighbour close enough to pair with.
    """
    cameras = list(range(len(serials)))
    pairs = rig_layout.neighbour_pairs(azimuths) if len(azimuths) == len(serials) else []
    solo = rig_layout.unpaired(cameras, pairs) if pairs else cameras
    steps: list[Step] = []
    total = 1 + len(pairs) + len(solo) + 1
    number = 2
    for first, second in pairs:
        steps.append(Step(
            key=f"pair{first}{second}",
            title=(f"Step {number} of {total}  --  cameras {first} + {second} "
                   f"({serials[first][-4:]} + {serials[second][-4:]})"),
            detail=("Hold or rest the board so it faces BETWEEN these two cameras; "
                    "it records once both see most of it and it is still. It does "
                    "not have to sit on the platform -- anywhere both cameras see "
                    "it works, a few cm up is often easier. Change the tilt and "
                    "position after every shot, and reach towards the CORNERS of "
                    "both images if you can (the cards say which is missing). If a "
                    "pose is impossible here, 'skip this step' keeps what you have."),
            needed=8,
            pair=(first, second),
            min_shared=SHARED_CORNERS,
            want_coverage=True,
        ))
        number += 1
    for index in solo:
        steps.append(Step(
            key=f"face{index}",
            title=f"Step {number} of {total}  --  face camera {index} ({serials[index]})",
            detail=(f"No other camera stands close to {serials[index]}, so it gets its "
                    "own step. Rest the board facing it squarely "
                    f"(facing > {FACING_GOOD:.2f}), hands off; then move it so it "
                    "reaches every corner of that camera's image."),
            needed=8,
            target=index,
            want_coverage=True,
        ))
        number += 1
    steps.append(Step(
        key="all",
        title=f"Step {number} of {total}  --  all cameras together",
        detail=("Lay the board nearly FLAT (tilt under ~15 deg) in the space the "
                "object will occupy, so at least three cameras see it. Move it to "
                "a new place or height after every shot -- on the platform, then "
                "raised a few centimetres. This ties both sides of the rig together."),
        needed=8,
        group=cameras,
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
    still: int = 0              # consecutive frame sets with the board motionless


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

        self.period = 1.0 / max(fps, 0.5)
        self.trigger_source = {}
        self.triggered = self._arm_software_trigger()
        for _, camera in self.handles:
            if not self.triggered:
                throttle(camera, fps)
            camera.BeginAcquisition()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _arm_software_trigger(self) -> bool:
        """Fire every camera from one software command, so a set is one instant.

        Free-running, each camera's newest image is taken whenever its own clock
        says, and this loop reads them one after another with a corner detection
        in between -- so the four images of one "set" were up to a few hundred
        milliseconds apart. A hand-held board moves in that time, and the frame
        then describes a different board in every view. On the 2026-09-30
        session 18 of 92 frames were inconsistent that way. Triggering all
        cameras back to back puts their exposures within ~3 ms (measured on this
        rig), which no hand can move a board in.
        """
        try:
            for serial, camera in self.handles:
                nodemap = camera.GetNodeMap()
                source = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerSource"))
                if PySpin.IsReadable(source):
                    self.trigger_source[serial] = source.GetCurrentEntry().GetSymbolic()
                set_bool(nodemap, "AcquisitionFrameRateEnable", False)
                ok = (set_enum(nodemap, "TriggerMode", "Off")
                      and set_enum(nodemap, "TriggerSelector", "FrameStart")
                      and set_enum(nodemap, "TriggerSource", "Software")
                      and set_enum(nodemap, "TriggerMode", "On"))
                set_enum(camera.GetTLStreamNodeMap(), "StreamBufferHandlingMode",
                         "NewestOnly")
                if not ok:
                    raise PySpin.SpinnakerException("software trigger not writable")
            return True
        except PySpin.SpinnakerException as error:
            print(f"software trigger unavailable ({error}); falling back to free-run, "
                  "where the cameras of one set are NOT exposed together")
            for _, camera in self.handles:
                set_enum(camera.GetNodeMap(), "TriggerMode", "Off")
            return False

    def _fire(self):
        for _, camera in self.handles:
            try:
                PySpin.CCommandPtr(camera.GetNodeMap().GetNode("TriggerSoftware")).Execute()
            except PySpin.SpinnakerException:
                pass

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
        # findHomography returns None on a degenerate point set -- corners that
        # came back collinear, which a board held almost edge-on really does
        # produce. The framing is then unknowable rather than zero, but the
        # detection is still worth reporting, so fall back to what the
        # no-detection path above returns instead of leaving these unbound.
        in_frame, nudge = 0, ""
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

        # Mask the board out: it is the most feature-rich thing in view and it
        # is moved on purpose, so left in it out-votes the background and is
        # reported as camera drift (239 px on 2026-10-14 after the board was
        # simply slid across the platform).
        mask = None
        if corners is not None and len(corners) >= 4:
            mask = np.full(gray.shape, 255, np.uint8)
            hull = cv.convexHull(np.asarray(corners, np.float32).reshape(-1, 1, 2))
            cv.fillConvexPoly(mask, hull.astype(np.int32), 0)
            mask = cv.erode(mask, np.ones((121, 121), np.uint8))
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
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
            started = time.monotonic()
            # Grab the whole set first, then process it: detection takes tens of
            # milliseconds per image, and in free-run that delay would separate
            # the cameras' exposures.
            if self.triggered:
                self._fire()
            grabbed = []
            for serial, camera in self.handles:
                try:
                    image = camera.GetNextImage(self.timeout_ms)
                except PySpin.SpinnakerException:
                    continue
                try:
                    if image.IsIncomplete():
                        continue
                    grabbed.append((serial, np.array(image.GetNDArray(), copy=True),
                                    time.time_ns()))
                finally:
                    image.Release()
            # Publish the whole set at once. Updating camera by camera let the
            # GUI snapshot a mix of this set and the previous one, and save it
            # as if it were one instant (measured: 396 ms apart).
            updates = [self._process(serial, raw, stamp) for serial, raw, stamp in grabbed]
            with self.lock:
                for update in updates:
                    update()
            if self.triggered:
                time.sleep(max(0.0, self.period - (time.monotonic() - started)))

    def _process(self, serial, raw, stamp):
        gray = cv.cvtColor(raw, BAYER_TO_GRAY) if raw.ndim == 2 else raw
        corners, ids, facing, in_frame, nudge = self._detect(serial, gray)
        self._monitor(serial, gray, corners)

        moved = 1e9
        earlier = self.previous[serial]
        current = None
        if corners is not None:
            # searchsorted needs sorted ids, and the detector does not return
            # them in order -- comparing unsorted lists matched the wrong corners
            # and made "still" a coin toss.
            order = np.argsort(ids)
            current = (ids[order], corners[order])
        if current is not None and earlier is not None:
            (new_ids, new_corners), (old_ids, old_corners) = current, earlier
            shared = np.intersect1d(new_ids, old_ids)
            if len(shared) >= 6:
                a = new_corners[np.searchsorted(new_ids, shared)]
                b = old_corners[np.searchsorted(old_ids, shared)]
                moved = float(np.linalg.norm(a - b, axis=1).mean())
        self.previous[serial] = current

        preview = cv.cvtColor(raw, BAYER_TO_BGR) if raw.ndim == 2 else raw
        preview = cv.resize(preview, self.preview_size,
                            interpolation=cv.INTER_AREA)
        if corners is not None:
            scale = np.array([self.preview_size[0] / raw.shape[1],
                              self.preview_size[1] / raw.shape[0]])
            dot = max(2, self.preview_size[0] // 200)
            for point in (corners * scale).astype(int):
                cv.circle(preview, tuple(point), dot, (0, 255, 255), -1)

        p99 = float(np.percentile(raw, 99))

        def update():               # called with self.lock held
            state = self.state[serial]
            state.raw = raw
            state.preview = preview
            state.corners = corners
            state.ids = ids
            state.facing = facing
            state.in_frame = in_frame
            state.nudge = nudge
            state.p99 = p99
            state.moved = moved
            state.still = state.still + 1 if moved <= STILL_PIXELS else 0
            state.stamp_ns = stamp
        return update

    def snapshot(self) -> dict[str, CameraState]:
        with self.lock:
            return {serial: CameraState(
                serial=state.serial, raw=state.raw, preview=state.preview,
                corners=state.corners, ids=state.ids, facing=state.facing,
                p99=state.p99, moved=state.moved, stamp_ns=state.stamp_ns,
                in_frame=state.in_frame, nudge=state.nudge,
                sharp_scene=state.sharp_scene, sharp_board=state.sharp_board,
                drift=state.drift, still=state.still)
                for serial, state in self.state.items()}

    def auto_exposure(self, target_p99: float = 205.0, max_exposure_us: float = 60000.0,
                      max_gain_db: float = 6.0, iterations: int = 5) -> str:
        """Bring every camera to the same brightness, each with its own settings.

        Exposure first and gain only for the shortfall: gain is amplification,
        so it buys brightness at the price of the noise that limits sub-pixel
        corner localisation. The board is held still and the cameras are
        software-triggered, so a long exposure costs nothing here; gain is
        capped at 6 dB (it used to be 18). The live p99 the monitor already maintains is the
        measurement, so this needs no extra grabs of its own.
        """
        # The cameras' own auto exposure/gain overwrite every value written
        # while they run -- and they come back on whenever a camera loses
        # power (a replugged cable). On 2026-10-09 that left this loop
        # "setting" 60 ms while the cameras sat at 15 ms and 17.9 dB.
        for _, camera in self.handles:
            nodemap = camera.GetNodeMap()
            set_enum(nodemap, "ExposureAuto", "Off")
            set_enum(nodemap, "GainAuto", "Off")
            # Start from no gain: the loop below only ever adds gain for a
            # shortfall exposure cannot cover, so gain left over by the camera's
            # own auto mode would otherwise stay forever.
            set_float(nodemap, "Gain", 0.0)
        time.sleep(max(0.6, self.monitor_period * 0.4))   # let p99 refresh
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
        for serial, camera in self.handles:
            try:
                camera.EndAcquisition()
            except PySpin.SpinnakerException:
                pass
            try:
                # Trigger mode off before the source can be written back; the
                # capture script sets its own source, but leave things as found.
                set_enum(camera.GetNodeMap(), "TriggerMode", "Off")
                if serial in self.trigger_source:
                    set_enum(camera.GetNodeMap(), "TriggerSource",
                             self.trigger_source[serial])
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

    def finish(self, steps: list[Step], extra: dict | None = None) -> Path:
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
            "trigger_source": (extra or {}).get(
                "trigger_source", "none (guided free-run, board verified stationary)"),
            "camera_stats": [{"camera_index": i, "serial": s,
                              "source_pixel_format": "BayerRG8",
                              "saved_pixel_format": "BayerRG8"}
                             for i, s in enumerate(self.serials)],
            "guided_capture": True,
            "board": str(self.board_path),
            "world_frames": [0, world[0].captured if world else 0],
            "steps": [{"key": s.key, "captured": s.captured, "needed": s.needed}
                      for s in steps],
            "layout": (extra or {}).get("layout", {}),
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
        self.world_views = {}       # camera index -> (ids, corners, size) of the world pose
        self.layout = {}            # what plan_after_world was given, for session.json

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
        if getattr(args, "auto_exposure", True):
            # Balance exposure once the cameras have delivered a few frames, so
            # every session starts from the lighting actually in the room
            # rather than from whatever the last tool left in the cameras.
            self.root.after(2500, self.startup_exposure)

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
        step = self.step()
        if step is not None:
            if step.key == "world":
                self.plan_rest()
            self.current += 1
            self.notify("step skipped")

    def nudge(self, factor: float):
        self.notify("exposure -> " + self.rig.scale_exposure(factor) + " us", 3.0)

    def finish(self):
        self.running = False
        self.saved_path = self.writer.finish(self.steps, {
            "trigger_source": ("software (all cameras fired together)"
                               if self.rig.triggered else
                               "none (guided free-run, board verified stationary)"),
            "layout": self.layout})
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
            if step.pair is not None and state.facing < step.pair_facing:
                return (f"{serial} reads facing {state.facing:.2f}; turn the board "
                        f"towards it (both cameras need > {step.pair_facing:.2f})")

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
            frac = step.frame_frac if step.frame_frac is not None else self.args.in_frame_frac
            need = int(round(frac * total))
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

        # Only the cameras this step needs: a camera catching a flickering edge
        # of the board never accumulates still sets and would block the step,
        # and with every camera fired together its image is of the same
        # instant anyway.
        for serial in watched:
            if snapshot[serial].still < STILL_SETS:
                return ("board still moving -- rest it on something and take your "
                        "hands off")

        if step.want_distinct_poses:
            state = snapshot[watched[0]]
            centre = state.corners.mean(axis=0)
            signature = np.array([centre[0], centre[1], state.facing * 600.0])
            gains = self.coverage_gain(step, snapshot)
            if step.captured >= step.needed and step.coverage_short() and not gains:
                return ("enough poses here -- move the board towards the "
                        + self.coverage_hint(step))
            if not gains:
                for previous in step.signatures:
                    if np.linalg.norm(signature - previous) < self.args.novelty:
                        return ("this pose is already recorded -- change the angle "
                                "or the distance")
            step.signatures.append(signature)
        return None

    def coverage_gain(self, step: Step, snapshot) -> bool:
        """Would recording this pose reach a part of a watched image not yet covered?"""
        if not step.want_coverage:
            return False
        for index in (step.pair or (step.target,)):
            state = snapshot[self.rig.serials[index]]
            if state.corners is None or state.raw is None:
                continue
            height, width = state.raw.shape[:2]
            cells = rig_layout.coverage_cells(state.corners, width, height)
            if cells - step.coverage.get(index, set()):
                return True
        return False

    def record_coverage(self, step: Step, snapshot):
        if not step.want_coverage:
            return
        for index in (step.pair or (step.target,)):
            state = snapshot[self.rig.serials[index]]
            if state.corners is None or state.raw is None:
                continue
            height, width = state.raw.shape[:2]
            step.coverage.setdefault(index, set()).update(
                rig_layout.coverage_cells(state.corners, width, height))

    def coverage_hint(self, step: Step) -> str:
        short = step.coverage_short()
        if not short:
            return "anywhere new"
        index = short[0]
        missing = rig_layout.missing_cells(step.coverage.get(index, set()))
        return (f"{missing[0].upper()} of camera {index}'s image"
                if missing else "anywhere new")

    def plan_rest(self):
        """Work out where the cameras stand from the world pose, then plan."""
        azimuths = {}
        for index, serial in enumerate(self.rig.serials):
            view = self.world_views.get(index)
            if view is None:
                continue
            ids, corners, size = view
            K, dist = self.args.prior_intrinsics.get(
                serial, (rig_layout.nominal_K(*size), None))
            azimuth = rig_layout.camera_azimuth(
                self.rig.object_points.astype(np.float64), ids, corners, K, dist)
            if azimuth is not None:
                azimuths[index] = azimuth
        if self.args.intrinsics_from:
            rest = plan_extrinsics_only(self.rig.serials, azimuths)
        else:
            rest = plan_after_world(self.rig.serials, azimuths)
        self.steps.extend(rest)
        self.layout = {"azimuth_deg": {str(k): round(v, 1) for k, v in azimuths.items()},
                       "steps": [step.key for step in rest]}
        ring = ", ".join(f"cam{k} {v:.0f}" for k, v in
                         sorted(azimuths.items(), key=lambda item: item[1]))
        print(f"camera azimuths around the board (deg): {ring}")
        print("plan: " + " -> ".join(step.key for step in self.steps))

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
            # With reused intrinsics (--intrinsics-from) refocusing would void
            # them, so a soft BOARD reading -- usually a board seen nearly
            # edge-on, whose foreshortened edges read soft -- only warns.
            blocking += [f for f in faults
                         if not (f == "BOARD OUT OF FOCUS" and self.args.intrinsics_from)]
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

    def startup_exposure(self):
        if not self.running or self.phase != "preflight":
            return
        if not any(state.p99 > 0 for state in self.rig.snapshot().values()):
            self.root.after(1000, self.startup_exposure)    # no frame yet
            return
        print("auto exposure at start-up (skip with --no-auto-exposure)", flush=True)
        self.auto_exposure()
        print(self.message, flush=True)

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
        cover = ""
        index = self.rig.serials.index(serial)
        if step is not None and step.want_coverage and index in (step.pair or (step.target,)):
            covered = step.coverage.get(index, set())
            missing = rig_layout.missing_cells(covered)
            cover = (f"   cover {len(covered)}/9"
                     + (f" -> {missing[0]}" if missing and
                        len(covered) < rig_layout.COVERAGE_GOAL else " ok"))
        return chr(10).join([
            f"facing {state.facing:4.2f}  corners {count:3d}{shared}  "
            + ("still" if state.still >= STILL_SETS else "MOVING"),
            framing + cover,
            f"focus {focus}  drift {state.drift:.1f}px  p99 {state.p99:3.0f} {bright}",
        ])


    def render_preflight(self, snapshot):
        rows, ready = self.preflight(snapshot)
        self.title_var.set("Pre-flight  --  check the rig before spending half an hour")
        self.bar.configure(maximum=100)
        self.detail_var.set(
            "Take the soft object off and put the board FLAT on the platform so every "
            "camera can see it, then leave the rig alone." + chr(10) +
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
        advice = FIX.get(first.split()[0], "") if first else ""
        if first == "BOARD OUT OF FOCUS" and self.args.intrinsics_from:
            advice = ("DO NOT touch the focus ring: the reused intrinsics belong to the "
                      "current focus. A board seen nearly edge-on reads soft; move it a "
                      "little or go ahead -- this warning does not block the start.")
        self.status_var.set(advice if first else
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
                    frac = (step.frame_frac if step.frame_frac is not None
                            else self.args.in_frame_frac)
                    need = frac * len(self.rig.object_points)
                    ok = state.corners is not None and state.in_frame >= need
                    colour = GOOD if ok else WARN
                    tag = "IN VIEW" if ok else (
                        "BOARD NOT SEEN" if state.corners is None
                        else f"MOVE {state.nudge.upper() or 'TO CENTRE'}")
                    if ok and in_pair and state.facing < step.pair_facing:
                        colour, tag = WARN, f"TURN TOWARDS ME ({state.facing:.2f})"
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
                self.record_coverage(step, snapshot)
                if step.key == "world":
                    self.world_views = {
                        i: (snapshot[s].ids, snapshot[s].corners,
                            snapshot[s].raw.shape[1::-1])
                        for i, s in enumerate(self.rig.serials)
                        if snapshot[s].corners is not None and snapshot[s].raw is not None}
                self.notify(f"recorded set {step.captured} (frame {index})", 1.2)
                if step.done:
                    if step.key == "world":
                        self.plan_rest()
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
    parser.add_argument("--fps", type=float, default=6.0,
                        help="frame sets per second (the cameras are fired together)")
    parser.add_argument("--interval-ms", type=int, default=120)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--novelty", type=float, default=90.0,
                        help="how different a pose must be from ones already kept")
    parser.add_argument("--trigger-hz", type=float, default=24.2)
    parser.add_argument("--debug", action="store_true",
                        help="print the accept/reject decision every tick")
    parser.add_argument("--in-frame-frac", type=float, default=0.85,
                        help="fraction of the board that must be inside the frame "
                             "before an all-cameras set is recorded. Pair steps "
                             "always demand the whole board, which is what the "
                             "strict solve uses")
    parser.add_argument("--no-solve", action="store_true",
                        help="just save; skip the solve and the quality verdict")
    parser.add_argument("--calib-python", type=Path, default=None,
                        help="interpreter of the calibration venv; found "
                             "automatically at <repo>/.venv-calib")
    parser.add_argument("--no-auto-exposure", dest="auto_exposure", action="store_false",
                        help="keep the cameras' current exposure instead of balancing it "
                             "automatically when the window opens")
    parser.add_argument("--prior-calibration", type=Path,
                        default=Path(__file__).resolve().parent / "calibration.json",
                        help="an earlier calibration.json; its intrinsics only help "
                             "work out where the cameras stand, so an outdated one "
                             "is fine and a missing one falls back to a nominal lens")
    parser.add_argument("--intrinsics-from", type=Path, default=None,
                        help="solve with the K and distortion of an earlier "
                             "calibration.json instead of fitting them. Valid while "
                             "no lens has been refocused or had its iris touched; the "
                             "session then only has to locate the cameras, which "
                             "needs far fewer poses (calibrate.py checks the fit)")
    parser.add_argument("--world-z-offset", type=float, default=None, metavar="MM",
                        help="passed to calibrate.py: put z = 0 on the platform, not "
                             "on the printed face, e.g. -1.2 for the board on its glass")
    parser.add_argument("--autoclose", type=float, default=0.0,
                        help="close and save after N seconds; for smoke-testing only")
    args = parser.parse_args()

    args.prior_intrinsics = {}
    if args.prior_calibration and args.prior_calibration.exists():
        prior = json.loads(args.prior_calibration.read_text(encoding="utf-8"))
        for camera in prior.get("cameras", []):
            args.prior_intrinsics[camera["serial"]] = (
                np.asarray(camera["K"], np.float64),
                np.asarray(camera["dist"], np.float64))

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
    extra = []
    if args.intrinsics_from:
        extra += ["--intrinsics-from", str(args.intrinsics_from.resolve())]
    if args.world_z_offset is not None:
        extra += ["--world-z-offset", str(args.world_z_offset)]
    return session_report.report(
        app.saved_path, args.board.resolve(), calib_python,
        [(s.key, s.captured, s.needed) for s in app.steps], extra)


if __name__ == "__main__":
    raise SystemExit(main())
