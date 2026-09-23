# Multi-camera calibration for the tactile rig

Solves intrinsics + extrinsics for the Blackfly S cameras in
[`sensor-data-collection/`](../sensor-data-collection), expressed in the frame of
the tactile array, in millimetres — so a reconstruction made from these cameras
is directly comparable to the 16×16 sensor underneath it.

Built on OpenCV 5's `cv::calibrateMultiview` (per-camera intrinsics → maximum
spanning tree of camera pairs → global Levenberg-Marquardt). The rig's Arduino
D9 trigger already makes every camera expose at the same instant, which is the
hard prerequisite that multi-view extrinsics calibration cannot work without.

```
make_board.py     -> board.json + a printable ChArUco target
capture_gui.py    -> a guided session, recorded only when the pose is good
detect_corners.py -> detections.npz   (one capture session -> corner observations)
calibrate.py      -> calibration.json (intrinsics, extrinsics, world frame)
validate.py       -> PASS/WARN/FAIL table + metric error in millimetres
inspect_calib.py  -> look at the error instead of reading it: re-projection,
                     epipolar lines, and the rig in 3D
```

The capture-side tools (`capture_gui`, `pose_helper`, `tune_exposure`,
`check_trigger`, `reset_camera_auto_exposure`) run in the **capture** venv
because they need PySpin; everything else runs in the calibration venv.

---

## 1. Install

`calibrateMultiview` exists **only in OpenCV 5.x** — it is absent from every
4.x release. The `opencv-python` 5.0 wheel requires `numpy>=2`, which conflicts
with the `numpy==1.26.4` that PySpin needs, so calibration gets its own
environment. That costs nothing: calibration is offline post-processing of saved
BMPs and never touches the camera SDK.

```powershell
py -3.10 -m venv .venv-calib
.\.venv-calib\Scripts\Activate.ps1
python -m pip install -r calibration\requirements.txt
```

ArUco lives in OpenCV's main `objdetect` module since 4.7, so `opencv-python` is
enough — `opencv-contrib-python` is not needed.

## 2. Print the target

```powershell
# the board currently in board\ -- sized for the 116 x 178 mm glass plate
python calibration\make_board.py --out-dir calibration\board `
    --squares-x 11 --squares-y 15 --square-mm 9 --marker-mm 6.75 `
    --dict DICT_4X4_100 --margin-mm 7
```

That is a 99 x 135 mm board on a 113 x 171 mm page (7 mm white quiet zone all
round, plus the caption/scale-bar strip), so it sits inside the plate with ~3 mm
of slack across and ~7 mm along. 140 ChArUco corners, 82 markers.

Rule of thumb: the board's long side should be **half to two thirds of the image
width** at the working distance. For a wider field of view, scale `--square-mm`
up and keep the same square count. If the substrate changes, re-fit the page:
it is `squares_x * square_mm + 2 * margin_mm` wide, and 22 mm taller than the
same expression in y -- `make_board.py` reserves that strip for the caption.

Then:

1. print the PDF at **100% / actual size** (never "fit to page");
2. glue it to **glass or aluminium** — a sheet of paper that curls by half a
   millimetre is the largest error source in this whole procedure;
3. **measure a square with callipers** and, if the print scaled, write the
   measured number into `board.json`. Everything downstream is scaled by it.

## 3. Capture

### The guided way (recommended)

One command does the whole thing -- pre-flight, capture, solve, verdict:

```powershell
cd "sensor-data-collection\Python files"
.\.venv\Scripts\Activate.ps1
python -m pip install opencv-python==4.9.0.80        # once; keeps numpy 1.26 for PySpin

python ..\..\calibration\capture_gui.py `
    --board ..\..\calibration\board\board.json `
    --output C:\tactile_calib\guided
```

There is no second environment to activate and no path to retype: the solve
runs itself as subprocesses under `<repo>/.venv-calib`. The standalone
`focus_check.py`, `check_stability.py` and `tune_exposure.py --auto` still exist
for working on one thing at a time, but the GUI now covers all three.

The window sizes itself to the display it opens on: point sizes and the preview
resolution are derived from the screen height rather than hard-coded, because the
same numbers that fit a laptop are unreadable on a 4K panel, and this is read at
arm's length with both hands on the board. Each camera is a card whose coloured
title bar says what that camera needs right now -- `IN VIEW`, `MOVE RIGHT UP`,
`TURN TOWARDS ME`, `LENS OUT OF FOCUS`, or a grey `not needed this step` so you are
never trying to satisfy four cards at once. Underneath sits one line of advice for
the fault actually in hand, and a progress bar for the protocol as a whole.

**Pre-flight gates the session.** Nothing records until every camera passes the
four checks that have each cost a whole session here, none of which announce
themselves during capture: lens focus, board inside the depth of field, exposure
neither dark nor clipped, and no camera movement since the window opened. Panels
are green or red, `START CALIBRATION` is disabled until they are all green, and
`start anyway` is there when you know better. Drift keeps being measured through
the capture, so by the end it covers the whole half hour.


When you press **finish and save** it writes the session, then grades it without
being asked: how far each camera drifted during the run, followed by the full
solve (`detect_corners` -> `calibrate` -> `validate`) run under the calibration
interpreter as subprocesses, and a one-line verdict -- GOOD, MARGINAL, NOT
USABLE, or UNUSABLE if a camera moved. Drift outranks every other row: no
calibration of a rig that changed halfway exists, so the pixel numbers below it
are beside the point. `--no-solve` skips the grading; `--calib-python` points at
the interpreter if it is not at `<repo>/.venv-calib`.

`capture_gui.py` walks through the whole protocol: it names the step, shows all
four live views, and records a frame set **only** when the board is genuinely
still and in the pose that step needs. A finished run is a usable run, instead
of a two-minute gamble you can only grade afterwards.

It deliberately does not use the Arduino trigger. Hardware sync matters when the
target moves between cameras' exposures; this protocol holds the board still and
verifies stationarity in every contributing view before saving, which is the
stricter condition. The output is an ordinary session directory, ~1 GB instead
of 27 GB because only good poses reach disk, and `session.json` records the
world-segment range to pass to `--world-frames`.

The steps it enforces, in order:

| Step | What it wants |
|---|---|
| world | Board flat and still on the tactile array. Same pose recorded 12x for averaging. |
| face 0..N | Board held so it FACES each camera in turn (`facing > 0.55`), 8 distinct poses each. |
| pairs | Board shared between adjacent cameras, 6 poses each. |

**The whole board must be inside the frame, not merely detected.** The strict
multi-view solve uses only frames in which a camera measured *every* corner, so a
target held near the edge of the field of view yields none of them -- the session
finishes looking healthy and the co-visibility matrix comes back empty two steps
later. Each panel now reads `入画 n/140` with the direction to move, and the
pair/group steps refuse to record until the watched cameras see all of it. The
world and face steps only report it: with the board flat on the array three of the
four views cannot contain all of it however it is placed, so gating there would
deadlock the run.

**Check focus before every session.** `focus_check.py` reports the 10-90% rise
width of a black/white edge: under 2.2 px is sharp, over 3.2 px means the corner
refiner is fitting a ramp rather than an edge. Its `scene` column reads the
static rig at its true working distance (is the LENS focused?) and `board` reads
the target (is it inside the depth of field?). This is not a nicety -- one lost
session was three cameras at ~2 px and one at 6.8 px; the blurred one detected
too little of the board to share a single frame with any neighbour, which left
the camera graph disconnected and the extrinsics unsolvable at any threshold.

**"Facing" is the thing to watch.** Intrinsics are only determined when a camera
sees the board close to square-on, and eyeballing that from behind the rig does
not work -- a board that looks "tilted a bit" is still 70 degrees edge-on to a
low-mounted camera. `pose_helper.py` is the same live readout without recording,
useful for a minute of practice first.

### The manual way

Call the capture script directly (never `run_multimodal_workflow.py`: it does not
forward `--exposure-us`, and it deletes the BMPs unless you remember `--keep-bmp`).

```powershell
python ..\..\calibration\check_trigger.py --port COM3   # proves frames AND brightness
python capture_3blackfly_sensor_force.py `
    --port COM3 --cameras 4 --no-force-export-monitor `
    --output C:\tactile_calib\session
```

* Leave the images as raw Bayer -- do *not* pass `--demosaic-on-save`. BayerRG8
  is a third the size of BGR8 and `detect_corners.py` demosaics correctly.
* Set exposure with `tune_exposure.py --auto`, not with `--exposure-us`: the
  cameras need different gains when the rig is lit unevenly, and the capture
  script would force one value on all four.
* Always run `check_trigger.py` first. A camera that cannot service the trigger
  records **zero images** while the capture reports success.

### Camera-state tools

| Tool | Purpose |
|---|---|
| `tune_exposure.py --auto` | brings every camera to the same brightness, each with its own exposure/gain |
| `tune_exposure.py --prepare` | clears the free-run frame-rate cap, sets `TriggerOverlap=ReadOut`, checks the exposure fits the trigger period |
| `check_trigger.py` | end-to-end: real trigger, real handshake, counts frames and reports brightness |
| `inspect_calib.py` | see the re-projection error and epipolar lines on the actual photographs (calibration venv) |
| `check_stability.py` | watches the static scene for camera drift; run it before trusting any calibration |
| `focus_check.py` | live edge-sharpness per camera; separates a misfocused lens from a board held outside the depth of field |
| `pose_helper.py` | live "facing" score per camera while you rehearse holding the board |
| `reset_camera_auto_exposure.py` | puts the rig back on auto exposure when calibration is done |

These matter because Spinnaker settings live in the camera's volatile memory:
they survive program exit, SpinView, and every later capture, until the camera
loses power. A tool that changes one and does not restore it silently breaks the
next run -- which is why `tune_exposure.py` always finishes with
`prepare_for_capture`.

## 4. Solve

```powershell
python calibration\detect_corners.py <calib_dir> --board calibration\board\board.json ^
    --out calib\detections.npz --stride 2 --debug-dir calib\debug

python calibration\calibrate.py calib\detections.npz --out calib\calibration.json ^
    --world-frames 0:240 --world-flip-z

python calibration\validate.py calib\calibration.json calib\detections.npz --plots calib\report
```

`detect_corners.py` prints a co-visibility matrix: the number of frames in which
each **pair** of cameras sees the complete board. Thin entries there are the
single best predictor of a weak extrinsic, and are cheap to fix by re-shooting
before you solve. If that matrix stays thin no matter how you shoot — views too
oblique for the whole board to fit — add `--common-corners 24` to `calibrate.py`
(see the notes below).

`--world-frames A:B` is the `capture_sequence_index` range of the world segment.
`--world-flip-z` matters: a ChArUco board's own +z axis points *away* from the
printed face, so a board lying face-up under the cameras yields camera positions
at negative z. The flag rotates the world frame 180° about x so that **+z is
height above the tactile array**, which is what the shape ground truth wants.
Add `--world-z-offset` to subtract the thickness of the substrate the board is
glued to.

## 5. Accept or re-shoot

`validate.py` works on frames `calibrate.py` deliberately withheld (contiguous
blocks — at 24 fps a random split would put near-duplicate frames on both sides
and flatter the result). It triangulates the ChArUco corners in 3D and compares
them with the printed board:

| Check | Target | Why it matters |
|---|---|---|
| per-camera intrinsics RMS | ≤ 0.30 px | optics / focus / blur |
| multi-view solve RMS | ≤ 0.50 px | model fits the data it was fitted to |
| held-out re-projection RMS | ≤ 0.50 px | it also fits data it has not seen |
| board rigid-fit residual RMS | ≤ 0.20 mm | 3D shape error |
| **inter-corner distance error P95** | **≤ 0.30 mm** | **metric scale — the honest number** |
| worst-pair epipolar RMS | ≤ 1.0 px | pairwise geometry |
| median triangulation parallax | ≥ 15° | whether depth is resolvable at all |

The distance-error row is the one to trust. Low pixel errors with a high
distance error means the printed square size in `board.json` is wrong or the
board is not flat. A low parallax means the cameras sit too close together in
angle for depth to be constrained — spread them 30–60° apart.

`validate.py` exits non-zero if any row FAILs, so it can gate a capture session
in a script.

### Seeing it rather than reading it

```powershell
python calibration\inspect_calib.py calib\calibration.json calib\detections.npz `
    --session <capture_dir>
```

`validate.py` says whether the calibration passes. This says *where* it fails,
which is the question you have while standing at the rig. "7.9 px" is impossible
to picture; two marks 8 px apart on a photograph is not.

* **re-projection**: every corner two cameras saw is triangulated and projected
  back into all four views. Green is what was measured, red is what the
  calibration predicts, and the line between them is the error at `--magnify`
  times life size. A correct calibration puts the cross on the dot everywhere.
  Error vectors that all lean the same way are a systematic pose or intrinsics
  error; a random hedgehog is detection noise.
* **epipolar lines**: click anywhere in any view and the corresponding line is
  drawn in the other three. It depends only on the calibration -- no 3D
  reconstruction, no board -- so clicking a screw head or the corner of the
  sensor is a genuine independent check rather than a restatement of the fit.
  The line is re-distorted before drawing; at k1 = -0.19 a straight line would be
  wrong by tens of pixels at the frame edge, the same size as the error being
  judged.
* **3D panel**: camera positions, optical axes, and the triangulated board in the
  tactile array's frame.

Keys: left/right change frame, `a` jumps to the worst frame, `m` cycles the
magnification, `r` clears the click. It reads a finished session, so checking a
calibration costs seconds and needs no new capture.

## 6. Output

`calibration.json` is keyed by **camera serial**, never by enumeration index,
because index order can change between runs. For each camera it stores `K`,
`dist`, `R_world_to_cam`, `t_world_to_cam` (mm), the 3×4 projection matrix `P`,
and the camera position and optical axis in world coordinates. The pose
convention is stated in the file itself:

```
X_cam = R_world_to_cam @ X_world + t_world_to_cam
```

Re-calibrate whenever a camera is bumped, refocused, or has its ROI/binning
changed. A quick regression check between sessions is one board shot plus
`validate.py`.

---

## Things that will bite you (and are already handled here)

**The saved BMPs are raw Bayer.** `demosaic_on_save` is off by default in the
capture script, so a BMP read as grayscale still carries the RGGB mosaic, which
wrecks sub-pixel corner localisation. `session_io.py` demosaics using the format
in `session.json`. Note that GenICam and OpenCV name Bayer patterns one pixel
apart — Spinnaker's `BayerRG8` is OpenCV's `BayerBG`. The self-test asserts
this mapping, because getting it wrong swaps red and blue without any error.

**`calibrateMultiview` wants object points in metres.** It normalises them by
dividing by the *squared* maximum pairwise distance (`getScaleOfObjPoints`
returns `NORM_L2SQR` but is used as a length). Points given in millimetres shrink
by ~100× and trip the internal collinearity guard with `Pattern points are
collinear!`. `calibrate.py` converts to metres for the call and scales the
returned translations back to millimetres.

**Partial observation is only half-supported upstream, and it does not fail
quietly.** OpenCV asks you to fill unobserved corners with `(-1, -1)`. The final
LM stage is robustified and ignores them, but `calibrateCamera`, `solvePnP` and
`registerCameras` in the initialisation stages are plain least squares and
receive the same arrays — and a `(-1, -1)` sits a thousand pixels from the truth.
Measured on this repository's own synthetic rig: placeholder-padded partial views
diverge to an RMS of **2×10¹⁴ px**, while the same data restricted to
placeholder-free cells solves to **0.17 px**.

So `calibrate.py` computes intrinsics itself from the genuinely detected corners,
and only ever passes the solver corners that were really observed. By default
that means cells where the camera saw the complete board. If a rig's views never
share a whole board, `--common-corners 24` solves on the 24 corners the views do
share instead — a smaller pattern, but every value in it is a real measurement.
On the self-test rig the two modes agree to within 0.02° and 0.04 mm.

**`detectionMask` is per view, not per point -- so most of the data never reaches
the solver.** `calibrateMultiview` takes a (camera x frame) mask, so a cell counts
only when that camera measured the *whole* pattern in that frame. A ChArUco target
seen at an angle almost never is complete: on one session here 27252 corners were
detected, the strict solve used about 11000 of them, and `--common-corners 40`
used 6520 -- 24%. Nothing in the geometry requires that; a corner seen by one
camera in one frame constrains that camera's pose and that frame's board pose
perfectly well. `bundle.py` therefore adds a final sparse Levenberg-Marquardt over
every detected corner, seeded from the multi-view result (6 parameters per camera,
6 per frame, 2 residuals per corner). Measured on that session it moved the
held-out re-projection RMS 4.54 -> 3.21 px, the board rigid-fit 0.196 -> 0.150 mm,
the worst-pair epipolar 9.10 -> 6.42 px, and the metric distance error P95
0.470 -> 0.349 mm. It will not repair a rig that moved mid-session -- one rigid set
of extrinsics is simply the wrong model for that -- so it is a refinement, not a
rescue. `--no-refine` turns it off.

**Intrinsics are not refined by the multi-view stage.** `calibrateMultiview`
optimises only extrinsics and per-frame board poses — its parameter vector is
`(N-1)×6 + frames×6`. Supplying good intrinsics up front therefore costs nothing
and is strictly better than letting it fit them from placeholder-contaminated
arrays.

**Focal length and range are coupled.** A 0.25% focal-length error shows up as a
0.25% error in camera distance. This is why the acceptance criterion is a
millimetre distance error on held-out frames rather than a pixel residual.

## Self-test

```powershell
python calibration\tests\run_selftest.py --work-dir <scratch>\calib_selftest
```

Renders a synthetic four-camera rig with known intrinsics, extrinsics, lens
distortion and Bayer mosaic, runs the real scripts on it, and checks the
recovered parameters against the values used to render. Currently 46/46 checks:
focal length within 0.25%, orientation within 0.14°, camera position within
0.9 mm at 420 mm range, and a held-out metric error of 0.09 mm. It covers both
corner-selection modes, the world-frame flip, and the GenICam/OpenCV Bayer
naming. Run it after changing anything here. `--cameras 3` exercises a
three-camera rig (36/36 checks).

## What actually goes wrong on this rig

Recorded because each of these cost a capture session and none of them announce
themselves:

* **A camera that cannot service the trigger records nothing.** A free-run frame
  rate cap left at 5 fps, or `TriggerOverlap=Off` with an exposure that does not
  fit the trigger period, makes every pulse land while the sensor is busy. The
  capture script runs to completion and writes a session with zero images.
  `check_trigger.py` catches it in ten seconds.
* **Exposure is not recorded when the capture script does not set it.** A session
  shot six times too dark looks identical in the logs. `detect_corners.py` now
  reports p99 per camera and refuses to stay quiet about it.
* **A board that is never square-on cannot calibrate.** Holding the board flat
  in front of low-mounted cameras gives 70-degree incidence in every frame;
  focal length and principal point then come out with fx and fy differing by
  percent. `detect_corners.py` reports a "pose diversity" column, and
  `capture_gui.py` refuses to record until the pose is right.
* **A camera that drifts mid-session cannot be calibrated at all, and nothing
  downstream says so.** Solved on any single capture step, one session gave
  0.5-1.2 px on every camera pair; solved on all nine steps together, 7.9 px.
  Every step fitted itself and none fitted the others, because the rig it
  described stopped existing halfway through. Re-fitting intrinsics, freeing them
  in stereo, richer distortion models and outlier rejection all changed nothing --
  the residual is irreducible by construction. The evidence was in the pixels all
  along: over half an hour the static background moved 21.9 px in one camera and
  6.6 px in another, while the other two moved 0.0 and 1.4 px.
  `check_stability.py` measures this in twenty minutes with no board and no solve.
  Suspect it whenever per-step and whole-session residuals disagree by an order of
  magnitude.
* **A warped target sets a floor on accuracy.** 0.6 mm of dome across the board
  produced ~2.6 px of irreducible reprojection error here -- adding distortion
  terms changed nothing, while restricting to the central third halved it. Mount
  the print on glass.
* **`cornerSubPix` and detection succeed on bad data.** Every one of the failures
  above still produced plausible-looking detections. The pixel residual is what
  tells you, and the metric error on held-out frames is what settles it.
