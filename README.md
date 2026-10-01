# 3D_Reconstruction

**The real-world capture rig for camera-free 3D shape + force reconstruction of
soft deformable objects.**

A flexible 16×16 piezoresistive array is attached to a soft object. Four
externally-triggered cameras and a Mark-10 force gauge record what is happening
to it. After training, the goal is to reconstruct the object's 3D deformation
and the contact force **from the tactile signal alone** — the cameras and the
gauge are training-time ground truth, and are not present at inference.

That is the whole reason this repository is fussy about calibration: **the
cameras *are* the shape ground truth**, so calibration error becomes label error
that nothing downstream can detect.

---

## Layout

| Folder | What it is |
|---|---|
| [`sensor-data-collection/`](./sensor-data-collection) | The rig: Arduino firmware, synchronized capture, video export, force alignment, replay and bench tools |
| [`calibration/`](./calibration) | Multi-camera calibration — intrinsics and extrinsics in the tactile array's own frame, in millimetres |
| [`calibration/calibration.json`](./calibration/calibration.json) | **The calibration currently in use.** Keyed by camera serial |

Each folder has its own README with the full detail. This one is the map and the
current state.

**Captured data does not live in this repository.** Sessions go to a path you
pass with `--output` (the rig writes ~288 MiB/s, and a single run is tens of
gigabytes before video export). `.gitignore` keeps images, video and `.npz` out
of version control; `calibration.json` and `board.json` are small and are
committed on purpose.

---

## The pipeline, end to end

Two Python environments, because they cannot share one. Everything that touches
a camera needs PySpin, which pins `numpy==1.26`; `cv::calibrateMultiview` exists
only in OpenCV 5, which needs `numpy>=2`.

| Environment | Where | Used by |
|---|---|---|
| **capture** | `sensor-data-collection/Python files/.venv` | everything that talks to a camera or the Arduino |
| **calibration** | `.venv-calib` (repo root) | the offline solve: `detect_corners`, `calibrate`, `validate`, `inspect_calib` |

### 1. Calibrate (once per rig configuration)

```powershell
cd "sensor-data-collection\Python files"
.\.venv\Scripts\Activate.ps1

python ..\..\calibration\check_stability.py --minutes 20     # does the rig hold still?
python ..\..\calibration\tune_exposure.py --auto --gain-db 0 --max-exposure-us 40000
python ..\..\calibration\capture_gui.py `
    --board ..\..\calibration\board\board.json `
    --output C:\tactile_calib\guidedN --in-frame-frac 1.0
```

`capture_gui.py` walks the protocol, records only poses that qualify, and then
solves and grades itself — GOOD / MARGINAL / NOT USABLE. Copy the resulting
`calibration.json` over [`calibration/calibration.json`](./calibration/calibration.json)
and commit it.

`--in-frame-frac 1.0` matters: the default (0.85) lets through frames where a
camera sees 119 of the 140 corners, but the strict multi-view solve only uses
frames where it saw **all** of them. At 0.85 a session can finish looking healthy
and leave the solver with nothing.

See [`calibration/README.md`](./calibration/README.md) for the protocol, the
acceptance criteria, and a long list of failures that each cost a session.

### 2. Capture

```powershell
python ..\..\calibration\tune_exposure.py --auto    # under the measurement lighting
python ..\..\calibration\tune_exposure.py --prepare # does the exposure fit the trigger period?

python capture_3blackfly_sensor_force.py --port COM3 --cameras 4 --output D:\tactile_data\run01
```

**Start the capture first, then run the force test**, so the whole force curve
falls inside the recording. Force samples with no images are not usable as
ground truth.

Do not pass `--exposure-us` or `--gain-db` — they overwrite what
`tune_exposure.py` stored in the cameras. Lighting may change freely between
calibration and capture; **the lenses may not.** Focus and aperture are part of
the calibration, so adjust brightness with exposure time and gain only.

### 3. Export video

```powershell
python export_multimodal_mp4.py D:\tactile_data\run01\capture_YYYYmmdd_HHMMSS --crf 12
```

Encodes the raw Bayer BMPs to one MP4 per camera and writes
`multimodal_video_alignment.csv`, the per-frame join table. Add
`--delete-images` to drop the BMPs — it only deletes after every MP4 has been
decoded and verified against the expected frame count. CRF 12 rather than the
default 18 because this is measurement data; note the output is `yuv420p`, so
chroma is half resolution either way.

### 4. Attach the force curve

```powershell
python align_force_curve_only.py D:\tactile_data\run01\capture_YYYYmmdd_HHMMSS "force.csv"
```

IntelliMESUR runs on a separate tablet, so no file timestamp is meaningful. This
recovers the offset from the data by correlating the force curve against tactile
features across the whole recording, and reports a confidence and a runner-up
margin. **Check that confidence** — `ambiguous` means the result should not be
trusted.

> **The tactile signal is inversely related to load** (`V_sensor = Vdrive -
> V_adc`), so the true correlation is *negative*. An earlier `align_force.py`
> rejected negative correlations and therefore never considered the correct peak;
> it has been removed. Anything new that aligns these two streams must use the
> absolute correlation.

### 5. Verify

```powershell
python replay_capture_2d.py <session>            # 2D top-down — the check view
python replay_capture_multimodal.py <session>    # 3D bar view
```

2D is the default for checking a capture: no occlusion, and it registers
directly against the camera previews. The 3D view's z-axis is **sensor voltage,
not displacement** — it is a visualisation, not a physical shape.

---

## Current state

**Calibration** — [`calibration/calibration.json`](./calibration/calibration.json),
solved 2026-09-30 from `guided8`, rig stable throughout (0.00 px drift on all
four cameras).

```
inter-corner distance error P95   0.173 mm   (target ≤ 0.30)   PASS
board rigid-fit residual RMS      0.100 mm   (target ≤ 0.20)   PASS
worst per-camera intrinsics RMS   0.924 px   (target ≤ 0.30)   FAIL
worst pair epipolar RMS           1.325 px   (target ≤ 1.00)   over
```

Usable, with **0.17 mm as the error floor** on any reconstructed shape. The
metric rows are the acceptance criterion — focal length and range are coupled, so
a millimetre error on held-out frames is the honest measure. The pixel rows are
over target because the extrinsics came from a sub-pattern solve (80 shared
corners, not the full 140) and the board was never held square-on enough to pin
down the intrinsics. Re-calibrating with `--in-frame-frac 1.0` and better facing
is the fix; post-processing is not.

**Known problems, in the order they will bite:**

- **The tactile array is degrading, and quickly.** Between two sessions sixteen
  hours apart on 2026-09-30 the signal span fell from 0.87 V to 0.06 V, cells
  that interpolation could not repair went from 26 to 16996, and row 13 went from
  failing in 9% of frames to **100%**. Rows 3, 4, 14, 16 and column 1 are now
  failing too. The pattern is whole rows and columns, which is a
  multiplexer/wiring signature rather than wear. **Check this with
  `test/live_sensor_2d.py` before trusting a capture** — a session recorded
  through a dead array looks completely normal in the logs.
- **The force gauge is out of calibration** (due 2026-07-22). Fine for
  correlation and relative work; treat the newton values with caution as absolute
  ground truth.
- **Write bandwidth is a real constraint.** Four cameras at 24 fps is ~288 MiB/s.
  A disk that cannot sustain it drops frames per camera, independently, and the
  session's `sequential_pairing_clean` goes false. Video export recovers a usable
  set by keeping only the frames every camera and the sensor share, but the
  frames are then not contiguous in the capture timeline.
- **`check_trigger.py` returns zero on a working rig.** It is still worth
  running, but a zero is a reason to try the real capture, not proof of a broken
  trigger. See the end of [`calibration/README.md`](./calibration/README.md).

---

## Bench tools

| Tool | Use |
|---|---|
| `test/live_sensor_2d.py` | live 2D pressure image — **the fastest way to see whether the array is alive** |
| `test/live_sensor_3d.py` | live 3D bars; judge the deformation shape before committing to a capture |
| `calibration/inspect_calib.py` | re-projection, epipolar lines, and the rig in 3D — where a calibration fails, not just whether |
| `calibration/check_stability.py` | 20 minutes, no board, no solve: does the rig hold still? |
| `calibration/focus_check.py` | edge sharpness per camera; separates a misfocused lens from a board outside the depth of field |
| `calibration/pose_helper.py` | live "facing" score while rehearsing how to hold the board |
| `calibration/tests/run_selftest.py` | synthetic rig end to end; run after changing anything in `calibration/` |

---

## History

This repository previously also vendored `DeformFieldBench/`, the simulation and
material-parameter line of the project (MPM simulation, three-view video →
material parameters). It was removed on 2026-09-30 to keep this repository to the
physical rig; it remains in the git history and in its own upstream repository.

`run_multimodal_workflow.py` and `repair_sensor_session.py` were removed at the
same time — the workflow wrapper did not forward exposure arguments and deleted
source images by default, and the repair script was only reachable through it.
Call the capture, export and alignment steps directly, as above.
