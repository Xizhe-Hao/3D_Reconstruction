# 3D_Reconstruction

**Camera-free 3D shape + force reconstruction for soft deformable objects — unified workspace.**

This repository holds the whole real-world pipeline of the project in one place: capture synchronized tactile + multi-camera + force data, calibrate the cameras in the frame of the tactile array, and reconstruct the object's 3D deformation from the cameras. The reconstructed shape and the measured force are the training targets for a model that later works from the tactile signal alone.

| Folder | Origin | Role in the pipeline |
|---|---|---|
| [`sensor-data-collection/`](./sensor-data-collection) | [dfk0411/sensor-data-collection](https://github.com/dfk0411/sensor-data-collection) | **1. Capture**: firmware + synchronized acquisition (16×16 tactile array + 4 cameras + Mark-10 force gauge) |
| [`calibration/`](./calibration) | this repo | **2. Calibrate**: intrinsics + extrinsics of the 4 cameras, in millimetres, in the tactile-array frame → `calibration.json` |
| [`reconstruction/`](./reconstruction) | [3365538768/ND_mvtracker_reconstruction](https://github.com/3365538768/ND_mvtracker_reconstruction) @ `4355999` | **3. Reconstruct**: metric multi-view depth (DUSt3R) + 4D point tracking (MVTracker) → per-frame 3D trajectories and dense point clouds |

The simulation benchmark that used to live here (`DeformFieldBench/`) has been removed because the project no longer uses simulation. It is still in git history (commit `9926942`) and upstream at [3365538768/DeformFieldBench](https://github.com/3365538768/DeformFieldBench).

---

## 1. The Big Picture

```
  sensor-data-collection            calibration                    reconstruction
 ┌─────────────────────────┐   ┌─────────────────────────┐   ┌──────────────────────────────┐
 │ 16x16 tactile ─► input  │   │ ChArUco sessions        │   │ DUSt3R: metric depth per     │
 │ 4x FLIR (D9 trigger)    │   │  ─► calibration.json    │   │   timestamp, poses fixed to  │
 │ Mark-10 ─► force GT     │   │  (K, dist, R|t, mm,     │   │   calibration.json           │
 │                         │   │   tactile-array frame)  │   │ MVTracker: 3D point tracks   │
 │ session/                │   └────────────┬────────────┘   │   over time                  │
 │  camera_i_<serial>*.mp4 │                │                │ ─► tracks_4d.npz/.csv/.rrd   │
 │  multimodal_video_*.json│────────────────┴──────────────► │ ─► ply_dense/ per frame      │
 │  multimodal_video_*.csv │   copy calibration.json into    └──────────────┬───────────────┘
 └─────────────────────────┘   the session folder                           │
                                                                            ▼
                              Train: tactile signal ──► shape (+ force)
                              Deploy: CAMERA-FREE reconstruction
```

- The **tactile array is the input**, and the only sensor that remains at deployment. "Camera-free" means *no cameras at inference*; cameras supervise training only.
- The **cameras are the shape answer key**. Consumer RGB-D was rejected because its depth precision cannot resolve millimetre-level deformation on a textureless surface. The rig grew from three views to four to improve triangulation coverage.
- The **Mark-10 is the force answer key**, the stream this project adds on top of the published shape-only work (flexible sensor array + cage-based 3D Gaussian modelling, arXiv:2603.19543).

Because `calibration.json` puts the world origin on the tactile array and the reconstruction keeps that world frame (converted to metres), reconstructed points line up directly with the 16×16 sensor cells underneath them.

**This also sets the accuracy budget.** The cameras are the ground truth, so calibration error becomes label error that nothing downstream can detect. See *Current state* below for what the calibration in this repository is actually worth.

---

## 2. `sensor-data-collection/`: Capture

Records **synchronized** data from three sources:

| Stream | Hardware | Role |
|---|---|---|
| Tactile frames (16×16, 8-bit) | Velostat piezoresistive array, scanned by **Arduino Nano + dual CD74HC4067 multiplexers** | **Model input** (the only stream that remains at deployment) |
| Video (4 views) | 4× Teledyne FLIR **Blackfly S USB3** (PySpin / Spinnaker SDK) | **Shape ground truth** (training only) |
| Force curve | **Mark-10** force gauge + IntelliMESUR tablet (CSV via email) | **Force ground truth** (training only) |

### 2.1 Hardware (`Arduino files/`)

- **`MatrixArrayDual4067Binary.ino`** is the one firmware to flash (Arduino Nano, ATmega328P).
  - Scans the 16×16 array through two CD74HC4067 muxes: row address pins **D4–D7** (inhibit **D8**), column address pins **A1–A4** (inhibit **A5**).
  - Streams binary frames over serial at **500000 baud**: magic `M16B`, version, dimensions, 8-bit ADC payload (256 bytes, row-major), frame index, device timestamp, 16-bit checksum.
  - **Camera sync**: emits a 100 µs rising-edge pulse on **D9**, wired in parallel to the hardware-trigger input of *every* camera. This pulse is what makes the videos and the tactile stream line up. Cameras must be configured for external hardware trigger.
  - The firmware stops pulsing D9 if the host goes quiet for 2 s (`HEARTBEAT_TIMEOUT_MS`) or stops draining the serial stream — `Serial.flush()` sits immediately before the pulse. Any tool that handshakes but does not do both will see a trigger that looks dead.

### 2.2 Acquisition software (`Python files/`)

| Script | What it does |
|---|---|
| `capture_3blackfly_sensor_force.py` | **Stage A.** Simultaneous camera + tactile capture (`--port COMx`, `--cameras 4`, `--output <dir>`) |
| `export_multimodal_mp4.py` | Converts captured frames to synchronized MP4s and writes `multimodal_video_export.json` + `multimodal_video_alignment.csv`, the inputs the reconstruction reads |
| `align_force_curve_only.py` | **Stage B.** Aligns the emailed Mark-10 CSV to the session using *curve-shape matching* between the force curve and the tactile signal (timestamps are ignored). Prints `score` / `confidence` / `peak_margin` |
| `replay_capture_2d.py` | **Verification view.** Replays a session with a 2D top-down tactile image beside the camera previews and force curve |
| `replay_capture_multimodal.py` | Same replay with a 3D bar panel (z-axis is voltage, not displacement) |

A `run_multimodal_workflow.py` wrapper used to front these. It was removed: it did not forward `--exposure-us`, and it deleted the source BMPs unless `--keep-bmp` was remembered. Its `repair_sensor_session.py` helper went with it. Call the three stages directly.

**The tactile signal is inversely related to load** (`V_sensor = Vdrive - V_adc`), so force and tactile features correlate *negatively*. Anything that aligns or regresses these two streams must use the absolute correlation; a tool that rejected negative correlations silently produced an alignment 50 s away from the truth before it was removed.

### 2.3 Quick start (Windows, tested config)

1. Python 3.10 (64-bit), Spinnaker SDK 4.3.0.190 + matching Teledyne PySpin wheel, NumPy 1.26.4.
2. Flash `Arduino files/MatrixArrayDual4067Binary.ino` (board: Arduino Nano; if upload fails, try the *Old Bootloader* processor option). Note the COM port.
3. Verify D9 reaches all four camera trigger inputs; confirm each camera streams in SpinView, then **close SpinView**.
4. ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r "sensor-data-collection\Python files\requirements.txt"
   python -m pip install "path\to\teledyne\pyspin\wheel"
   ```
5. Set exposure under the lighting you will actually use:
   ```powershell
   python ..\..\calibration\tune_exposure.py --auto
   python ..\..\calibration\tune_exposure.py --prepare   # does the exposure fit the trigger period?
   ```
6. **Stage A** (capture): `python capture_3blackfly_sensor_force.py --port COM3 --cameras 4 --output D:\tactile_data\run01`
   → **start the capture first**, then run the force test inside that window → `Ctrl+C` → export with `python export_multimodal_mp4.py <session> --crf 12`.
7. **Stage B** (force import): `python align_force_curve_only.py <session folder> <force csv>` (prefer `confidence=high`; `ambiguous` means do not trust it).
8. Verify: `python replay_capture_2d.py <session folder>`.

Two rules that are easy to break and expensive to discover:

- **Never pass `--exposure-us` or `--gain-db` to the capture** — they overwrite what `tune_exposure.py` stored in the cameras.
- **Lighting may change freely between calibration and capture; the lenses may not.** Focus and aperture are part of the calibration. Adjust brightness with exposure time, gain, or a lamp — never the lens.

Full details: [`sensor-data-collection/README.md`](./sensor-data-collection/README.md).

---

## 3. `calibration/`: Calibrate

Multi-camera ChArUco calibration built on OpenCV 5's `calibrateMultiview`, in its own venv (`.venv-calib`, because OpenCV 5 needs `numpy>=2` and PySpin needs 1.26). Output is `calibration.json`: per-camera `K`, `dist`, `R_world_to_cam`, `t_world_to_cam`, units millimetres, world frame on the tactile array, **keyed by camera serial rather than index**.

```
make_board.py -> capture_gui.py -> detect_corners.py -> calibrate.py -> validate.py / inspect_calib.py
```

The guided path does all of it and grades itself:

```powershell
python ..\..\calibration\check_stability.py --minutes 20     # does the rig hold still at all?
python ..\..\calibration\tune_exposure.py --auto --gain-db 0 --max-exposure-us 40000
python ..\..\calibration\capture_gui.py `
    --board ..\..\calibration\board\board.json `
    --output C:\tactile_calib\guidedN --in-frame-frac 1.0
```

- `--gain-db 0` because the board is stationary: long exposure costs nothing and 18 dB of gain measurably degrades sub-pixel corner localisation. The auto-tuner cannot lower gain on its own once it is high, so set it explicitly.
- `--in-frame-frac 1.0` because the default (0.85) records frames where a camera sees 119 of the 140 corners, while the strict multi-view solve only uses frames where it saw **all** of them. At 0.85 a session can finish looking healthy and leave the solver with nothing.

Recalibrate whenever a camera is moved or refocused; the reconstruction trusts these poses completely (DUSt3R's global alignment keeps them fixed). `check_stability.py` measures whether the rig holds still at all, and `session_report` prints per-camera drift after every guided session — a camera that moves mid-session cannot be calibrated at any residual.

Full details: [`calibration/README.md`](./calibration/README.md), which also lists the failures that each cost a capture session.

---

## 4. `reconstruction/`: Reconstruct

For each sampled timestamp, DUSt3R matches all pairs among the four synchronized views and globally aligns one metric scene with the calibrated intrinsics and poses held fixed. MVTracker then tracks 3D query points through time over those depths. Outputs, in metres, in the calibration world frame:

- `tracks_4d.npz` / `tracks_4d.csv`: sparse point trajectories `(track_id, frame, time, x, y, z, visible)`, keyed by the source frame index and `video_time_s`, so they join back to the tactile/force rows in `multimodal_video_alignment.csv`
- `tracks_4d.rrd`: Rerun playback (cameras, fused point cloud, tracks)
- `ply/`, `ply_dense/`: tracked points and fused dense RGB-D point cloud per timestamp

MVTracker gives point trajectories, not a watertight deforming mesh.

### 4.1 From a capture session to a reconstruction

The adapter (`reconstruction/scripts/test_session_adapter.py`) reads a session folder exactly as `export_multimodal_mp4.py` leaves it. The one manual step is to put the calibration next to the videos:

```
<session>/
  camera_0_<serial>_*.mp4 … camera_3_<serial>_*.mp4   # from export_multimodal_mp4.py
  multimodal_video_export.json                        # from export_multimodal_mp4.py
  multimodal_video_alignment.csv                      # from export_multimodal_mp4.py
  calibration.json                                    # copy from calibration/
```

The adapter checks that there are **exactly 4 cameras**, that each calibrated serial appears in the matching video file name, that `calibration.json` is in millimetres with the `X_cam = R_world_to_cam @ X_world + t_world_to_cam` convention, and that the video resolution matches the calibration.

### 4.2 Running it (Linux + CUDA GPU)

The reconstruction environment needs Linux, Conda and a CUDA 12.1-compatible driver (the setup scripts use `conda`, `wget`, `md5sum`), so it runs on the lab GPU server rather than the Windows capture PC. Run everything from `reconstruction/`:

```bash
git submodule update --init --recursive      # MVTracker + DUSt3R at pinned commits
cd reconstruction
bash scripts/setup_mvtracker.sh              # conda env "mvtracker" + MVTracker checkpoint
bash scripts/setup_duster.sh                 # DUSt3R + 2.1 GB checkpoint (MD5-checked)

# put the session (with calibration.json) at reconstruction/data/<name>/, then:
SESSION_DIR=data/<name> OUTPUT_DIR=outputs/<name> END_FRAME=<frame_count-1> \
  bash scripts/run_duster_tracking_full.sh

# browse the result
conda run --no-capture-output -n mvtracker python scripts/mvtracker_visualizer_gradio.py \
  --result-dir outputs/<name>/frames_0_<end>_target_96_duster
```

Start with a short smoke test (`--start 660 --end 666 --target-frames 7 --duster-ga-niter 50`) before a full 96-frame run; each timestamp is a full multi-view DUSt3R alignment. `reconstruction/data/` and `outputs/` are git-ignored.

Full details (Chinese): [`reconstruction/README.md`](./reconstruction/README.md), [`reconstruction/docs/`](./reconstruction/docs).

---

## 5. Current state

**Calibration** — [`calibration/calibration.json`](./calibration/calibration.json), solved 2026-09-30, rig stable throughout (0.00 px drift on all four cameras).

```
inter-corner distance error P95   0.173 mm   (target ≤ 0.30)   PASS
board rigid-fit residual RMS      0.100 mm   (target ≤ 0.20)   PASS
worst per-camera intrinsics RMS   0.924 px   (target ≤ 0.30)   FAIL
worst pair epipolar RMS           1.325 px   (target ≤ 1.00)   over
```

Usable, with **0.17 mm as the error floor** on any reconstructed shape. The metric rows are the acceptance criterion — focal length and range are coupled, so a millimetre error on held-out frames is the honest measure and a pixel residual is not. The pixel rows are over target for real reasons: the extrinsics came from a sub-pattern solve (80 shared corners, not the full 140) because two cameras rarely saw the whole board, and the intrinsics are under-determined because the board was never held square-on enough. Re-calibrating with `--in-frame-frac 1.0` and better facing is the fix; post-processing is not.

**Known problems, in the order they will bite:**

- **The tactile array is degrading, and quickly.** Between two sessions sixteen hours apart on 2026-09-30 the signal span fell from 0.87 V to 0.06 V, cells that interpolation could not repair went from 26 to 16996, and row 13 went from failing in 9 % of frames to **100 %**. Rows 3, 4, 14, 16 and column 1 are now failing too. Whole rows and columns is a multiplexer/wiring signature, not wear. **Check with `test/live_sensor_2d.py` before trusting a capture** — a session recorded through a dead array looks completely normal in the logs.
- **The force gauge is out of calibration** (due 2026-07-22). Fine for correlation and relative work; treat the newton values with caution as absolute ground truth.
- **Write bandwidth is a real constraint.** Four cameras at 24 fps is ~288 MiB/s. A disk that cannot sustain it drops frames per camera, independently, and the session's `sequential_pairing_clean` goes false. Video export then keeps only the frames every camera and the sensor share, so the exported set is clean but not contiguous in the capture timeline — use `capture_sequence_index` where the gaps matter.
- **`check_trigger.py` returns zero on a working rig.** It reported 0 complete and 0 incomplete frames on all four cameras, twice, immediately before a capture that recorded 179/179 frames with clean pairing. Treat a zero as a reason to try the real capture, not as proof of a broken trigger.

## 6. What Comes Next

1. Reproduce the four-camera + tactile capture and the shape-reconstruction result of the predecessor paper, now with the calibrated DUSt3R + MVTracker reconstruction as shape ground truth.
2. Add force: train *tactile → shape + contact force*, keeping the camera-free / zero-shot properties.
3. With measured force + reconstructed deformation, move toward real-world material/physical understanding and robot applications (force-aware soft grippers, contact-rich data collection for embodied AI).

## 7. Repository Hygiene

- Large data (capture sessions, reconstruction outputs, checkpoints, `.ply`/`.rrd`/`.mp4`) must **not** be committed; see `.gitignore`. Use lab storage. `calibration.json` and `board.json` are small and *are* committed on purpose.
- `sensor-data-collection/` and `reconstruction/` mirror upstream repos. `reconstruction/submodule/*` are third-party submodules pinned to specific commits: keep them clean, and put project code in `reconstruction/scripts/`.
- Keep new code (force pipeline, tactile→shape models) in **new top-level folders**.
