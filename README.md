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

### 2.2 Acquisition software (`Python files/`)

| Script | What it does |
|---|---|
| `run_multimodal_workflow.py` | **Main entry point.** Orchestrates the full Stage-A capture (`--port COMx`, optional `--camera-serials`, `--output`) |
| `capture_3blackfly_sensor_force.py` | Simultaneous camera + tactile capture (called by the workflow) |
| `repair_sensor_session.py` | Detects/repairs corrupted tactile frames in a session |
| `export_multimodal_mp4.py` | Converts captured frames to synchronized MP4s and writes `multimodal_video_export.json` + `multimodal_video_alignment.csv`, the inputs the reconstruction reads |
| `align_force_curve_only.py` | **Stage B.** Aligns the emailed Mark-10 CSV to the session using *curve-shape matching* between the force curve and the tactile signal (timestamps are ignored). Prints `score` / `confidence` / `peak_margin` |
| `replay_capture_2d.py` | **Verification view.** Replays a session with a 2D top-down tactile image beside the camera previews and force curve |
| `replay_capture_multimodal.py` | Same replay with a 3D bar panel (z-axis is voltage, not displacement) |

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
5. **Stage A** (capture): `python run_multimodal_workflow.py --port COM7` → baseline ≈1 s → run IntelliMESUR + perform the press/bend action → baseline ≈1 s → `Ctrl+C` → wait for MP4 verification.
6. **Stage B** (force import): `python align_force_curve_only.py <session folder> <force csv>` (prefer `confidence=high`).
7. Verify: `python replay_capture_2d.py <session folder>`.

Full details: [`sensor-data-collection/README.md`](./sensor-data-collection/README.md).

---

## 3. `calibration/`: Calibrate

Multi-camera ChArUco calibration built on OpenCV 5's `calibrateMultiview`, in its own venv (`.venv-calib`, because OpenCV 5 needs `numpy>=2` and PySpin needs 1.26). Output is `calibration.json`: per-camera `K`, `dist`, `R_world_to_cam`, `t_world_to_cam`, units millimetres, world frame on the tactile array.

```
make_board.py -> capture_gui.py -> detect_corners.py -> calibrate.py -> validate.py / inspect_calib.py
```

Recalibrate whenever a camera is moved or refocused; the reconstruction trusts these poses completely (DUSt3R's global alignment keeps them fixed).

Full details: [`calibration/README.md`](./calibration/README.md).

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

## 5. What Comes Next

1. Reproduce the four-camera + tactile capture and the shape-reconstruction result of the predecessor paper, now with the calibrated DUSt3R + MVTracker reconstruction as shape ground truth.
2. Add force: train *tactile → shape + contact force*, keeping the camera-free / zero-shot properties.
3. With measured force + reconstructed deformation, move toward real-world material/physical understanding and robot applications (force-aware soft grippers, contact-rich data collection for embodied AI).

## 6. Repository Hygiene

- Large data (capture sessions, reconstruction outputs, checkpoints, `.ply`/`.rrd`/`.mp4`) must **not** be committed; see `.gitignore`. Use lab storage.
- `sensor-data-collection/` and `reconstruction/` mirror upstream repos. `reconstruction/submodule/*` are third-party submodules pinned to specific commits: keep them clean, and put project code in `reconstruction/scripts/`.
- Keep new code (force pipeline, tactile→shape models) in **new top-level folders**.
