# 3D_Reconstruction

**Camera-free 3D shape + force reconstruction for soft deformable objects — unified workspace.**

This repository consolidates the two prior codebases of our group into one place, as the starting point for the next stage of the project: extending camera-free soft-body deformation reconstruction with **force sensing**.

> **Nothing in the two source folders has been modified.** All source code is copied verbatim from the original repositories; only this top-level README (and `.gitignore`) are new.

| Folder | Origin | Role in the project |
|---|---|---|
| [`sensor-data-collection/`](./sensor-data-collection) | [dfk0411/sensor-data-collection](https://github.com/dfk0411/sensor-data-collection) | **Real-world rig**: hardware firmware + synchronized multimodal data acquisition (tactile array + 3 cameras + force gauge) |
| [`DeformFieldBench/`](./DeformFieldBench) | [3365538768/DeformFieldBench](https://github.com/3365538768/DeformFieldBench) | **Simulation benchmark + models**: material-parameter estimation from deformation videos (training, evaluation, simulation) |

---

## 1. The Big Picture

The overall research line of the group:

```
                         REAL WORLD                                    SIMULATION
        ┌────────────────────────────────────────────┐   ┌─────────────────────────────────────┐
        │  sensor-data-collection  (this = data rig)  │   │  DeformFieldBench (benchmark+models) │
        │                                             │   │                                      │
        │  16x16 tactile array ──► INPUT signal       │   │  Taichi/Warp MPM simulation          │
        │  3x FLIR cameras     ──► SHAPE ground truth │   │  ──► 3-view RGB deformation videos   │
        │  Mark-10 gauge       ──► FORCE ground truth │   │  ──► stress / flow / force fields    │
        │                                             │   │  ──► GT material params (E, nu,      │
        │  (synchronized, aligned capture sessions)   │   │       rho, sigma_y)                  │
        └────────────────────┬────────────────────────┘   └──────────────────┬───────────────────┘
                             │                                                │
                             ▼                                                ▼
              Train: tactile signal ──► shape (+ force)          Train: video ──► material params
              Deploy: CAMERA-FREE reconstruction                 (vision + simulation, no touch)
```

Two complementary lines:

1. **Shape line (real world, tactile).** A flexible piezoresistive tactile array is attached to a soft object. Three externally-triggered cameras and a Mark-10 force gauge act as *training-time ground truth only*. The goal: after training, reconstruct the object's 3D deformation (and now: contact force) **from the tactile signal alone — no cameras at inference time** ("camera-free"). The published predecessor of this line is the zero-shot deformation reconstruction paper (flexible sensor array + cage-based 3D Gaussian modeling, arXiv:2603.19543).
2. **Physics line (simulation, vision).** DeformFieldBench asks a deeper question: not "what shape is it?" but "**what is it made of?**" — inferring material parameters (Young's modulus `E`, Poisson's ratio `nu`, density `rho`, yield stress `sigma_y`) from three-view RGB deformation videos, using MPM simulation for data generation and ground truth.

**Where this repository is heading:** force is the physical bridge between the two lines (material properties = the relationship between force and deformation). By adding calibrated force to the real-world rig, the shape line can grow from geometry reconstruction toward real-world, touch-based physical understanding.

---

## 2. `sensor-data-collection/` — Real-World Acquisition Rig

Records **synchronized** data from three sources:

| Stream | Hardware | Role |
|---|---|---|
| Tactile frames (16×16, 8-bit) | Velostat piezoresistive array, scanned by **Arduino Nano + dual CD74HC4067 multiplexers** | **Model input** (the only stream that remains at deployment) |
| Video (3 views) | 3× Teledyne FLIR **Blackfly S USB3** (PySpin / Spinnaker SDK) | **Shape ground truth** (training only) |
| Force curve | **Mark-10** force gauge + IntelliMESUR tablet (CSV via email) | **Force ground truth** (training only) |

### 2.1 Hardware (`Arduino files/`)

- **`MatrixArrayDual4067Binary.ino`** — the one firmware to flash (Arduino Nano, ATmega328P).
  - Scans the 16×16 array through two CD74HC4067 muxes: row address pins **D4–D7** (inhibit **D8**), column address pins **A1–A4** (inhibit **A5**).
  - Streams binary frames over serial at **500000 baud**: magic `M16B`, version, dimensions, 8-bit ADC payload (256 bytes, row-major), frame index, device timestamp, 16-bit checksum.
  - **Camera sync**: emits a 100 µs rising-edge pulse on **D9** — wired to the hardware-trigger input of *all three* cameras. This pulse is what makes the three videos and the tactile stream line up. Cameras must be configured for external hardware trigger.

### 2.2 Acquisition software (`Python files/`)

| Script | What it does |
|---|---|
| `run_multimodal_workflow.py` | **Main entry point.** Orchestrates the full Stage-A capture (`--port COMx`, optional `--camera-serials`, `--output`) |
| `capture_3blackfly_sensor_force.py` | Simultaneous camera + tactile capture (called by the workflow) |
| `repair_sensor_session.py` | Detects/repairs corrupted tactile frames in a session |
| `export_multimodal_mp4.py` | Converts captured frames to synchronized color MP4s, verifies, cleans up BMPs |
| `align_force_curve_only.py` | **Stage B.** Aligns the emailed Mark-10 CSV to the session using *curve-shape matching* between the force curve and the tactile signal (timestamps are ignored). Prints `score` / `confidence` / `peak_margin` |
| `replay_capture_multimodal.py` | Replays a session to visually verify synchronization |
| `requirements.txt` | Python deps (PySpin installed separately from the Teledyne wheel) |

### 2.3 Quick start (Windows, tested config)

1. Python 3.10 (64-bit), Spinnaker SDK 4.3.0.190 + matching Teledyne PySpin wheel, NumPy 1.26.4.
2. Flash `Arduino files/MatrixArrayDual4067Binary.ino` (board: Arduino Nano; if upload fails, try the *Old Bootloader* processor option). Note the COM port.
3. Verify D9 reaches all three camera trigger inputs; confirm each camera streams in SpinView, then **close SpinView**.
4. ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r "sensor-data-collection\Python files\requirements.txt"
   python -m pip install "path\to\teledyne\pyspin\wheel"
   ```
5. **Stage A** — capture: `python run_multimodal_workflow.py --port COM7` → baseline ≈1 s → run IntelliMESUR + perform the press/bend action → baseline ≈1 s → `Ctrl+C` → wait for MP4 verification.
6. **Stage B** — force import: `python align_force_curve_only.py <session folder> <force csv>` (prefer `confidence=high`).
7. Verify: `python replay_capture_multimodal.py <session folder>`.

Full details: [`sensor-data-collection/README.md`](./sensor-data-collection/README.md) (original, unmodified).

---

## 3. `DeformFieldBench/` — Simulation Benchmark & Models

A benchmark for **material parameter estimation from deformable-object videos**: 5000 simulated samples; each `sample_pack.npz` holds three-view RGB deformation videos, force masks, projected flow fields, stress fields, object masks, and GT parameters (`E`, `nu`, `rho`, `sigma_y`).

- **Dataset**: [Physical Field Material Parameter 5000 (Kaggle)](https://www.kaggle.com/datasets/anonymous336/physical-field-material-parameter-5000) — *not stored in this repo*; download or regenerate via simulation.
- **Checkpoints**: [HandsomeHusky/DeformFieldBench (Hugging Face)](https://huggingface.co/HandsomeHusky/DeformFieldBench/tree/main) — `bash pretrained/download_models.sh`.
- All "videos" here are **simulation renders** (Taichi/Warp MPM), not real footage.

### 3.1 Map of the codebase

| Folder | What it contains |
|---|---|
| `simulation/` | MPM physics simulation (Taichi + Warp): `mpm_solver_warp/`, `particle_filling/`, a vendored `gaussian-splatting/` renderer, per-action configs (`press/drop/shear/stretch/bend_cube_jelly.json`, …) |
| `configs/` | Run configs: `simulation/` (dataset generation), `my_model/`, `logic_model/` (training) |
| `my_model/` | **Arch4** — supervised parameter-regression baseline (`my_model.train`) |
| `logic_model/` | **Logic Model** — the final model (`logic_v2_dino`: frozen DINOv2 encoder, temporal transformer adapter, multi-view fusion, dense field heads + physics bottleneck) |
| `eval/`, `eval_abalation/` | Unified evaluation: parameter metrics (MAE/RMSE/MAPE/R²), field metrics (MSE/SSIM/IoU), ablation configs, **parameter-ambiguity replay** (re-simulate with predicted params, compare via SSIM/PSNR) |
| `vlm_benchmark/` | Benchmarks vision-language models (OpenAI/DashScope/ARK clients) on the same task |
| `my_utils/`, `utils/` | Sample packing, LMDB caching, camera/rendering utilities |
| `pretrained/` | Checkpoint download script + registry (`models.yaml`) |
| `scripts/` | Entry-point shell scripts (`run_simulation_sample_pack.sh`, `train_logic.sh`, `train_my_model_param_only.sh`, `eval_*.sh`, `setup_env.sh`) |

### 3.2 Typical commands

```bash
# Environment (Python 3.10; torch 2.8; taichi 1.5.0; warp_lang 0.10.1)
pip install -r DeformFieldBench/requirements.txt
source DeformFieldBench/scripts/setup_env.sh

# Generate one simulated sample (press/drop/shear/stretch/bend)
PLY_PATH=/path/to/object.ply SIM_TYPE=press \
CONFIG=configs/simulation/train_config_dataset_full.json \
OUTPUT_PATH=outputs/simulation_sample \
bash scripts/run_simulation_sample_pack.sh

# Train the supervised baseline (Arch4, parameters only)
NUM_GPUS=8 bash scripts/train_my_model_param_only.sh

# Train the final Logic Model
NUM_GPUS=8 bash scripts/train_logic.sh
```

Full details: [`DeformFieldBench/README.md`](./DeformFieldBench/README.md) (original, unmodified).

---

## 4. How the Pieces Fit Together (and What Comes Next)

**Roles of each stream, stated once and precisely:**

- The **tactile array is the input** — the only sensor that remains at deployment. "Camera-free" means *no cameras at inference*; cameras supervise training only.
- The **three cameras are the shape answer key** (multi-view triangulation; consumer RGB-D was rejected because its depth precision cannot resolve millimeter-level soft-body deformation on a textureless surface).
- The **Mark-10 is the force answer key** — the new stream this project adds on top of the published shape-only work.

**Next-stage direction (why this consolidation exists):**

1. Reproduce the 3-camera + tactile capture (Section 2) and the shape-reconstruction result of the predecessor paper.
2. Add force: train *tactile → shape + contact force*, keeping the camera-free / zero-shot properties.
3. Bridge to the physics line: with measured force + reconstructed deformation, move toward real-world material/physical understanding — the question DeformFieldBench answers in simulation — and toward robot applications (force-aware soft grippers, contact-rich data collection for embodied AI).

## 5. Repository Hygiene

- Large data (capture sessions, `auto_output/` datasets, checkpoints) must **not** be committed — see `.gitignore`. Use the Kaggle/HF links or lab storage.
- The two source folders are snapshots; upstream repos remain the canonical history. Keep any new code (force pipeline, new models) in **new top-level folders** so the provenance stays clean.
