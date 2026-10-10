# Reconstruction: multi-view depth + 4D tracking of a pressed soft object

This folder turns one synchronized capture session (4 RGB cameras + RealSense D455 +
Mark-10 force gauge + tactile array, recorded with `../sensor-data-collection`) into:

- **metric depth maps** for each of the 4 RGB cameras at every frame (Depth Anything 3 giant, conditioned on our calibration),
- **3D point trajectories** over time (MVTracker) and a fused dense point cloud per frame,
- a **Rerun recording** that plays every camera, every depth map, the D455, the 3D reconstruction, and the force-gauge comparison on one shared timeline,
- an **accuracy report**: how deep the reconstruction thinks the indenter pressed vs. what the force gauge measured.

Only code and docs live in git. Sessions, model weights, and outputs stay on your disk
(see [What is not in git](#what-is-not-in-git)).

---

## Contents

1. [Pipeline at a glance](#1-pipeline-at-a-glance)
2. [Folder layout](#2-folder-layout)
3. [Setup](#3-setup)
4. [Input: a capture session](#4-input-a-capture-session)
5. [Run the reconstruction](#5-run-the-reconstruction)
6. [Look at the results](#6-look-at-the-results)
7. [Measure accuracy against the force gauge](#7-measure-accuracy-against-the-force-gauge)
8. [Results so far (session 20261009)](#8-results-so-far-session-20261009)
9. [Known limitations and open questions](#9-known-limitations-and-open-questions)
10. [Other depth backends](#10-other-depth-backends)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Pipeline at a glance

```
 session folder (from sensor-data-collection)
   camera_<i>_<serial>_6fps.mp4  x4      calibration.json  (4 RGB cameras, world frame, mm)
   d455/color, d455/depth                ../calibration/d455_world.json  (D455 pose in the same world frame)
   multimodal_video_alignment.csv        force_samples_aligned.csv  (Mark-10, ~85 Hz)
        │
        ▼
 run_test_session.py
   2/8 decode       undistort + resize the 4 videos (default 512x384)
   3/8 depth        DA3 giant, all 4 views of a timestamp in one pass, given our K and R|t
                    -> metric depth in metres (run in its own venv as a subprocess)
   4/8 queries      seed points on the object from the depth maps
   5-6/8 MVTracker  track those points in 3D through time (sliding 12-frame window)
   7/8 files        tracks_4d.npz/.csv, per-frame PLY (tracks + dense fused cloud)
   8/8 rerun        tracks_4d.rrd (3D + all camera views + depth panels)
        │
        ├─► eval_indentation.py   indentation depth per frame: force gauge vs reconstruction
        │                         (4 RGB, 4 RGB + D455, D455 alone) -> indentation.csv
        └─► export_rerun.py        rebuild tracks_4d.rrd with those curves on the timeline
```

Why these pieces:

- **DA3 giant** is not a metric monocular model on its own. We feed it the 4 undistorted views of
  one timestamp together with our calibrated intrinsics and metric world-to-camera extrinsics, and use
  its `align_to_input_ext_scale=True` mode, so its depth comes out in our world scale. Every timestamp is
  processed independently, so nothing assumes the scene is static.
- **MVTracker** gives *correspondence over time* (which surface point went where), which per-frame
  depth alone cannot.
- **The force gauge is the ground truth for indentation depth**: once the indenter touches the sponge,
  the crosshead travel since contact is exactly how deep the surface at the indenter went.
- **The D455** is calibrated into the same world frame as the 4 RGB cameras, so its measured depth can be
  shown in 3D, compared pixel-by-pixel with the reconstruction, and fused as a fifth view.

---

## 2. Folder layout

```
reconstruction/
├── scripts/
│   ├── run_test_session.py          main pipeline (decode -> depth -> MVTracker -> files -> Rerun)
│   ├── test_session_adapter.py      reads a session folder (videos, calibration, alignment)
│   ├── da3_depth_backend.py         launches DA3 in .venv-da3 and caches its output
│   ├── da3_worker.py                the DA3 inference itself (runs inside .venv-da3)
│   ├── duster_depth_backend.py      alternative depth: DUSt3R global alignment
│   ├── foundationstereo_*.py        alternative depth: rectified stereo pairs
│   ├── rerun_views.py               camera images, depth panels, D455, indentation curves, layout
│   ├── export_rerun.py              rebuild tracks_4d.rrd from a finished run (no model rerun)
│   ├── eval_indentation.py          force-gauge ground truth vs reconstructed indentation, per frame
│   ├── eval_accuracy.py             static / plane / board / press accuracy tests
│   ├── mvtracker_visualizer_gradio.py   web UI: embedded Rerun + deformation MP4 renderer
│   ├── duster_depth_viewer_gradio.py    web UI: 4 RGB + 4 depth maps side by side (DA3 or DUSt3R)
│   ├── download_assets.py           MVTracker checkpoint
│   └── setup_*.sh                   Linux/conda environment setup
├── docs/                            extra notes (Chinese): data format, DA3 validation, DUSt3R flow
├── submodule/                       third-party code pinned to fixed commits (do not edit)
│   ├── mvtracker/                   ethz-vlg/mvtracker @ ceea8ad
│   ├── depth-anything-3/            bytedance-seed/depth-anything-3 @ 3d835ec
│   ├── duster/                      ethz-vlg/duster @ 51baa9c
│   └── FoundationStereo/            NVlabs/FoundationStereo @ 6e88068
└── outputs/                         results and DA3 weights (git-ignored)
```

Keep project code in `scripts/`. Never put files into `submodule/*`; the parent repo pins their commits.

---

## 3. Setup

```bash
git clone --recurse-submodules <repo-url>
cd 3D_Reconstruction/reconstruction
# already cloned without submodules?
git submodule update --init --recursive
```

Two Python environments are used, because DA3 and MVTracker pin different dependencies:

| Environment | Used for | Called how |
|---|---|---|
| **MVTracker env** | everything you run by hand (`run_test_session.py`, viewers, evaluation) | you run it directly |
| **DA3 env** | DA3 inference only | `run_test_session.py` starts it as a subprocess (`--da3-python`) |

### 3a. Windows + RTX 50-series (what we use now)

The repo's `setup_*.sh` scripts pin PyTorch 2.3/2.4 + CUDA 12.1, which **do not support Blackwell GPUs**
(RTX 50xx, sm_120). On our Windows workstation (RTX 5070, 12 GB) the environments were built by hand with
[uv](https://docs.astral.sh/uv/) instead (the commands below reproduce these versions; adjust if a pin has moved):

| venv | Python | Key packages |
|---|---|---|
| `.venv-mvtracker` | 3.10 | torch 2.7.1+cu128, `submodule/mvtracker/requirements.txt` (numpy 1.24.3), gradio, rerun-sdk 0.21 |
| `.venv-da3` | 3.12 | torch 2.7.1+cu128, torchvision 0.22.1, xformers 0.0.31.post1, `-e submodule/depth-anything-3`, plus `addict ruamel.yaml pytz sniffio` (used but not declared by DA3) |

```powershell
cd reconstruction
uv venv .venv-mvtracker --python 3.10
uv pip install --python .venv-mvtracker\Scripts\python.exe torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-mvtracker\Scripts\python.exe -r submodule\mvtracker\requirements.txt gradio rerun-sdk==0.21.0

uv venv .venv-da3 --python 3.12
uv pip install --python .venv-da3\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 xformers==0.0.31.post1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-da3\Scripts\python.exe -e submodule\depth-anything-3 addict ruamel.yaml pytz sniffio
```

Notes:
- xformers' attention kernels crash on sm_120, but DA3 only uses its SwiGLU op, which works.
- `pointops` is not installed, so MVTracker uses its slower `torch.cdist` KNN fallback (it logs a warning). This is fine, just slower and heavier on VRAM.

Model weights:

```powershell
$env:HF_HOME = "$PWD\.cache\huggingface"
.venv-mvtracker\Scripts\python.exe scripts\download_assets.py          # MVTracker checkpoint -> submodule/mvtracker/checkpoints/
.venv-da3\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('depth-anything/DA3-GIANT', revision='7cd62ae9315b9dff094d2d300e4ad012640607dd', local_dir='outputs/models/DA3-GIANT', allow_patterns=['config.json','model.safetensors'])"
```

**Strongly recommended on Windows:** NVIDIA Control Panel → *Manage 3D settings* → **CUDA - Sysmem Fallback
Policy → Prefer No Sysmem Fallback**. Without it, running out of VRAM does not raise an error; the driver
spills into system RAM and the whole PC freezes.

### 3b. Linux + conda (original setup, older GPUs)

```bash
bash scripts/setup_mvtracker.sh        # conda env "mvtracker" + MVTracker checkpoint
bash scripts/setup_da3.sh              # .venv-da3 + DA3-GIANT weights in outputs/models/DA3-GIANT
bash scripts/setup_duster.sh           # only for the DUSt3R backend
bash scripts/setup_foundationstereo.sh # only for the FoundationStereo backend
```

On Linux, prefix commands with `conda run --no-capture-output -n mvtracker python ...` and use
`--da3-python .venv-da3/bin/python` (the default).

---

## 4. Input: a capture session

A session is the folder written by `sensor-data-collection` (capture → `export_multimodal_mp4.py` →
`align_force_curve_only.py`). The reconstruction reads:

| File | Used for |
|---|---|
| `camera_<i>_<serial>_6fps.mp4` (x4), `multimodal_video_export.json` | RGB frames (synchronized by hardware trigger) |
| `calibration.json` | intrinsics + extrinsics of the 4 cameras, world frame in mm (z up, z = 0 on the platform) |
| `multimodal_video_alignment.csv` | per video frame: D455 frame files, force time / load / crosshead distance |
| `force_samples_aligned.csv` | full-rate Mark-10 samples, used to find each press's contact point |
| `d455/d455.json`, `d455/color/*.jpg`, `d455/depth/*.png` | D455 stream (depth PNG in device units, 1 mm) |
| `../calibration/d455_world.json` (in the repo) | D455 colour + depth camera poses in the same world frame, plus a small depth correction; produced by `calibration/calibrate_d455.py` |

Format details: [docs/MVTRACKER_DATA_TEST.md](docs/MVTRACKER_DATA_TEST.md).

> **Time alignment is estimated, not hardware-synced.** The 4 cameras, tactile array, and D455 share
> triggers/timestamps, but the Mark-10 does not. `align_force_curve_only.py` finds the force log's start
> by matching the force curve to the tactile signal (see `force_alignment.json`: score, runner-up, confidence).

---

## 5. Run the reconstruction

From `reconstruction/` (PowerShell; this is the run that produced the numbers in section 8):

```powershell
cd reconstruction
$env:HF_HOME = "$PWD\.cache\huggingface"
$S = "..\sensor-data-collection\test\data\20261009"

.venv-mvtracker\Scripts\python.exe scripts\run_test_session.py `
  --session-dir $S --start 0 --target-frames 368 --max-frames 368 --width 512 --height 384 `
  --depth-backend da3 --da3-python .venv-da3\Scripts\python.exe --da3-process-res 504 `
  --da3-confidence-percentile 10 --min-depth-m 0.03 --max-depth-m 2 `
  --query-grid-size 16 --device cuda --output-dir outputs\20261009
```

The output folder name encodes the request: `outputs/20261009/seconds_0_end_target_368_da3/`.

Important options:

| Option | Meaning |
|---|---|
| `--start`, `--end` | time range in **video seconds** (`--end` omitted = to the end) |
| `--target-frames N` | sample N evenly spaced frames from the range; or use `--step K` for every K-th frame |
| `--max-frames` | safety limit; must be ≥ the number of frames selected |
| `--width/--height` | working resolution for depth and tracking (lower = less VRAM) |
| `--da3-confidence-percentile` | drop the least-confident X % of DA3 pixels per view |
| `--min-depth-m/--max-depth-m` | depth outside this range is discarded |
| `--query-grid-size` | seed grid per view; tracks ≈ 4 × size² before filtering (main VRAM knob for MVTracker) |
| `--no-rerun-views` | skip camera/depth/D455 panels in the `.rrd` (smaller file) |

What a run writes:

| File | Content |
|---|---|
| `da3/frame_NNNNN.npz`, `da3/depths.npz`, `da3/manifest.json` | raw DA3 output per timestamp + cache fingerprint |
| `depths_da3_m.npz` | the filtered depth MVTracker actually used, `(views, frames, H, W)` in metres |
| `tracks_4d.npz`, `tracks_4d.csv` | trajectories `(frames, tracks, 3)` in world metres + visibility |
| `ply/tracks_<frame>.ply`, `ply_dense/scene_<frame>.ply` | per-frame visible tracks and fused dense cloud (world, metres) |
| `tracks_4d.rrd` | Rerun recording (see section 6) |
| `run.json` | everything about the run (frames, times, settings) |

**Resuming / reusing depth.** DA3 is cached per timestamp with a fingerprint of the RGB, calibration,
frame indices, model, and settings. If a run crashes after depth, rerun the *same* command and DA3 is
skipped (do not pass `--recompute-depth`). To track a sub-range without recomputing depth, point the `npz`
backend at an existing cache; it picks frames by video frame index:

```powershell
.venv-mvtracker\Scripts\python.exe scripts\run_test_session.py `
  --session-dir $S --start 17 --end 57 --step 1 --max-frames 241 --width 512 --height 384 `
  --depth-backend npz --depth-path outputs\20261009\seconds_0_end_target_368_da3\da3\depths.npz `
  --min-depth-m 0.03 --max-depth-m 2 --query-grid-size 16 --device cuda --output-dir outputs\20261009
```

**VRAM.** Step 6/8 (MVTracker) is the peak. It tracks in 12-frame windows, but the whole clip's RGB-D is
on the GPU, so memory grows with frames × resolution. 368 frames at 512×384 with grid 16 completed on the
12 GB RTX 5070.
If it doesn't: lower `--query-grid-size` (8), `--width/--height` (384×288), or the frame count (`--step 2`).

---

## 6. Look at the results

### Web UI (Gradio + Rerun)

```powershell
.venv-mvtracker\Scripts\python.exe scripts\mvtracker_visualizer_gradio.py `
  --result-dir outputs\20261009\seconds_0_end_target_368_da3 `
  --viewer-url "http://127.0.0.1:9090/?url=ws://127.0.0.1:9877"
```

Open http://127.0.0.1:7861 and click **Load interactive Rerun**. (Or open the file directly:
`.venv-mvtracker\Scripts\rerun.exe <result-dir>\tracks_4d.rrd`.)

Everything is on one `frame_index` timeline (= video frame index), so scrubbing moves all panels together:

| Panel | What it shows |
|---|---|
| 3D tracks + clouds | fused reconstructed cloud, track points and trails, the 4 RGB camera frustums, the D455 frustums and its measured depth as a 3D cloud, and the yellow circle where indentation is measured |
| RGB cam 0–3 | the 4 camera frames with the tracks projected on them |
| Recon depth cam 0–3 | the depth each RGB camera's reconstruction used |
| D455 colour / D455 measured depth | the RealSense stream |
| Recon depth in D455 view | the fused reconstruction rendered into the D455 depth camera, pixel-aligned with the measured depth |
| \|D455 − recon\| | per-pixel absolute difference, colour scale 0–20 mm |
| D455 vs recon depth (mm) | per-frame median / P90 / signed difference of the two maps above |
| Indentation depth (mm) | force-gauge ground truth vs each method (section 7) |
| Indentation error vs force gauge (mm) | each method minus ground truth |

Hover any depth panel to read the value in metres. If the layout looks stale, click the ↺ next to
*Blueprint* (the viewer remembers old layouts).

### Rebuild the `.rrd` without rerunning any model

After changing visualization code, or after running `eval_indentation.py`:

```powershell
.venv-mvtracker\Scripts\python.exe scripts\export_rerun.py --result-dir outputs\20261009\seconds_0_end_target_368_da3
```

It re-decodes the frames, reuses `tracks_4d.npz` and the depth cache, picks up `indentation.csv` if present,
and overwrites `tracks_4d.rrd` (~930 MB for 368 frames; `--view-depth-downsample 4` makes it smaller).

### Depth maps only

```powershell
.venv-mvtracker\Scripts\python.exe scripts\duster_depth_viewer_gradio.py `
  --result-dir outputs\20261009\seconds_0_end_target_368_da3 --server-port 7862
```

4 RGB views beside their depth maps, synchronized playback (works for DA3 and DUSt3R results).

---

## 7. Measure accuracy against the force gauge

```powershell
.venv-mvtracker\Scripts\python.exe scripts\eval_indentation.py --result-dir outputs\20261009\seconds_0_end_target_368_da3
.venv-mvtracker\Scripts\python.exe scripts\export_rerun.py    --result-dir outputs\20261009\seconds_0_end_target_368_da3
```

Writes `indentation.csv` (one row per frame) and `indentation_summary.json`, and prints the error table.

### Ground truth: the force gauge

1. In the ~85 Hz force samples, a **press** is a stretch where the load stays above the unloaded level
   + 1 N for at least 0.5 s. (The gauge is not zeroed; the unloaded level is estimated, e.g. −7.4 N.
   Short spikes when the crosshead starts or stops are ignored.)
2. The crosshead distance at the start of a press is that press's **contact distance**. Each press gets its
   own, because the sponge does not fully recover between presses (265.5 → 265.1 → 264.7 → 264.2 mm).
3. For every video frame inside a press: **indentation = contact distance − current distance**.

### Reconstructed indentation (same measurement for every method)

1. **Where:** a disk of radius 30 mm around the dent centre (world xy), z between 0 and 120 mm.
   The centre is found from the D455 at peak load (centroid of the cells that dropped by at least half
   the deepest drop, on the sponge top), or given with `--centre-xy X Y`. The D455 only chooses *where*
   to look; it does not enter the RGB-only numbers.
2. **Height map:** 2 mm cells; each view gives the median z of its points in each cell.
3. **Rest surface:** the height map averaged over the 6 unloaded frames just before that press.
4. **Drop** = rest height − current height, per cell. **Indentation estimate = 95th percentile** of the drop
   in the disk. The indenter hides the bottom of the dent; the deepest *visible* surface is its rim, which
   moves with the punch. The indenter itself sits above the surface (negative drop) and is ignored.

The three methods:

| Method | Views fused per cell |
|---|---|
| `rgb4` | the 4 RGB cameras' reconstructed depth (median of the 4 per-view cell medians) |
| `rgb4_d455` | the same plus the D455 measured depth as a 5th view (median of 5; the D455 is one vote, not outnumbering the others by pixel count) |
| `d455` | the D455 alone (reference sensor) |

Useful options: `--centre-xy`, `--radius`, `--cell`, `--percentile`, `--contact-threshold`, `--reference-frames`.

### Other accuracy tests (`eval_accuracy.py`)

Need dedicated segments in the capture; all results in mm.

| Test | Record | Answer key | Reports |
|---|---|---|---|
| `--static A:B` | nothing moves | depth must not change | per-pixel temporal noise per camera |
| `--plane A:B` | empty platform | z = 0 by calibration | platform height, flatness, camera-to-camera offset |
| `--board A:B` | ChArUco board lying still | corners triangulated from calibration (~0.16 mm) | signed depth error per camera |
| `--press A:B` | indenter pressing | crosshead travel since contact | surface drop in a ring just outside a known indenter (`--ring`, `--press-rect`) |

```powershell
.venv-mvtracker\Scripts\python.exe scripts\eval_accuracy.py --run-dir <result-dir> --static 0:2.5 --board 3:6 --press 8:30 --ring 5,9
.venv-mvtracker\Scripts\python.exe scripts\eval_accuracy.py --selftest   # plants known errors in synthetic data
```

---

## 8. Results so far (session 20261009)

Sponge, 4 presses to 3.6 / 8.1 / 12.7 / 17.2 mm, 368 frames at 6 fps, DA3 giant at 512×384.

**Indentation vs force gauge** (154 loaded frames, mm; negative bias = reconstructed too shallow):

| Method | Mean \|error\| | Bias | RMSE | Max \|error\| (frame, time) | Slope recon/truth | Relative error* |
|---|---|---|---|---|---|---|
| 4 RGB | 3.3 | −3.3 | 3.8 | 7.2 (frame 311, 51.8 s) | 0.66 | 38 % |
| **4 RGB + D455** | **2.4** | **−2.4** | **2.8** | **6.1** (frame 316, 52.7 s) | **0.75** | **28 %** |
| D455 alone | 4.8 | +4.7 | 6.3 | 18.2 (frame 289, 48.2 s) | 1.51 | 50 % |

\* mean of \|error\| / truth over frames with truth ≥ 2 mm.

What this says:
- **Both RGB-based methods under-estimate the dent by a roughly constant fraction**: the absolute error grows
  with depth, the relative error stays near 30–37 % (4 RGB) and 23–28 % (4 RGB + D455) above 5 mm. The depth
  network smooths the dent. Correlation with the truth is 0.97–0.98, so the shape of the curve is right.
- **Adding the D455 as a fifth view helped in every setting we tried** (percentile 90/95/98, radius 20/30/40 mm).
- **The D455 alone is unreliable here**: flying pixels at the indenter's edges make it over-read deep presses,
  and its result swings strongly with the measurement settings.
- The worst errors are all in the deepest press, during the hold (crosshead not moving), so they are
  reconstruction error, not time misalignment.

**Reconstruction vs D455 depth, whole scene** (all pixels both see): median \|difference\| ≈ 5.5 mm per frame,
signed median ≈ 0 mm, i.e. no scale or offset error between the calibrated rigs. The D455 itself is good to
~1–3 mm after its correction (board check during `calibrate_d455.py`).

---

## 9. Known limitations and open questions

- **The indenter's size is not used yet.** `eval_indentation.py` reads a 30 mm disk and a percentile. With the
  known indenter footprint we should read only a thin band just outside its edge (as `eval_accuracy.py --press`
  does with `--ring` / `--press-rect`) and mask the indenter out; this should remove most of the sensitivity
  to `--percentile`/`--radius` and to D455 flying pixels.
- **"4 RGB + D455" is fused at evaluation time** (per height-map cell), not fed into DA3 or MVTracker.
- **Force time alignment is estimated** (curve matching, no shared clock). At 6 fps one frame is 167 ms; during
  ramps (~7 mm/s) a one-frame offset is ~1 mm of ground truth. Hold phases are unaffected.
- **Unloaded frames between presses** show 2.5–4 mm of "dent": partly real (the sponge recovers slowly after
  release), partly noise.
- **RGB scaling into MVTracker:** `run_test_session.py` passes RGB in 0–1 and MVTracker divides by 255 again
  (the official MVTracker demo does the same). Under bf16 this leaves very little colour information in the
  feature encoder. Worth testing whether passing 0–255 improves tracking.
- **Calibration accuracy is the ceiling.** Calibration error becomes label error nothing downstream can detect.

---

## 10. Other depth backends

`--depth-backend` also accepts:

| Backend | Idea | Notes |
|---|---|---|
| `duster` | DUSt3R pairwise pointmaps + global alignment with our poses | Linux setup script; see [docs/DUSTER_RECONSTRUCTION_TRACKING_ZH.md](docs/DUSTER_RECONSTRUCTION_TRACKING_ZH.md) |
| `foundationstereo` | rectified stereo on camera pairs `0-2,1-3` at full 2048×1536 | non-commercial license; its own conda env; slow, GB-sized cache |
| `moge2` | monocular MoGe-2 | fast baseline, not multi-view consistent |
| `npz` | any precomputed metric depth (`--depth-path`) | also how to reuse a DA3 cache for a sub-range |

DA3 validation notes: [docs/DA3_VALIDATION_ZH.md](docs/DA3_VALIDATION_ZH.md).

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| PC freezes at `6/8 inference` | VRAM full and spilling to RAM: set *Prefer No Sysmem Fallback* (section 3a), then lower `--query-grid-size`, resolution, or frames |
| Crash after DA3 finished | rerun the same command; DA3 is read from its cache |
| `rerun CLI is not installed` in the web UI | fixed: the viewer now looks next to the running Python; or activate the venv first |
| Rerun shows an old layout | click ↺ next to *Blueprint* |
| `pointops not found` warning | harmless; slower KNN fallback |
| `ModuleNotFoundError: dust3r` | `git submodule update --init --recursive`, then `setup_duster.sh` |
| A submodule shows as modified | something was written inside `submodule/*`; move it to `scripts/` |

### What is not in git

`.gitignore` excludes `outputs/`, `data/`, `.cache/`, venvs, checkpoints, `*.npz`, `*.npy`, `*.ply`, `*.rrd`,
`*.mp4`, and model weights. Before committing:

```bash
git status --short
git submodule status --recursive
```

Commit only code, docs, `.gitmodules`, and submodule pointers. Never force-add data, outputs, or weights.
