# Tactile Sensor Data Collection

This project records synchronized data from:

- one 16x16 tactile sensor;
- four Teledyne Blackfly S USB3 cameras; and
- one Mark-10 force gauge running IntelliMESUR on its own tablet.

The camera count is not hard-coded. Four is the default everywhere; pass
`--cameras N` to capture with a different number. Post-processing and replay
read the count back from the capture session, so sessions recorded with three
cameras still repair, export, and replay unchanged.

The workflow has two stages:

1. capture and process camera/sensor data;
2. import and align the force CSV received by email.

## 1. Computer Environment Setup

### 1.1 Recommended Software

The tested Windows configuration is:

| Component | Version |
|---|---|
| Python | CPython 3.10.11, 64-bit |
| Spinnaker SDK / SpinView | 4.3.0.190 |
| PySpin | Matching `cp310-win_amd64` Teledyne wheel |
| NumPy | 1.26.4 |

The Spinnaker SDK and PySpin versions must match. Install PySpin only from the
wheel supplied by Teledyne; do not install an unrelated package named
`pyspin` from PyPI.

Official references:

- [Teledyne PySpin installation guide](https://www.teledynevisionsolutions.com/support/support-center/technical-guidance/iis/installing-pyspin-for-the-spinnaker-sdk/)
- [Teledyne Spinnaker SDK downloads](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/GetResourcesSupportDownloads/)
- [Python 3.10.11](https://www.python.org/downloads/release/python-31011/)

### 1.2 Install Spinnaker and Verify the Cameras

1. Install 64-bit Python 3.10.
2. Install the full 64-bit Spinnaker SDK, including SpinView and USB3 camera
   drivers.
3. Connect all four cameras.
4. Open SpinView and confirm that every camera can stream.
5. Exit SpinView before starting Python acquisition.

SpinView and PySpin should not access the same camera simultaneously.

### 1.2.1 USB3 Bandwidth for Four Cameras

Four full-resolution cameras need more USB3 bandwidth and more sustained disk
write throughput than three. Before the first four-camera trial:

- put the cameras on separate USB3 host controllers where possible; two
  cameras sharing one controller can drop frames that three cameras never did;
- keep `--output` on a fast drive. Raw Bayer BMP at 2048x1536 and 24 fps costs
  roughly 72 MiB/s per camera, so four cameras need about 290 MiB/s sustained.
  The capture program prints this estimate at startup; and
- if `incomplete_images` or `dropped_images` appear in `session.json` for the
  newly added camera only, suspect the USB3 controller before the sensor.

### 1.3 Create the Python Environment

Open the project folder in File Explorer, right-click inside the folder, and
choose **Open in Terminal**. Then run:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PySpin is not included in `requirements.txt`. Install the Teledyne PySpin
wheel that matches the installed Spinnaker SDK and Python 3.10 by typing:

```powershell
python -m pip install "path\to\your\pyspin\wheel\file"
```

### 1.4 Configure Arduino and IntelliMESUR

1. Install Arduino IDE and the USB-serial driver required by the Arduino Nano.
2. Upload:
   `Arduino files/MatrixArrayDual4067Binary.ino`.
3. Record the Arduino COM port.
4. On the force-gauge tablet, confirm IntelliMESUR communication, units, and
   CSV export.
5. Email the original exported CSV to the acquisition computer after each
   experiment. Do not open and re-save it in Excel.

## 2. Experiment Workflow

### 2.1 Prepare the Experiment

Before every trial:

- connect the Arduino, sensor, and all four cameras;
- verify that D9 reaches all four rising-edge camera trigger inputs. The same
  D9 pulse is wired in parallel to every camera; the firmware needs no change
  when a camera is added;
- close SpinView;
- zero the force gauge;
- prepare one IntelliMESUR run on the tablet;
- keep the camera viewpoints fixed; and
- record the sensor ID, trial ID, loading type, and camera viewpoints.

Open the project folder in a terminal and activate its environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2.2 Stage A: Capture Cameras and Sensor

Start acquisition:

```powershell
python capture_3blackfly_sensor_force.py --port COM3 --cameras 4 --output D:\tactile_data\run01
```

Replace `COM3` with the Arduino port shown in Device Manager, and give
`--output` a path on a disk that can sustain the write load — four cameras at
24 fps is **~288 MiB/s**, and a disk that cannot keep up drops frames silently
until the stream buffers overflow. The script refuses to start below
`--minimum-free-gb` (20 by default).

Do **not** pass `--exposure-us` or `--gain-db`: they overwrite the values
`tune_exposure.py --auto` stored in the cameras.

There used to be a `run_multimodal_workflow.py` wrapper around this. It was
removed: it did not forward the exposure arguments, and it deleted the source
BMPs unless `--keep-bmp` was remembered. Call the capture script directly, then
export and align in the two explicit steps below.

Camera serial numbers are detected and sorted automatically. To force a
specific camera order, add `--camera-serials CAM0 CAM1 CAM2 CAM3` — supply
exactly as many serial numbers as there are cameras.

The workflow expects four cameras and stops if a different number is found.
To run a different rig, add `--cameras N`, for example `--cameras 3` to repeat
an old three-camera setup.

To store data elsewhere, add `--output`, followed by the destination folder.
The folder can be dragged directly from File Explorer into the terminal.

Then:

1. wait for all cameras to report armed and for `Capture running`;
2. record approximately one second of unloaded baseline;
3. start the IntelliMESUR run on the tablet;
4. perform the pressure, bending, or combined action;
5. stop and export the IntelliMESUR run;
6. record approximately one additional second of unloaded baseline;
7. press `Ctrl+C` once; and
8. wait for MP4 generation and verified BMP cleanup.

Stage A is complete after all four color MP4 files have been verified and the
terminal prints the new capture session folder. One MP4 is written per camera,
named after its `camera_<index>_<serial>` folder.

### 2.3 Stage B: Import the Emailed Force CSV

Download the corresponding CSV attachment and run:

```powershell
python align_force_curve_only.py "path\to\your\capture\session\folder" "path\to\the\corresponding\force\csv"
```

The program asks for the capture session folder and the downloaded force CSV.
Drag each one from File Explorer into the terminal when prompted.

The aligner uses only force and sensor curve shapes. Email, creation,
modification, and download timestamps are ignored.

Review the printed `score`, `confidence`, and `peak_margin`. Prefer
`confidence=high`; inspect `low` or `ambiguous` results before using them.

The experiment is complete when force import finishes without an error and
prints its alignment score.

#### Faster, press-by-press alternative

`align_force_curve_only.py` searches by brute force and takes many minutes on a
10-minute run. `align_force_presses.py` does the same job in about a second and
labels every tactile frame:

```powershell
python align_force_presses.py "path\to\session" "path\to\force.csv"
```

It finds the time scale and offset with an FFT cross-correlation (the export's
`Time` column is not real time: it ran 5 % slow on 2026-10-12 and 17 % slow on
2026-10-09), then pairs every press and warps the clock through the contact
moments, which also absorbs drift that is not a straight line. It subtracts an
untared force offset, cuts a capture that stalled (no frame for > 1 s), and
writes `<session>/force_alignment/frames.csv` (per frame: load, distance,
indentation from contact, press number, loading/hold/unloading) with
`alignment.json` and `alignment.png`.

### 2.4 Quick Verification

To check the synchronized videos, sensor, and force data, run:

```powershell
python replay_capture_2d.py "path\to\your\capture\session\folder"
```

The replay window shows one preview per camera along the top row, with the
tactile image and the force curve below. Up to four cameras share a single
preview row; beyond that the previews wrap onto additional rows and the window
grows taller. Each preview is labelled with its camera index, so `[3]` is the
fourth camera.

**Use the 2D view to check a capture.** A top-down image lines up directly with
the camera previews above it, nothing is hidden behind anything else, and cell
addresses are immediate. It also marks the contact patch: `×` is the peak cell
and `○` the contact centroid, so "the indenter is here, the signal is there" is
one glance. Note that the colour scale is sized from the session's own signal
unless you pass `--range`.

Two things the 2D view is good at catching:

- **Baseline drift.** If the title says the contact covers over half the array,
  the whole sensor has moved away from the baseline taken at the start of the
  run, and the centroid is meaningless. Re-check the unloaded period.
- **Bad rows or columns.** A dead or offset line shows as a solid stripe along
  one edge instead of a wall of bars hiding everything behind it.

Useful options: `--labels` for per-cell values, `--no-smooth` for raw cells,
`--range` for a fixed colour scale, and `--mode voltage` for absolute voltage.
Keys: `space` play/pause, arrows step, `m` markers, `i` smoothing, `q` quit.

#### Exporting the view as a video

The replay windows are viewers and save nothing on their own. To get a file of
the whole view — camera previews, tactile image, force curve, contact markers —
add `--export`. No window opens; it renders offscreen and encodes with the
bundled FFmpeg.

```powershell
python replay_capture_2d.py "会话文件夹" --export clip.mp4 --export-range 42 49
python replay_capture_2d.py "会话文件夹" --export clip.gif --export-fps 8 --export-dpi 60
python replay_capture_2d.py "会话文件夹" --export whole_run.mp4
```

The extension picks the format: `.mp4` for archiving, `.gif` for slides and
chat. Options:

| Option | Meaning |
|---|---|
| `--export-range START_S END_S` | Export only this span of capture time |
| `--export-fps` | Output rate, sampled from the capture timeline so playback is real-time (default 12) |
| `--export-dpi` | Resolution; the figure is 17 in wide, so 80 → 1360 px (default 80) |

Rendering costs roughly **0.3 s per output frame**, because each frame decodes
four camera streams and redraws the whole figure. Measured on this workstation:
7 s of capture at 12 fps took 28 s. A whole 57 s session at 12 fps is about 4
minutes. Prefer `--export-range` around the event of interest.

Note that the sensor panel shows whatever the data says, so if the array has
open cells they appear in the video too. Fix the data before exporting anything
you plan to show.

### 2.5 The 3D View

The original replay is still there and takes the same arguments:

```powershell
python replay_capture_multimodal.py "path\to\your\capture\session\folder"
```

Bar height is a second visual channel for magnitude, which helps when judging
relative depth or showing the rig to someone. Prefer it for that; prefer the 2D
view for verifying a capture. Be aware that the z-axis is **voltage, not
displacement** — the mapping from one to the other is what this project is
trying to learn, so the height is not a deformation profile.

Both replays share their session loading, camera decoding, force panel, and
playback controls: `replay_capture_2d.py` imports them from
`replay_capture_multimodal.py`, so keep the two files in the same folder.

## 3. Bench Tools (`test/`)

Standalone sensor-only diagnostics. They need no cameras, no force gauge, and
no capture session — just the Arduino. Run them from the `test` folder with the
same virtual environment. None of them writes into a capture session; they only
read the sensor.

### 3.1 `live_sensor_2d.py` — Live 2D Pressure Image

Shows the array as a smooth 2D image using the same signal definition as the
capture pipeline (`V_sensor = Vdrive - V_adc`, then baseline subtraction), so
what looks right here looks the same in `replay_capture_multimodal.py`.

```powershell
python live_sensor_2d.py --list-ports          # find the Arduino COM port
python live_sensor_2d.py COM3                  # baseline-subtracted (default)
python live_sensor_2d.py COM3 --mode voltage   # absolute sensor voltage
python live_sensor_2d.py COM3 --mode raw       # raw 8-bit ADC
python live_sensor_2d.py --simulate            # no hardware, synthetic press
python live_sensor_2d.py --selftest            # headless render check, saves a PNG
```

Keep the sensor unloaded for the first second: the tool medians the first 24
frames into the no-load baseline before it starts showing contact.

| Key | Action |
|---|---|
| `b` | re-capture the no-load baseline |
| `i` | toggle smooth / raw-cell rendering |
| `v` | toggle per-cell value labels |
| `+` / `-` | display gain up / down |
| `]` / `[` | display range up / down |
| `s` | save a PNG snapshot |
| `q` or `Esc` | quit |

Useful options: `--gain` and `--range` for weak signals, `--rot90 N`,
`--flip-ud`, `--flip-lr` to orient the image to the physical sensor, and
`--contact-threshold` to tune contact detection.

### 3.2 `live_sensor_3d.py` — Live 3D Bar View

The bench counterpart of the 3D panel in `replay_capture_multimodal.py`: same
signal definition, same bar geometry and shading, but fed live from the
Arduino. Use it to judge the deformation shape before a full capture.

```powershell
python live_sensor_3d.py COM3                  # 3D bars, baseline-subtracted
python live_sensor_3d.py COM3 --style surface  # smooth surface, ~2x faster
python live_sensor_3d.py --simulate            # no hardware, synthetic press
python live_sensor_3d.py --selftest            # headless render + timing report
```

3D redraws are much slower than the sensor's 24 Hz: measured on this
workstation, `bars` sustains about 8 fps and `surface` about 13 fps. That is a
*display* limit only — acquisition still runs at full rate and the viewer always
shows the newest frame, dropping intermediate ones. Use `--style surface` or
`--max-render-fps` if the window feels sluggish. For a fluid view of the
contact patch, prefer the 2D tool; use the 3D one to read deformation depth.

| Key | Action |
|---|---|
| `b` | re-capture the no-load baseline |
| `r` | toggle slow auto-rotation |
| `t` | toggle bars / surface |
| `0` | reset the viewing angle |
| `+` / `-` | display gain up / down |
| `]` / `[` | display range up / down |
| `s` | save a PNG snapshot |
| `q` or `Esc` | quit |

Drag with the mouse to orbit at any time. It shares `--mode`, `--gain`,
`--range`, `--rot90`, `--flip-ud`, and `--flip-lr` with the 2D tool, and imports
the serial protocol from `live_sensor_2d.py`, so keep the two files together.

### 3.3 `bench_quicklook.py` — Check a Bench Run

Runs captured with `--no-save-images` have no MP4s for the replay tools.
This reads `sensor_raw.csv` and the raw IntelliMESUR export in `<session>/force/`,
plots the tactile response, the load/distance curve and the peak contact map,
and warns about a wrong frame rate, dropped frames, an untared or out-of-calibration
force sensor, and a baseline that did not come back.

```powershell
python bench_quicklook.py <capture session folder>     # writes <session>/bench_quicklook.png
python bench_quicklook.py --selftest
```

### 3.4 `bench_report.py` — Quantify an Aligned Run

After `align_force_presses.py`, this turns the run into numbers and two report
pages in `<session>/report/`: the force / crosshead / tactile timeline, drift of
the tactile response at the same depth from cycle to cycle (absolute and in %,
next to the force), the change during every hold, calibration curves per cycle,
a held-out force prediction with its error by phase and depth, and 2D maps
(depth × cycle, one colour scale) with 3D bars and row/column sums.

```powershell
python bench_report.py <capture session folder>
```

### 3.5 `livesensor_view.py` — Minimal ADC Heatmap

A smaller viewer that plots the raw 8-bit ADC map with no baseline handling.
Use it to confirm the serial link and firmware are alive:

```powershell
python livesensor_view.py COM3
python livesensor_view.py COM3 --raw    # also print per-frame stats
```

## 4. Overhead RealSense D455 (independent depth reference)

The D455 is a cross-check for the multi-view reconstruction on large
deformations, not a finer ground truth: from its 52 cm minimum range it sees
~0.8 mm per pixel on the sponge and measures depth to ~1-2 mm, against 0.16 mm
for the calibrated Blackflys. Mount it rigidly on its own arm 55-60 cm above
the sponge, tilted 20-30 deg so the Mark-10 crosshead does not hide the
indenter, and clear of the Blackfly views. Needs `pyrealsense2` in the capture
venv (`python -m pip install pyrealsense2 --no-deps`, keeps numpy 1.26).

```powershell
python record_d455.py --preview --near                  # aim it at ~0.40 m (shows the valid range)
```

`--near` is the close-range setup: 1280x720 depth plus a disparity shift
(advanced mode) so the D455 measures 0.33-1.0 m instead of 0.49 m and beyond.
At 40 cm that is ~0.2 mm depth noise in theory and ~0.6 mm per pixel; the
background past 1 m drops out. The shift lives in the camera until power-off,
so every run writes it, and a run without `--near` sets it back to 0. With the
one-command capture add `--near` (before the `--`).

**Every capture** -- start the D455 first, stop it last, then attach it:

```powershell
python record_d455.py --output C:\tactile_data\run01    # terminal 1, keep running
python capture_3blackfly_sensor_force.py --port COM3 --cameras 4 --no-open-cell-interpolation --output C:\tactile_data\run01   # terminal 2
#   ... Ctrl+C the capture, then Ctrl+C the D455 ...
python align_d455.py C:\tactile_data\run01\capture_<stamp>
```

`align_d455.py` writes `<session>/d455/` with, for every frame set, the D455
frame closest in time (`depth/<seq>.png` uint16, `color/<seq>.jpg`,
`frames.csv` with the time difference). Synchronisation is by timestamp: the
D455 free-runs at 30 fps, so a match is at most ~17 ms off, which is ~0.1 mm of
crosshead travel at 5 mm/s. Without the RealSense metadata driver on Windows its
timestamps are host arrival times -- the same clock the Blackfly capture logs.
Recordings are rosbag2 `.db3` (librealsense >= 2.57), ~60 MB/s at the default
848x480 depth + colour; budget disk for it on top of the Blackflys.

**Once per mounting** -- put the D455 into the world frame. Print
`calibration/board_d455/charuco_7x10_25mm.pdf` at 100% on A4, glue it flat to a
rigid sheet, measure a square and fix `square_mm` in its `board.json` (the 9 mm
board is too small for the D455 to decode at 60 cm). Record as above with
`record_d455.py --color-size 1280x720`, lay or prop the board still in 5-10
places and tilts (~2 s each, some raised or tilted, not all flat), keep the
BMPs, run `align_d455.py`, then in the calibration venv:

```powershell
python calibration\calibrate_d455.py <session> --install    # -> calibration\d455_world.json
```

It triangulates the board corners with the Blackflys (world frame), solves the
D455 pose from the same corners in its colour image, and then checks the D455
depth at those corners against the triangulated truth -- the D455's own depth
error, in mm. Re-run it whenever the D455 or its arm is touched.
