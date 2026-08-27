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
python run_multimodal_workflow.py --port COM7
```

This command internally uses `capture_3blackfly_sensor_force.py`,
`repair_sensor_session.py`, and `export_multimodal_mp4.py`. Keep these files
beside `run_multimodal_workflow.py`; only the workflow entry point is run
manually.

Replace `COM7` with the Arduino port shown in Device Manager. By default,
captures are stored in the project's `captures` folder.

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

### 2.4 Quick Verification

To check the synchronized videos, sensor, and force data, run:

```powershell
python replay_capture_multimodal.py "path\to\your\capture\session\folder"
```

The replay window shows one preview per camera along the top row, with the
3D sensor surface and the force curve below. Up to four cameras share a single
preview row; beyond that the previews wrap onto additional rows and the window
grows taller. Each preview is labelled with its camera index, so `[3]` is the
fourth camera.

