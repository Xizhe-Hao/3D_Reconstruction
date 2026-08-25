# Tactile Sensor Data Collection

This project records synchronized data from:

- one 16x16 tactile sensor;
- three Teledyne Blackfly S USB3 cameras; and
- one Mark-10 force gauge running IntelliMESUR on its own tablet.

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
3. Connect all three cameras.
4. Open SpinView and confirm that every camera can stream.
5. Exit SpinView before starting Python acquisition.

SpinView and PySpin should not access the same camera simultaneously.

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

- connect the Arduino, sensor, and three cameras;
- verify that D9 reaches all three rising-edge camera trigger inputs;
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
specific camera order, add `--camera-serials CAM0 CAM1 CAM2`.

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

Stage A is complete after all three color MP4 files have been verified and the
terminal prints the new capture session folder.

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

