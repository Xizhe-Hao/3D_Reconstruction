#!/usr/bin/env python3
"""Capture four Blackfly S cameras, M16B sensor data, and force exports.

The cameras are configured and armed first. Python then performs a serial
READY/START/ACK handshake with the Arduino. Every Arduino sensor scan produces
one rising-edge camera trigger followed by one M16B sensor frame. IntelliMESUR
continues to own its serial port; this program watches its automatic CSV export
folder and aligns each exported force run to the tactile stream afterward.

The rig now uses four cameras, but the camera count is not hard-coded: use
--cameras N to capture a different number. The single Arduino D9 trigger line
must reach the OPTOIN input of every connected camera.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import queue
import shutil
import statistics
import struct
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import serial

try:
    import numpy as np
except ImportError:
    np = None

try:
    import PySpin
except ImportError:
    PySpin = None


MATRIX_ROWS = 16
MATRIX_COLS = 16
VALUE_COUNT = MATRIX_ROWS * MATRIX_COLS

BINARY_MAGIC = b"M16B"
BINARY_PROTOCOL_VERSION = 1
BINARY_PAYLOAD_TYPE_ADC_U8 = 1
BINARY_METADATA_FORMAT = "<BBBBII"
BINARY_METADATA_BYTES = struct.calcsize(BINARY_METADATA_FORMAT)
BINARY_FRAME_BODY_BYTES = BINARY_METADATA_BYTES + VALUE_COUNT
BINARY_FRAME_REST_BYTES = BINARY_FRAME_BODY_BYTES + 2
BINARY_FRAME_BYTES = len(BINARY_MAGIC) + BINARY_FRAME_REST_BYTES

ARDUINO_READY = b"M16_READY"
ARDUINO_ACK = b"M16_ACK"
COMMAND_START = b"M16_START\n"
COMMAND_PING = b"M16_PING\n"
COMMAND_STOP = b"M16_STOP\n"

DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 500_000
DEFAULT_CAMERA_COUNT = 4
COLOR_PIXEL_FORMAT_PREFIXES = (
    "Bayer",
    "RGB",
    "BGR",
    "YUV",
    "YCbCr",
)
ADC8_MAX_VALUE = 255.0
DEFAULT_ADC_REFERENCE_V = 5.0
DEFAULT_SENSOR_DRIVE_V = 5.0

# BEGIN OPTIONAL OPEN-CONTACT REPAIR
# These thresholds were added for the earlier sensor whose bad contacts
# appeared as abnormally high sensor voltages. A reliable replacement sensor
# may not need this spatial repair; raw acquisition and frame-loss handling
# are independent of these settings.
DEFAULT_OPEN_CELL_VOLTAGE_V = 4.8
DEFAULT_POINT_OPEN_VOLTAGE_V = 4.5
DEFAULT_POINT_NEIGHBOR_DELTA_V = 0.75
DEFAULT_LINE_OPEN_VOLTAGE_V = 4.5
DEFAULT_LINE_OPEN_FRACTION = 0.75
DEFAULT_LINE_NEIGHBOR_DELTA_V = 0.75
DEFAULT_INTERPOLATION_RADIUS = 3
DEFAULT_INTERPOLATION_MIN_NEIGHBORS = 3
# END OPTIONAL OPEN-CONTACT REPAIR

DEFAULT_FORCE_EXPORT_DIR = Path(
    r"C:\Mark-10 Software\IntelliMESUR\Export Data"
)
DEFAULT_FORCE_EXPORT_DELAY_S = 2.45
DEFAULT_FORCE_SEARCH_WINDOW_S = 5.0
DEFAULT_FORCE_POLL_INTERVAL_S = 0.05
DEFAULT_FORCE_STABLE_INTERVAL_S = 0.25
DEFAULT_FORCE_EXPORT_WAIT_S = 3.0
DEFAULT_FORCE_MIN_CORRELATION = 0.25
DEFAULT_SERIAL_RX_BUFFER_BYTES = 262_144
DEFAULT_SENSOR_QUEUE_FRAMES = 2048
DEFAULT_CAPTURE_OUTPUT = (
    Path(r"E:\tactile_sensor_captures")
    if Path("E:\\").exists()
    else Path(__file__).resolve().parent / "captures"
)
DEFAULT_MINIMUM_FREE_GB = 20.0


@dataclass
class SensorStats:
    frames: int = 0
    bad_frames: int = 0
    duplicate_frames: int = 0
    dropped_frames: int = 0
    last_frame_index: int = -1
    frames_with_open_cells: int = 0
    open_cell_samples: int = 0
    interpolated_cell_samples: int = 0
    unrepaired_cell_samples: int = 0
    detected_bad_rows: int = 0
    detected_bad_columns: int = 0
    reader_queue_peak: int = 0


@dataclass
class CameraStats:
    camera_index: int
    serial: str
    complete_images: int = 0
    saved_images: int = 0
    incomplete_images: int = 0
    dropped_images: int = 0
    timeouts: int = 0
    first_frame_id: int = -1
    last_frame_id: int = -1
    last_sequence_index: int = -1
    source_pixel_format: str = ""
    saved_pixel_format: str = ""


@dataclass
class ForceExportRecord:
    source_path: str
    archived_path: str
    detected_ns: int
    stable_ns: int
    source_creation_ns: int
    source_write_ns: int
    source_size: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Synchronously capture four Blackfly S cameras and a 16x16 "
            "M16B tactile sensor stream."
        )
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument(
        "--serial-rx-buffer",
        type=int,
        default=DEFAULT_SERIAL_RX_BUFFER_BYTES,
        help=(
            "Windows COM receive buffer in bytes. A large buffer prevents "
            "M16B loss while camera threads are writing images."
        ),
    )
    parser.add_argument(
        "--sensor-queue-frames",
        type=int,
        default=DEFAULT_SENSOR_QUEUE_FRAMES,
        help=(
            "Maximum decoded sensor frames buffered between the dedicated "
            "serial reader and CSV/correction worker."
        ),
    )
    parser.add_argument(
        "--trigger-source",
        default="Line0",
        help="Camera trigger input line, normally Line0 for OPTOIN.",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=DEFAULT_CAMERA_COUNT,
        help=(
            "Number of Blackfly S cameras that must be present. The rig "
            f"currently uses {DEFAULT_CAMERA_COUNT}; every camera needs the "
            "Arduino D9 trigger pulse on its OPTOIN input."
        ),
    )
    parser.add_argument(
        "--camera-serials",
        nargs="+",
        metavar="SERIAL",
        help=(
            "Optional camera serial numbers in the desired output order. "
            "Supply exactly --cameras values."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after this many sensor frames; 0 runs until Ctrl+C.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CAPTURE_OUTPUT,
        help=(
            "Capture root. Defaults to E:\\tactile_sensor_captures on this "
            "workstation because full-resolution capture needs sustained "
            "write bandwidth."
        ),
    )
    parser.add_argument(
        "--minimum-free-gb",
        type=float,
        default=DEFAULT_MINIMUM_FREE_GB,
        help=(
            "Refuse image capture when the output drive has less free space "
            "than this value; use 0 to disable the check."
        ),
    )
    parser.add_argument(
        "--image-format",
        choices=("bmp", "png", "tiff", "jpg", "raw"),
        default="bmp",
        help=(
            "Saved camera image format. Raw Bayer BMP is the default because "
            "it keeps up with 24 fps and preserves color for offline decoding."
        ),
    )
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Select an available Bayer/RGB camera format (default: enabled). "
            "Raw Bayer frames look grayscale until exported to color video. "
            "Use --no-color for true grayscale capture."
        ),
    )
    parser.add_argument(
        "--demosaic-on-save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Convert Bayer to BGR before saving each image. This is disabled "
            "by default because conversion plus compression cannot sustain "
            "three 2048x1536 cameras at 24 fps."
        ),
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Receive and log images without writing image files.",
    )
    parser.add_argument(
        "--exposure-us",
        type=float,
        help="Optional fixed exposure time. By default the current setting is kept.",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        help="Optional fixed camera gain. By default the current setting is kept.",
    )
    parser.add_argument("--stream-buffers", type=int, default=64)
    parser.add_argument("--image-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=5.0,
        help="Abort if the sensor or any camera produces no frame for this long.",
    )
    parser.add_argument("--handshake-timeout", type=float, default=15.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.5)
    parser.add_argument("--drain-timeout", type=float, default=5.0)
    parser.add_argument(
        "--adc-ref",
        type=float,
        default=DEFAULT_ADC_REFERENCE_V,
        help="ADC reference voltage used to interpret the received ADC8 values.",
    )
    parser.add_argument(
        "--vdrive",
        type=float,
        default=DEFAULT_SENSOR_DRIVE_V,
        help="Sensor drive voltage used by V_sensor = Vdrive - V_adc.",
    )

    # BEGIN OPTIONAL OPEN-CONTACT REPAIR CLI
    # Use --no-open-cell-interpolation to disable this during live capture.
    parser.add_argument(
        "--open-cell-voltage",
        type=float,
        default=DEFAULT_OPEN_CELL_VOLTAGE_V,
        help=(
            "Treat sensor voltages at or above this value as open/contact "
            "faults and spatially interpolate them."
        ),
    )
    parser.add_argument(
        "--point-open-voltage",
        type=float,
        default=DEFAULT_POINT_OPEN_VOLTAGE_V,
        help=(
            "Minimum sensor voltage for local-neighborhood open-contact "
            "detection."
        ),
    )
    parser.add_argument(
        "--point-neighbor-delta",
        type=float,
        default=DEFAULT_POINT_NEIGHBOR_DELTA_V,
        help=(
            "Required voltage excess over the surrounding 3x3 median before "
            "a local high-voltage cell is repaired."
        ),
    )
    parser.add_argument(
        "--line-open-voltage",
        type=float,
        default=DEFAULT_LINE_OPEN_VOLTAGE_V,
        help=(
            "Minimum sensor voltage used when detecting a mostly-open full "
            "row or column."
        ),
    )
    parser.add_argument(
        "--line-open-fraction",
        type=float,
        default=DEFAULT_LINE_OPEN_FRACTION,
        help="Required fraction of high-voltage cells in a bad row or column.",
    )
    parser.add_argument(
        "--line-neighbor-delta",
        type=float,
        default=DEFAULT_LINE_NEIGHBOR_DELTA_V,
        help=(
            "Required median voltage excess over adjacent parallel lines "
            "before a full row or column is repaired."
        ),
    )
    parser.add_argument(
        "--interpolation-radius",
        type=int,
        default=DEFAULT_INTERPOLATION_RADIUS,
        help="Maximum matrix-cell radius used to find normal neighbors.",
    )
    parser.add_argument(
        "--interpolation-min-neighbors",
        type=int,
        default=DEFAULT_INTERPOLATION_MIN_NEIGHBORS,
        help=(
            "Stop expanding the search after finding at least this many "
            "normal neighbors."
        ),
    )
    parser.add_argument(
        "--no-open-cell-interpolation",
        action="store_true",
        help="Disable generation of sensor_corrected.csv and anomaly logging.",
    )
    # END OPTIONAL OPEN-CONTACT REPAIR CLI

    parser.add_argument(
        "--force-export-dir",
        type=Path,
        default=DEFAULT_FORCE_EXPORT_DIR,
        help=(
            "IntelliMESUR automatic Export Run Data folder. COM7 remains "
            "owned by IntelliMESUR; this program only watches CSV files."
        ),
    )
    parser.add_argument(
        "--no-force-export-monitor",
        action="store_true",
        help="Disable IntelliMESUR export monitoring and force alignment.",
    )
    parser.add_argument(
        "--force-export-delay",
        type=float,
        default=DEFAULT_FORCE_EXPORT_DELAY_S,
        help=(
            "Approximate seconds from the end of a force run to automatic "
            "CSV creation; used only for coarse alignment."
        ),
    )
    parser.add_argument(
        "--force-search-window",
        type=float,
        default=DEFAULT_FORCE_SEARCH_WINDOW_S,
        help="Seconds searched on either side of the coarse force start time.",
    )
    parser.add_argument(
        "--force-min-correlation",
        type=float,
        default=DEFAULT_FORCE_MIN_CORRELATION,
        help="Minimum absolute correlation reported as medium confidence.",
    )
    parser.add_argument(
        "--force-export-wait",
        type=float,
        default=DEFAULT_FORCE_EXPORT_WAIT_S,
        help=(
            "Seconds to keep watching for IntelliMESUR's final automatic "
            "export after capture is stopped."
        ),
    )
    parser.add_argument(
        "--align-only",
        nargs=2,
        type=Path,
        metavar=("SESSION_DIR", "FORCE_CSV"),
        help=(
            "Skip camera capture and align one existing automatic force "
            "export to an existing capture session."
        ),
    )
    args = parser.parse_args()

    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.cameras < 1:
        parser.error("--cameras must be at least 1")
    if args.camera_serials and len(args.camera_serials) != args.cameras:
        parser.error(
            f"--camera-serials expects {args.cameras} serial numbers, "
            f"received {len(args.camera_serials)}"
        )
    if args.camera_serials and len(set(args.camera_serials)) != args.cameras:
        parser.error("--camera-serials must not repeat a serial number")
    if args.minimum_free_gb < 0:
        parser.error("--minimum-free-gb must be non-negative")
    if (
        args.color
        and not args.demosaic_on_save
        and args.image_format not in ("bmp", "tiff", "raw")
    ):
        parser.error(
            "Raw Bayer capture requires bmp, tiff, or raw. Use BMP for speed, "
            "or add --demosaic-on-save before selecting jpg/png."
        )
    if args.serial_rx_buffer < 4096:
        parser.error("--serial-rx-buffer must be at least 4096 bytes")
    if args.sensor_queue_frames < 16:
        parser.error("--sensor-queue-frames must be at least 16")
    if args.stream_buffers < 2:
        parser.error("--stream-buffers must be at least 2")
    if args.image_timeout_ms < 1:
        parser.error("--image-timeout-ms must be positive")
    if (
        args.stall_timeout <= 0
        or args.handshake_timeout <= 0
        or args.heartbeat_interval <= 0
    ):
        parser.error("Handshake, heartbeat, and stall timing must be positive")
    if args.adc_ref <= 0 or args.vdrive <= 0:
        parser.error("--adc-ref and --vdrive must be positive")
    if not 0 < args.open_cell_voltage < args.vdrive:
        parser.error("--open-cell-voltage must satisfy 0 < value < --vdrive")
    if not 0 < args.point_open_voltage < args.vdrive:
        parser.error("--point-open-voltage must satisfy 0 < value < --vdrive")
    if args.point_neighbor_delta < 0:
        parser.error("--point-neighbor-delta must be non-negative")
    if not 0 < args.line_open_voltage < args.vdrive:
        parser.error("--line-open-voltage must satisfy 0 < value < --vdrive")
    if not 0 < args.line_open_fraction <= 1:
        parser.error("--line-open-fraction must satisfy 0 < value <= 1")
    if args.line_neighbor_delta < 0:
        parser.error("--line-neighbor-delta must be non-negative")
    if args.interpolation_radius < 1:
        parser.error("--interpolation-radius must be at least 1")
    if args.interpolation_min_neighbors < 1:
        parser.error("--interpolation-min-neighbors must be at least 1")
    if args.force_export_delay < 0:
        parser.error("--force-export-delay must be non-negative")
    if args.force_search_window < 0:
        parser.error("--force-search-window must be non-negative")
    if not 0 <= args.force_min_correlation <= 1:
        parser.error("--force-min-correlation must be between 0 and 1")
    if args.force_export_wait < 0:
        parser.error("--force-export-wait must be non-negative")
    return args


def is_available_readable(node):
    return PySpin.IsAvailable(node) and PySpin.IsReadable(node)


def is_available_writable(node):
    return PySpin.IsAvailable(node) and PySpin.IsWritable(node)


def set_enum(nodemap, node_name, entry_name, required=True):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not is_available_writable(node):
        if required:
            raise RuntimeError(f"Camera enum {node_name} is not writable")
        return False

    entry = node.GetEntryByName(entry_name)
    if not is_available_readable(entry):
        if required:
            raise RuntimeError(
                f"Camera enum {node_name} has no readable {entry_name} entry"
            )
        return False

    node.SetIntValue(entry.GetValue())
    return True


def set_float(nodemap, node_name, value, required=True):
    node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
    if not is_available_writable(node):
        if required:
            raise RuntimeError(f"Camera float {node_name} is not writable")
        return None

    clamped = min(max(float(value), node.GetMin()), node.GetMax())
    increment = node.GetInc()
    if increment > 0:
        clamped = node.GetMin() + round(
            (clamped - node.GetMin()) / increment
        ) * increment
    node.SetValue(clamped)
    return clamped


def set_integer(nodemap, node_name, value, required=True):
    node = PySpin.CIntegerPtr(nodemap.GetNode(node_name))
    if not is_available_writable(node):
        if required:
            raise RuntimeError(f"Camera integer {node_name} is not writable")
        return None

    clamped = min(max(int(value), node.GetMin()), node.GetMax())
    increment = node.GetInc()
    if increment > 1:
        clamped = node.GetMin() + (
            (clamped - node.GetMin()) // increment
        ) * increment
    node.SetValue(clamped)
    return clamped


def read_camera_serial(camera):
    nodemap = camera.GetTLDeviceNodeMap()
    node = PySpin.CStringPtr(nodemap.GetNode("DeviceSerialNumber"))
    if not is_available_readable(node):
        raise RuntimeError("Unable to read a camera serial number")
    return str(node.GetValue())


def read_enum_symbolic(nodemap, node_name):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not is_available_readable(node):
        return "unknown"
    entry = node.GetCurrentEntry()
    if not is_available_readable(entry):
        return "unknown"
    return str(entry.GetSymbolic())


def available_enum_symbols(nodemap, node_name):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not is_available_writable(node):
        return []
    symbols = []
    for raw_entry in node.GetEntries():
        entry = PySpin.CEnumEntryPtr(raw_entry)
        if is_available_readable(entry):
            symbols.append(str(entry.GetSymbolic()))
    return symbols


def select_color_pixel_format(nodemap):
    available = available_enum_symbols(nodemap, "PixelFormat")
    color_formats = [
        name
        for name in available
        if name.startswith(COLOR_PIXEL_FORMAT_PREFIXES)
    ]
    if not color_formats:
        raise RuntimeError(
            "Camera exposes no Bayer/RGB/YUV pixel format; it may be a "
            "monochrome model"
        )

    bayer8 = [
        name
        for name in color_formats
        if name.startswith("Bayer") and name.endswith("8")
    ]
    preferred = (
        bayer8
        + [
            name
            for name in ("RGB8Packed", "BGR8")
            if name in color_formats
        ]
        + color_formats
    )
    selected = preferred[0]
    set_enum(nodemap, "PixelFormat", selected)
    return selected


def configure_camera(camera, args):
    nodemap = camera.GetNodeMap()
    stream_nodemap = camera.GetTLStreamNodeMap()

    set_enum(nodemap, "TriggerMode", "Off")
    set_enum(nodemap, "TriggerSelector", "FrameStart")
    set_enum(nodemap, "TriggerSource", args.trigger_source)
    set_enum(nodemap, "TriggerActivation", "RisingEdge")
    set_enum(nodemap, "ExposureMode", "Timed")
    set_enum(nodemap, "AcquisitionMode", "Continuous")
    if args.color:
        pixel_format = select_color_pixel_format(nodemap)
        saved_format = "BGR8" if args.demosaic_on_save else pixel_format
        print(
            f"  source pixel format: {pixel_format} -> saved {saved_format}"
        )
    else:
        pixel_format = read_enum_symbolic(nodemap, "PixelFormat")
        print(f"  pixel format: {pixel_format}")

    if args.exposure_us is not None:
        set_enum(nodemap, "ExposureAuto", "Off")
        actual = set_float(nodemap, "ExposureTime", args.exposure_us)
        print(f"  exposure: {actual:.1f} us")

    if args.gain_db is not None:
        set_enum(nodemap, "GainAuto", "Off")
        actual = set_float(nodemap, "Gain", args.gain_db)
        print(f"  gain: {actual:.2f} dB")

    set_enum(
        stream_nodemap,
        "StreamBufferCountMode",
        "Manual",
        required=False,
    )
    actual_buffers = set_integer(
        stream_nodemap,
        "StreamBufferCountManual",
        args.stream_buffers,
        required=False,
    )
    set_enum(
        stream_nodemap,
        "StreamBufferHandlingMode",
        "OldestFirst",
        required=False,
    )
    if actual_buffers is not None:
        print(f"  stream buffers: {actual_buffers}")

    set_enum(nodemap, "TriggerMode", "On")
    return pixel_format


def trigger_armed(camera):
    node = PySpin.CBooleanPtr(camera.GetNodeMap().GetNode("TriggerArmed"))
    if not is_available_readable(node):
        return None
    return bool(node.GetValue())


def wait_for_cameras_armed(cameras, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        states = [trigger_armed(camera) for camera in cameras]
        known_states = [state for state in states if state is not None]
        if not known_states:
            time.sleep(0.2)
            return
        if all(known_states) and len(known_states) == len(cameras):
            return
        time.sleep(0.01)
    raise TimeoutError("Not all cameras reported TriggerArmed before timeout")


def initialize_cameras(args):
    system = PySpin.System.GetInstance()
    camera_list = system.GetCameras()
    count = camera_list.GetSize()
    if count != args.cameras:
        camera_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError(
            f"Expected {args.cameras} cameras, found {count}. Check the USB3 "
            "connections, or pass --cameras to match the rig."
        )

    cameras_by_serial = {}
    for index in range(count):
        camera = camera_list.GetByIndex(index)
        cameras_by_serial[read_camera_serial(camera)] = camera

    if args.camera_serials:
        missing = [
            serial_number
            for serial_number in args.camera_serials
            if serial_number not in cameras_by_serial
        ]
        if missing:
            camera = None
            cameras_by_serial.clear()
            gc.collect()
            try:
                camera_list.Clear()
            finally:
                system.ReleaseInstance()
            raise RuntimeError(f"Requested camera serials not found: {missing}")
        ordered_serials = list(args.camera_serials)
    else:
        ordered_serials = sorted(cameras_by_serial)

    cameras = [cameras_by_serial[number] for number in ordered_serials]
    initialized = []
    pixel_formats = []
    try:
        for index, (serial_number, camera) in enumerate(
            zip(ordered_serials, cameras)
        ):
            print(f"Configuring camera {index}: {serial_number}")
            camera.Init()
            initialized.append(camera)
            pixel_formats.append(configure_camera(camera, args))
    except Exception as configuration_error:
        for camera in reversed(initialized):
            try:
                camera.DeInit()
            except Exception:
                pass

        camera = None
        initialized.clear()
        cameras.clear()
        cameras_by_serial.clear()
        pixel_formats.clear()
        gc.collect()
        cleanup_errors = []
        try:
            camera_list.Clear()
        except Exception as exc:
            cleanup_errors.append(f"camera list: {exc}")
        try:
            system.ReleaseInstance()
        except Exception as exc:
            cleanup_errors.append(f"system: {exc}")
        detail = (
            f"; cleanup warnings: {', '.join(cleanup_errors)}"
            if cleanup_errors
            else ""
        )
        raise RuntimeError(
            f"Camera configuration failed: {configuration_error}{detail}"
        ) from configuration_error

    return system, camera_list, cameras, ordered_serials, pixel_formats


def write_serial_command(serial_port, write_lock, command):
    with write_lock:
        serial_port.write(command)
        serial_port.flush()


def wait_for_arduino_handshake(serial_port, write_lock, timeout):
    deadline = time.monotonic() + timeout
    print("Waiting for Arduino READY...")

    while time.monotonic() < deadline:
        line = serial_port.readline().strip()
        if line != ARDUINO_READY:
            continue

        print("Arduino READY received; sending START...")
        write_serial_command(serial_port, write_lock, COMMAND_START)

        while time.monotonic() < deadline:
            response = serial_port.readline().strip()
            if response == ARDUINO_ACK:
                print("Arduino ACK received; synchronized capture started.")
                return
            if response == ARDUINO_READY:
                write_serial_command(serial_port, write_lock, COMMAND_START)

    raise TimeoutError("Arduino handshake timed out")


def heartbeat_worker(serial_port, write_lock, stop_event, interval):
    while not stop_event.wait(interval):
        try:
            write_serial_command(serial_port, write_lock, COMMAND_PING)
        except Exception:
            return


def decode_m16b_frame(frame):
    if len(frame) != BINARY_FRAME_BYTES or not frame.startswith(BINARY_MAGIC):
        return False
    body_start = len(BINARY_MAGIC)
    body_stop = body_start + BINARY_FRAME_BODY_BYTES
    body = frame[body_start:body_stop]
    received_checksum = struct.unpack_from(
        "<H", frame, body_stop
    )[0]
    if (sum(body) & 0xFFFF) != received_checksum:
        return False

    (
        version,
        rows,
        columns,
        payload_type,
        frame_index,
        device_millis,
    ) = struct.unpack_from(BINARY_METADATA_FORMAT, body, 0)
    if (
        version != BINARY_PROTOCOL_VERSION
        or rows != MATRIX_ROWS
        or columns != MATRIX_COLS
        or payload_type != BINARY_PAYLOAD_TYPE_ADC_U8
    ):
        return False

    payload = body[BINARY_METADATA_BYTES:]
    return frame_index, device_millis, payload


class M16BStreamReader:
    """Chunked, resynchronizing serial reader for the M16B stream."""

    def __init__(self, serial_port):
        self.serial_port = serial_port
        self.buffer = bytearray()

    def _read_more(self):
        waiting = max(1, int(self.serial_port.in_waiting))
        chunk = self.serial_port.read(min(waiting, 65536))
        if chunk:
            self.buffer.extend(chunk)
            return True
        return False

    def read_frame(self, stop_event):
        while not stop_event.is_set():
            magic_at = self.buffer.find(BINARY_MAGIC)
            if magic_at < 0:
                keep = len(BINARY_MAGIC) - 1
                if len(self.buffer) > keep:
                    del self.buffer[:-keep]
                self._read_more()
                continue
            if magic_at:
                del self.buffer[:magic_at]
            if len(self.buffer) < BINARY_FRAME_BYTES:
                self._read_more()
                continue

            candidate = bytes(self.buffer[:BINARY_FRAME_BYTES])
            decoded = decode_m16b_frame(candidate)
            if decoded is False:
                # Advance by one byte, not one whole candidate. This preserves
                # a valid frame beginning inside data following a corrupt one.
                del self.buffer[0]
                return False

            del self.buffer[:BINARY_FRAME_BYTES]
            return decoded
        return None


# BEGIN OPTIONAL OPEN-CONTACT REPAIR ALGORITHM
# This block detects isolated high-voltage cells and mostly high-voltage rows
# or columns, then replaces them from nearby normal cells.
def detect_bad_lines(
    adc8_values,
    adc_reference_v,
    sensor_drive_v,
    line_open_voltage_v,
    line_open_fraction,
    line_neighbor_delta_v,
):
    voltages = [
        sensor_drive_v - value * adc_reference_v / ADC8_MAX_VALUE
        for value in adc8_values
    ]
    matrix = [
        voltages[row * MATRIX_COLS:(row + 1) * MATRIX_COLS]
        for row in range(MATRIX_ROWS)
    ]
    required_high = max(
        1,
        int(line_open_fraction * max(MATRIX_ROWS, MATRIX_COLS) + 0.999999),
    )

    row_medians = [sorted(values)[len(values) // 2] for values in matrix]
    column_values = [
        [matrix[row][column] for row in range(MATRIX_ROWS)]
        for column in range(MATRIX_COLS)
    ]
    column_medians = [
        sorted(values)[len(values) // 2] for values in column_values
    ]

    bad_rows = []
    for row, values in enumerate(matrix):
        neighbor_rows = [
            candidate
            for candidate in (row - 1, row + 1)
            if 0 <= candidate < MATRIX_ROWS
        ]
        neighbor_level = sum(
            row_medians[candidate] for candidate in neighbor_rows
        ) / len(neighbor_rows)
        high_count = sum(
            value >= line_open_voltage_v for value in values
        )
        if (
            high_count >= required_high
            and row_medians[row] - neighbor_level >= line_neighbor_delta_v
        ):
            bad_rows.append(row)

    bad_columns = []
    for column, values in enumerate(column_values):
        neighbor_columns = [
            candidate
            for candidate in (column - 1, column + 1)
            if 0 <= candidate < MATRIX_COLS
        ]
        neighbor_level = sum(
            column_medians[candidate] for candidate in neighbor_columns
        ) / len(neighbor_columns)
        high_count = sum(
            value >= line_open_voltage_v for value in values
        )
        if (
            high_count >= required_high
            and column_medians[column] - neighbor_level
            >= line_neighbor_delta_v
        ):
            bad_columns.append(column)

    return tuple(bad_rows), tuple(bad_columns)


def interpolate_open_cells(
    adc8_values,
    adc_reference_v,
    sensor_drive_v,
    open_cell_voltage_v,
    point_open_voltage_v,
    point_neighbor_delta_v,
    line_open_voltage_v,
    line_open_fraction,
    line_neighbor_delta_v,
    maximum_radius,
    minimum_neighbors,
):
    """Replace isolated and full-line open-contact cells."""
    bad_adc8_limit = (
        (sensor_drive_v - open_cell_voltage_v)
        * ADC8_MAX_VALUE
        / adc_reference_v
    )
    bad_mask = [value <= bad_adc8_limit for value in adc8_values]
    voltages = [
        sensor_drive_v - value * adc_reference_v / ADC8_MAX_VALUE
        for value in adc8_values
    ]
    for row in range(MATRIX_ROWS):
        for column in range(MATRIX_COLS):
            index = row * MATRIX_COLS + column
            if bad_mask[index] or voltages[index] < point_open_voltage_v:
                continue
            neighbors = []
            for neighbor_row in range(
                max(0, row - 1),
                min(MATRIX_ROWS, row + 2),
            ):
                for neighbor_column in range(
                    max(0, column - 1),
                    min(MATRIX_COLS, column + 2),
                ):
                    if neighbor_row == row and neighbor_column == column:
                        continue
                    neighbors.append(
                        voltages[
                            neighbor_row * MATRIX_COLS + neighbor_column
                        ]
                    )
            if (
                neighbors
                and voltages[index] - statistics.median(neighbors)
                >= point_neighbor_delta_v
            ):
                bad_mask[index] = True

    bad_rows, bad_columns = detect_bad_lines(
        adc8_values,
        adc_reference_v,
        sensor_drive_v,
        line_open_voltage_v,
        line_open_fraction,
        line_neighbor_delta_v,
    )
    for row in bad_rows:
        for column in range(MATRIX_COLS):
            bad_mask[row * MATRIX_COLS + column] = True
    for column in bad_columns:
        for row in range(MATRIX_ROWS):
            bad_mask[row * MATRIX_COLS + column] = True

    bad_indices = [index for index, is_bad in enumerate(bad_mask) if is_bad]
    if not bad_indices:
        return list(adc8_values), (), (), (), ()

    corrected = list(adc8_values)
    unrepaired_indices = []

    for bad_index in bad_indices:
        center_row, center_column = divmod(bad_index, MATRIX_COLS)
        neighbors = []

        for radius in range(1, maximum_radius + 1):
            neighbors = []
            row_start = max(0, center_row - radius)
            row_stop = min(MATRIX_ROWS, center_row + radius + 1)
            column_start = max(0, center_column - radius)
            column_stop = min(MATRIX_COLS, center_column + radius + 1)

            for row in range(row_start, row_stop):
                for column in range(column_start, column_stop):
                    neighbor_index = row * MATRIX_COLS + column
                    if neighbor_index == bad_index or bad_mask[neighbor_index]:
                        continue
                    row_delta = row - center_row
                    column_delta = column - center_column
                    distance_squared = (
                        row_delta * row_delta + column_delta * column_delta
                    )
                    # Inverse-distance-squared weighting favors the closest
                    # normal cells while still using all valid nearby cells.
                    neighbors.append(
                        (adc8_values[neighbor_index], 1.0 / distance_squared)
                    )

            if len(neighbors) >= minimum_neighbors:
                break

        if neighbors:
            weighted_sum = sum(value * weight for value, weight in neighbors)
            total_weight = sum(weight for _, weight in neighbors)
            corrected[bad_index] = max(
                0,
                min(255, round(weighted_sum / total_weight)),
            )
        else:
            unrepaired_indices.append(bad_index)

    return (
        corrected,
        tuple(bad_indices),
        tuple(unrepaired_indices),
        bad_rows,
        bad_columns,
    )
# END OPTIONAL OPEN-CONTACT REPAIR ALGORITHM


def sensor_serial_reader_worker(
    serial_port,
    frame_queue,
    reader_done,
    stats,
    stats_lock,
    last_activity,
    stop_event,
    fatal_errors,
):
    reader = M16BStreamReader(serial_port)
    last_queued_frame_index = -1
    try:
        while not stop_event.is_set():
            result = reader.read_frame(stop_event)
            if result is None:
                continue
            if result is False:
                with stats_lock:
                    stats.bad_frames += 1
                continue

            frame_index, device_millis, payload = result
            last_activity[0] = time.monotonic()
            if frame_index <= last_queued_frame_index:
                with stats_lock:
                    stats.duplicate_frames += 1
                continue
            last_queued_frame_index = frame_index
            item = (
                time.time_ns(),
                frame_index,
                device_millis,
                bytes(payload),
            )
            while not stop_event.is_set():
                try:
                    frame_queue.put(item, timeout=0.1)
                    with stats_lock:
                        stats.reader_queue_peak = max(
                            stats.reader_queue_peak,
                            frame_queue.qsize(),
                        )
                    break
                except queue.Full:
                    continue
    except Exception as exc:
        if not stop_event.is_set():
            fatal_errors.put(f"Sensor serial reader failed: {exc}")
    finally:
        reader_done.set()


def sensor_writer_worker(
    frame_queue,
    reader_done,
    session_dir,
    stats,
    stats_lock,
    stop_event,
    fatal_errors,
    correction_enabled,
    adc_reference_v,
    sensor_drive_v,
    open_cell_voltage_v,
    point_open_voltage_v,
    point_neighbor_delta_v,
    line_open_voltage_v,
    line_open_fraction,
    line_neighbor_delta_v,
    interpolation_radius,
    interpolation_min_neighbors,
):
    raw_output_path = session_dir / "sensor_raw.csv"

    # OPTIONAL OPEN-CONTACT REPAIR OUTPUTS:
    # sensor_raw.csv is always untouched; these two files exist only when the
    # optional spatial correction is enabled.
    corrected_output_path = session_dir / "sensor_corrected.csv"
    anomaly_output_path = session_dir / "sensor_anomalies.csv"
    cell_names = [
        f"r{row + 1}c{column + 1}_adc8"
        for row in range(MATRIX_ROWS)
        for column in range(MATRIX_COLS)
    ]
    common_header = ["host_received_ns", "frame_index", "device_millis"]

    try:
        with raw_output_path.open(
            "w", newline="", encoding="utf-8"
        ) as raw_handle:
            raw_writer = csv.writer(raw_handle)
            raw_writer.writerow(common_header + cell_names)

            corrected_handle = None
            anomaly_handle = None
            corrected_writer = None
            anomaly_writer = None
            if correction_enabled:
                corrected_handle = corrected_output_path.open(
                    "w", newline="", encoding="utf-8"
                )
                anomaly_handle = anomaly_output_path.open(
                    "w", newline="", encoding="utf-8"
                )
                corrected_writer = csv.writer(corrected_handle)
                corrected_writer.writerow(common_header + cell_names)
                anomaly_writer = csv.writer(anomaly_handle)
                anomaly_writer.writerow(
                    common_header
                    + [
                        "open_cell_count",
                        "interpolated_cell_count",
                        "unrepaired_cell_count",
                        "bad_rows",
                        "bad_columns",
                        "open_cells",
                        "unrepaired_cells",
                    ]
                )

            try:
                while not (reader_done.is_set() and frame_queue.empty()):
                    try:
                        (
                            host_received_ns,
                            frame_index,
                            device_millis,
                            payload,
                        ) = frame_queue.get(timeout=0.1)
                    except queue.Empty:
                        if stop_event.is_set() and reader_done.is_set():
                            break
                        continue
                    adc8_values = list(payload)
                    common_values = [
                        host_received_ns,
                        frame_index,
                        device_millis,
                    ]
                    raw_writer.writerow(common_values + adc8_values)

                    bad_indices = ()
                    unrepaired_indices = ()
                    bad_rows = ()
                    bad_columns = ()
                    if correction_enabled:
                        # OPTIONAL OPEN-CONTACT REPAIR: this is the live call
                        # that substitutes detected high-voltage cells.
                        (
                            corrected_values,
                            bad_indices,
                            unrepaired_indices,
                            bad_rows,
                            bad_columns,
                        ) = interpolate_open_cells(
                            adc8_values,
                            adc_reference_v,
                            sensor_drive_v,
                            open_cell_voltage_v,
                            point_open_voltage_v,
                            point_neighbor_delta_v,
                            line_open_voltage_v,
                            line_open_fraction,
                            line_neighbor_delta_v,
                            interpolation_radius,
                            interpolation_min_neighbors,
                        )
                        corrected_writer.writerow(
                            common_values + corrected_values
                        )
                        if bad_indices:
                            anomaly_writer.writerow(
                                common_values
                                + [
                                    len(bad_indices),
                                    len(bad_indices)
                                    - len(unrepaired_indices),
                                    len(unrepaired_indices),
                                    ";".join(
                                        f"R{row + 1}" for row in bad_rows
                                    ),
                                    ";".join(
                                        f"C{column + 1}"
                                        for column in bad_columns
                                    ),
                                    ";".join(
                                        cell_names[index]
                                        for index in bad_indices
                                    ),
                                    ";".join(
                                        cell_names[index]
                                        for index in unrepaired_indices
                                    ),
                                ]
                            )

                    with stats_lock:
                        if (
                            stats.last_frame_index >= 0
                            and frame_index > stats.last_frame_index + 1
                        ):
                            stats.dropped_frames += (
                                frame_index - stats.last_frame_index - 1
                            )
                        stats.frames += 1
                        stats.last_frame_index = frame_index
                        if bad_indices:
                            stats.frames_with_open_cells += 1
                            stats.open_cell_samples += len(bad_indices)
                            stats.interpolated_cell_samples += (
                                len(bad_indices) - len(unrepaired_indices)
                            )
                            stats.unrepaired_cell_samples += len(
                                unrepaired_indices
                            )
                            stats.detected_bad_rows += len(bad_rows)
                            stats.detected_bad_columns += len(bad_columns)
                        frame_count = stats.frames
                    frame_queue.task_done()

                    if frame_count % 30 == 0:
                        raw_handle.flush()
                        if corrected_handle is not None:
                            corrected_handle.flush()
                        if anomaly_handle is not None:
                            anomaly_handle.flush()
            finally:
                if corrected_handle is not None:
                    corrected_handle.close()
                if anomaly_handle is not None:
                    anomaly_handle.close()
    except Exception as exc:
        if not stop_event.is_set():
            fatal_errors.put(f"Sensor writer failed: {exc}")


def is_timeout_exception(exception):
    error_code = getattr(exception, "errorcode", None)
    if error_code == PySpin.SPINNAKER_ERR_TIMEOUT:
        return True

    message = str(exception).lower()
    return "timeout" in message or "[-1011]" in message


def camera_worker(
    camera,
    stats,
    session_dir,
    image_format,
    save_images,
    demosaic_on_save,
    image_timeout_ms,
    stats_lock,
    last_activity,
    stop_event,
    fatal_errors,
):
    camera_dir = session_dir / f"camera_{stats.camera_index}_{stats.serial}"
    camera_dir.mkdir(parents=True, exist_ok=True)
    log_path = camera_dir / "frames.csv"
    image_processor = None
    if demosaic_on_save:
        image_processor = PySpin.ImageProcessor()
        image_processor.SetColorProcessing(
            PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR
        )

    try:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "host_received_ns",
                    "capture_sequence_index",
                    "camera_frame_id",
                    "camera_timestamp",
                    "complete",
                    "image_status",
                    "image_path",
                ),
            )
            writer.writeheader()

            while not stop_event.is_set():
                try:
                    image = camera.GetNextImage(image_timeout_ms)
                except PySpin.SpinnakerException as exc:
                    if stop_event.is_set():
                        break
                    if is_timeout_exception(exc):
                        with stats_lock:
                            stats.timeouts += 1
                        continue
                    fatal_errors.put(
                        f"Camera {stats.serial} acquisition failed: {exc}"
                    )
                    return

                host_received_ns = time.time_ns()
                last_activity[stats.camera_index] = time.monotonic()
                try:
                    frame_id = int(image.GetFrameID())
                    camera_timestamp = int(image.GetTimeStamp())

                    with stats_lock:
                        if stats.first_frame_id < 0:
                            stats.first_frame_id = frame_id
                        sequence_index = frame_id - stats.first_frame_id
                        if (
                            stats.last_frame_id >= 0
                            and frame_id > stats.last_frame_id + 1
                        ):
                            stats.dropped_images += (
                                frame_id - stats.last_frame_id - 1
                            )
                        stats.last_frame_id = frame_id
                        stats.last_sequence_index = sequence_index

                    complete = not image.IsIncomplete()
                    image_status = int(image.GetImageStatus())
                    relative_path = ""

                    if complete:
                        with stats_lock:
                            stats.complete_images += 1
                        if save_images:
                            filename = (
                                f"capture_{sequence_index:09d}_"
                                f"fid_{frame_id}.{image_format}"
                            )
                            image_path = camera_dir / filename
                            converted = None
                            try:
                                image_to_save = image
                                if demosaic_on_save:
                                    converted = image_processor.Convert(
                                        image,
                                        PySpin.PixelFormat_BGR8,
                                    )
                                    image_to_save = converted
                                image_to_save.Save(str(image_path))
                            finally:
                                if converted is not None:
                                    converted.Release()
                            relative_path = str(
                                image_path.relative_to(session_dir)
                            )
                            with stats_lock:
                                stats.saved_images += 1
                    else:
                        with stats_lock:
                            stats.incomplete_images += 1

                    writer.writerow(
                        {
                            "host_received_ns": host_received_ns,
                            "capture_sequence_index": sequence_index,
                            "camera_frame_id": frame_id,
                            "camera_timestamp": camera_timestamp,
                            "complete": int(complete),
                            "image_status": image_status,
                            "image_path": relative_path,
                        }
                    )
                    with stats_lock:
                        complete_count = stats.complete_images
                    if complete_count % 10 == 0:
                        handle.flush()
                finally:
                    image.Release()
    except Exception as exc:
        if not stop_event.is_set():
            fatal_errors.put(f"Camera {stats.serial} writer failed: {exc}")


def unique_archive_path(directory, source_name, suffix):
    candidate = directory / source_name
    if not candidate.exists():
        return candidate
    source = Path(source_name)
    return directory / f"{source.stem}_{suffix}{source.suffix}"


def archive_force_export(source_path, session_dir, detected_ns, stable_ns):
    force_dir = session_dir / "force"
    force_dir.mkdir(parents=True, exist_ok=True)
    source_stat = source_path.stat()
    destination = unique_archive_path(
        force_dir,
        source_path.name,
        str(detected_ns),
    )
    shutil.copy2(source_path, destination)
    return ForceExportRecord(
        source_path=str(source_path.resolve()),
        archived_path=str(destination.resolve()),
        detected_ns=int(detected_ns),
        stable_ns=int(stable_ns),
        source_creation_ns=int(source_stat.st_ctime_ns),
        source_write_ns=int(source_stat.st_mtime_ns),
        source_size=int(source_stat.st_size),
    )


def force_export_worker(
    export_dir,
    session_dir,
    records,
    records_lock,
    stop_event,
    poll_interval=DEFAULT_FORCE_POLL_INTERVAL_S,
    stable_interval=DEFAULT_FORCE_STABLE_INTERVAL_S,
):
    try:
        known = {}
        for path in export_dir.glob("*.csv"):
            try:
                stat = path.stat()
                known[str(path.resolve())] = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                continue

        pending = {}
        while not stop_event.is_set():
            now_monotonic = time.monotonic()
            try:
                paths = list(export_dir.glob("*.csv"))
            except OSError:
                stop_event.wait(poll_interval)
                continue

            for path in paths:
                try:
                    resolved = str(path.resolve())
                    stat = path.stat()
                except OSError:
                    continue

                signature = (stat.st_size, stat.st_mtime_ns)
                if known.get(resolved) == signature:
                    continue

                candidate = pending.get(resolved)
                if (
                    candidate is None
                    or candidate["signature"] != signature
                ):
                    pending[resolved] = {
                        "signature": signature,
                        "changed_at": now_monotonic,
                        "detected_ns": time.time_ns(),
                    }
                    continue

                if now_monotonic - candidate["changed_at"] < stable_interval:
                    continue

                stable_ns = time.time_ns()
                try:
                    record = archive_force_export(
                        path,
                        session_dir,
                        candidate["detected_ns"],
                        stable_ns,
                    )
                except OSError:
                    candidate["changed_at"] = now_monotonic
                    continue

                with records_lock:
                    records.append(record)
                known[resolved] = signature
                del pending[resolved]
                print(
                    "IntelliMESUR export archived: "
                    f"{Path(record.archived_path).name}",
                    flush=True,
                )

            stop_event.wait(poll_interval)
    except Exception as exc:
        print(f"Force export monitor warning: {exc}", flush=True)


def parse_intellimesur_export(path):
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        lines = handle.readlines()

    header_index = None
    metadata = {}
    for index, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        if stripped.startswith("Reading\t"):
            header_index = index
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[key.strip()] = value.strip()

    if header_index is None:
        raise ValueError("No IntelliMESUR Reading table was found")

    reader = csv.DictReader(lines[header_index:], delimiter="\t")
    fieldnames = [name for name in (reader.fieldnames or []) if name]
    load_field = next(
        (name for name in fieldnames if name.startswith("Load [")),
        None,
    )
    distance_field = next(
        (name for name in fieldnames if name.startswith("Distance [")),
        None,
    )
    time_field = next(
        (name for name in fieldnames if name.startswith("Time [")),
        None,
    )
    if load_field is None or time_field is None:
        raise ValueError("The force export has no load or time column")

    readings = []
    load_values = []
    distance_values = []
    relative_times = []
    for row in reader:
        try:
            reading = int(row["Reading"])
            load_value = float(row[load_field])
            relative_time = float(row[time_field])
            distance_value = (
                float(row[distance_field])
                if distance_field and row.get(distance_field)
                else float("nan")
            )
        except (KeyError, TypeError, ValueError):
            continue
        readings.append(reading)
        load_values.append(load_value)
        distance_values.append(distance_value)
        relative_times.append(relative_time)

    if len(relative_times) < 2:
        raise ValueError("The force export contains fewer than two data rows")

    return {
        "metadata": metadata,
        "reading": np.asarray(readings, dtype=np.int64),
        "load": np.asarray(load_values, dtype=np.float64),
        "distance": np.asarray(distance_values, dtype=np.float64),
        "time_s": np.asarray(relative_times, dtype=np.float64),
        "load_field": load_field,
        "distance_field": distance_field,
        "time_field": time_field,
        "duration_s": float(relative_times[-1] - relative_times[0]),
    }


def load_sensor_alignment_data(session_dir, args):
    corrected_v2 = session_dir / "sensor_corrected_v2.csv"
    corrected = session_dir / "sensor_corrected.csv"
    raw = session_dir / "sensor_raw.csv"
    if corrected_v2.exists():
        sensor_path = corrected_v2
    elif corrected.exists():
        sensor_path = corrected
    else:
        sensor_path = raw
    if not sensor_path.exists():
        raise FileNotFoundError(
            "No sensor_corrected_v2.csv, sensor_corrected.csv, "
            "or sensor_raw.csv"
        )

    host_times = []
    frame_indices = []
    device_millis = []
    adc_rows = []
    with sensor_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cell_fields = [
            name for name in (reader.fieldnames or [])
            if name.endswith("_adc8")
        ]
        if len(cell_fields) != VALUE_COUNT:
            raise ValueError(
                f"Expected {VALUE_COUNT} sensor cells, found "
                f"{len(cell_fields)}"
            )
        for row in reader:
            try:
                host_times.append(int(row["host_received_ns"]))
                frame_indices.append(int(row["frame_index"]))
                device_millis.append(int(row["device_millis"]))
                adc_rows.append([float(row[name]) for name in cell_fields])
            except (KeyError, TypeError, ValueError):
                continue

    if len(host_times) < 10:
        raise ValueError("Fewer than 10 valid tactile sensor frames")

    adc = np.asarray(adc_rows, dtype=np.float64)
    sensor_voltage = args.vdrive - adc * args.adc_ref / ADC8_MAX_VALUE
    return {
        "path": str(sensor_path.resolve()),
        "host_ns": np.asarray(host_times, dtype=np.int64),
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        "device_millis": np.asarray(device_millis, dtype=np.int64),
        "voltage": sensor_voltage,
    }


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    window = max(1, min(int(window), len(values)))
    if window == 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.full(window, 1.0 / window)
    return np.convolve(padded, kernel, mode="valid")


def pearson_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 8:
        return None
    left = left[valid] - np.mean(left[valid])
    right = right[valid] - np.mean(right[valid])
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def build_sensor_features(sensor):
    host_ns = sensor["host_ns"]
    voltage = sensor["voltage"]
    dt_s = float(np.median(np.diff(host_ns))) / 1e9
    baseline_count = max(
        3,
        min(len(voltage), int(round(1.0 / max(dt_s, 1e-6)))),
    )
    baseline = np.median(voltage[:baseline_count], axis=0)
    delta = voltage - baseline
    features = {
        "mean_delta_v": np.mean(delta, axis=1),
        "median_delta_v": np.median(delta, axis=1),
        "rms_delta_v": np.sqrt(np.mean(delta * delta, axis=1)),
        "spatial_std_v": np.std(voltage, axis=1),
    }

    try:
        centered = voltage - np.mean(voltage, axis=0)
        left_vectors, singular_values, _ = np.linalg.svd(
            centered,
            full_matrices=False,
        )
        features["pca1"] = left_vectors[:, 0] * singular_values[0]
    except np.linalg.LinAlgError:
        pass

    smooth_window = max(1, int(round(0.20 / max(dt_s, 1e-6))))
    return {
        name: moving_average(values, smooth_window)
        for name, values in features.items()
    }, dt_s, smooth_window


def find_force_alignment(force, sensor, record, args):
    features, sensor_dt_s, smooth_window = build_sensor_features(sensor)
    force_time_s = force["time_s"] - force["time_s"][0]
    force_duration_s = float(force_time_s[-1])
    anchor_ns = record.detected_ns or record.source_creation_ns
    coarse_start_ns = int(
        anchor_ns
        - (force_duration_s + args.force_export_delay) * 1e9
    )
    step_s = max(0.005, min(0.025, sensor_dt_s / 4.0))
    offsets_s = np.arange(
        -args.force_search_window,
        args.force_search_window + step_s * 0.5,
        step_s,
    )

    best = None
    for offset_s in offsets_s:
        start_ns = coarse_start_ns + int(round(offset_s * 1e9))
        candidate_export_delay_s = (
            (anchor_ns - start_ns) / 1e9 - force_duration_s
        )
        if candidate_export_delay_s < 0:
            continue
        relative_s = (sensor["host_ns"] - start_ns) / 1e9
        in_run = (relative_s >= 0.0) & (
            relative_s <= force_duration_s
        )
        if np.count_nonzero(in_run) < 15:
            continue

        force_at_sensor = np.interp(
            relative_s[in_run],
            force_time_s,
            force["load"],
        )
        force_at_sensor = moving_average(
            force_at_sensor,
            min(smooth_window, len(force_at_sensor)),
        )

        for feature_name, feature_values in features.items():
            sensor_values = feature_values[in_run]
            correlations = {
                "level": pearson_correlation(
                    sensor_values,
                    force_at_sensor,
                ),
                "derivative": pearson_correlation(
                    np.diff(sensor_values),
                    np.diff(force_at_sensor),
                ),
            }
            for metric, correlation in correlations.items():
                if correlation is None:
                    continue
                score = abs(correlation)
                if best is None or score > best["score"]:
                    best = {
                        "score": float(score),
                        "signed_correlation": float(correlation),
                        "feature": feature_name,
                        "metric": metric,
                        "estimated_force_start_ns": int(start_ns),
                        "coarse_force_start_ns": int(coarse_start_ns),
                        "coarse_offset_s": float(offset_s),
                        "overlap_sensor_frames": int(
                            np.count_nonzero(in_run)
                        ),
                    }

    if best is None:
        best = {
            "score": 0.0,
            "signed_correlation": 0.0,
            "feature": None,
            "metric": "export_delay_only",
            "estimated_force_start_ns": int(coarse_start_ns),
            "coarse_force_start_ns": int(coarse_start_ns),
            "coarse_offset_s": 0.0,
            "overlap_sensor_frames": 0,
        }

    best["force_duration_s"] = force_duration_s
    best["force_samples"] = int(len(force_time_s))
    best["sensor_samples"] = int(len(sensor["host_ns"]))
    best["sensor_period_s"] = float(sensor_dt_s)
    best["estimated_export_delay_s"] = float(
        (anchor_ns - best["estimated_force_start_ns"]) / 1e9
        - force_duration_s
    )
    if best["score"] >= 0.70:
        best["confidence"] = "high"
    elif best["score"] >= args.force_min_correlation:
        best["confidence"] = "medium"
    else:
        best["confidence"] = "low"
    return best, features


def nearest_sensor_indices(sensor_ns, force_ns):
    positions = np.searchsorted(sensor_ns, force_ns)
    positions = np.clip(positions, 0, len(sensor_ns) - 1)
    previous = np.maximum(positions - 1, 0)
    use_previous = (
        np.abs(force_ns - sensor_ns[previous])
        <= np.abs(sensor_ns[positions] - force_ns)
    )
    return np.where(use_previous, previous, positions)


def write_force_alignment_outputs(
    session_dir,
    force,
    sensor,
    record,
    alignment,
    features,
    output_stem,
):
    force_dir = session_dir / "force"
    force_dir.mkdir(parents=True, exist_ok=True)
    force_time_s = force["time_s"] - force["time_s"][0]
    start_ns = alignment["estimated_force_start_ns"]

    sensor_output = force_dir / f"sensor_force_{output_stem}.csv"
    sensor_relative_s = (sensor["host_ns"] - start_ns) / 1e9
    sensor_in_run = (
        (sensor_relative_s >= 0)
        & (sensor_relative_s <= force_time_s[-1])
    )
    with sensor_output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "host_received_ns",
            "sensor_row_index",
            "capture_sequence_index",
            "frame_index",
            "device_millis",
            "force_time_s",
            "force_load",
            "force_distance",
            "force_in_run",
            *features.keys(),
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(sensor["host_ns"])):
            in_run = bool(sensor_in_run[index])
            relative_s = float(sensor_relative_s[index])
            row = {
                "host_received_ns": int(sensor["host_ns"][index]),
                "sensor_row_index": index,
                "capture_sequence_index": int(
                    sensor["frame_index"][index]
                ),
                "frame_index": int(sensor["frame_index"][index]),
                "device_millis": int(sensor["device_millis"][index]),
                "force_time_s": f"{relative_s:.6f}" if in_run else "",
                "force_load": (
                    f"{np.interp(relative_s, force_time_s, force['load']):.9g}"
                    if in_run
                    else ""
                ),
                "force_distance": (
                    f"{np.interp(relative_s, force_time_s, force['distance']):.9g}"
                    if in_run
                    else ""
                ),
                "force_in_run": int(in_run),
            }
            for name, values in features.items():
                row[name] = f"{values[index]:.9g}"
            writer.writerow(row)

    force_output = force_dir / f"force_samples_{output_stem}.csv"
    force_host_ns = start_ns + np.rint(force_time_s * 1e9).astype(np.int64)
    nearest = nearest_sensor_indices(sensor["host_ns"], force_host_ns)
    with force_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "reading",
                "estimated_host_ns",
                "force_time_s",
                "force_load",
                "force_distance",
                "nearest_sensor_row_index",
                "nearest_capture_sequence_index",
                "nearest_sensor_frame_index",
                "nearest_sensor_delta_ms",
            ]
        )
        for index in range(len(force_time_s)):
            sensor_index = int(nearest[index])
            delta_ms = (
                int(force_host_ns[index])
                - int(sensor["host_ns"][sensor_index])
            ) / 1e6
            writer.writerow(
                [
                    int(force["reading"][index]),
                    int(force_host_ns[index]),
                    f"{force_time_s[index]:.6f}",
                    f"{force['load'][index]:.9g}",
                    f"{force['distance'][index]:.9g}",
                    sensor_index,
                    int(sensor["frame_index"][sensor_index]),
                    int(sensor["frame_index"][sensor_index]),
                    f"{delta_ms:.6f}",
                ]
            )

    alignment_output = force_dir / f"alignment_{output_stem}.json"
    payload = {
        **alignment,
        "source_force_export": record.source_path,
        "archived_force_export": record.archived_path,
        "sensor_source": sensor["path"],
        "force_columns": {
            "load": force["load_field"],
            "distance": force["distance_field"],
            "time": force["time_field"],
        },
        "force_metadata": force["metadata"],
        "sensor_force_file": str(sensor_output.resolve()),
        "force_samples_file": str(force_output.resolve()),
    }
    with alignment_output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return {
        "alignment": payload,
        "alignment_path": alignment_output,
        "sensor_force_path": sensor_output,
        "force_samples_path": force_output,
    }


def process_force_exports(session_dir, records, args):
    force_dir = session_dir / "force"
    force_dir.mkdir(parents=True, exist_ok=True)
    with (force_dir / "exports.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, indent=2)

    if not records:
        return {
            "exports_detected": 0,
            "best_alignment": None,
        }
    if np is None:
        raise RuntimeError(
            "NumPy is required for force/sensor alignment but is unavailable"
        )

    sensor = load_sensor_alignment_data(session_dir, args)
    completed = []
    for index, record in enumerate(records, start=1):
        try:
            force = parse_intellimesur_export(Path(record.archived_path))
            alignment, features = find_force_alignment(
                force,
                sensor,
                record,
                args,
            )
            output_stem = f"{index:02d}_{Path(record.archived_path).stem}"
            result = write_force_alignment_outputs(
                session_dir,
                force,
                sensor,
                record,
                alignment,
                features,
                output_stem,
            )
            completed.append(result)
        except Exception as exc:
            print(
                f"Force alignment warning for {record.archived_path}: {exc}"
            )

    if not completed:
        return {
            "exports_detected": len(records),
            "best_alignment": None,
        }

    best = max(
        completed,
        key=lambda item: item["alignment"]["score"],
    )
    shutil.copy2(
        best["alignment_path"],
        session_dir / "force_alignment.json",
    )
    shutil.copy2(
        best["sensor_force_path"],
        session_dir / "sensor_force_aligned.csv",
    )
    shutil.copy2(
        best["force_samples_path"],
        session_dir / "force_samples_aligned.csv",
    )
    print(
        "Force alignment: "
        f"score={best['alignment']['score']:.3f}, "
        f"confidence={best['alignment']['confidence']}, "
        f"feature={best['alignment']['feature']}, "
        f"metric={best['alignment']['metric']}"
    )
    return {
        "exports_detected": len(records),
        "alignments_completed": len(completed),
        "best_alignment": best["alignment"],
        "best_alignment_file": "force_alignment.json",
        "sensor_force_file": "sensor_force_aligned.csv",
        "force_samples_file": "force_samples_aligned.csv",
    }


def create_session_directory(output_root):
    session_name = datetime.now().strftime("capture_%Y%m%d_%H%M%S")
    session_dir = output_root.expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def save_session_metadata(
    session_dir,
    args,
    serial_numbers,
    sensor_stats,
    camera_stats,
    force_summary,
):
    sequential_pairing_clean = (
        sensor_stats.frames > 0
        and sensor_stats.dropped_frames == 0
        and all(
            item.complete_images == sensor_stats.frames
            and item.incomplete_images == 0
            and item.dropped_images == 0
            for item in camera_stats
        )
    )
    metadata = {
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "arduino_port": args.port,
        "baud": args.baud,
        "serial_rx_buffer_bytes": args.serial_rx_buffer,
        "sensor_queue_frames": args.sensor_queue_frames,
        "trigger_source": args.trigger_source,
        "camera_count": len(serial_numbers),
        "requested_camera_count": args.cameras,
        "camera_serials": serial_numbers,
        "image_format": args.image_format,
        "color_requested": args.color,
        "demosaic_on_save": args.demosaic_on_save,
        "images_enabled": not args.no_save_images,
        "sensor_protocol": "M16B v1",
        "sensor_correction": {
            "enabled": not args.no_open_cell_interpolation,
            "raw_file": "sensor_raw.csv",
            "corrected_file": (
                "sensor_corrected.csv"
                if not args.no_open_cell_interpolation
                else None
            ),
            "anomaly_file": (
                "sensor_anomalies.csv"
                if not args.no_open_cell_interpolation
                else None
            ),
            "method": (
                "isolated/open-line detection with spatial "
                "inverse-distance-squared interpolation"
            ),
            "adc_reference_v": args.adc_ref,
            "sensor_drive_v": args.vdrive,
            "open_cell_voltage_v": args.open_cell_voltage,
            "point_open_voltage_v": args.point_open_voltage,
            "point_neighbor_delta_v": args.point_neighbor_delta,
            "line_open_voltage_v": args.line_open_voltage,
            "line_open_fraction": args.line_open_fraction,
            "line_neighbor_delta_v": args.line_neighbor_delta,
            "maximum_radius_cells": args.interpolation_radius,
            "minimum_neighbors": args.interpolation_min_neighbors,
        },
        "sequential_pairing_clean": sequential_pairing_clean,
        "sensor_stats": asdict(sensor_stats),
        "camera_stats": [asdict(item) for item in camera_stats],
        "force_export": {
            "monitor_enabled": not args.no_force_export_monitor,
            "export_directory": str(args.force_export_dir),
            "coarse_export_delay_s": args.force_export_delay,
            "correlation_search_half_window_s": args.force_search_window,
            **(force_summary or {}),
        },
    }
    with (session_dir / "session.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main():
    args = parse_args()
    if args.align_only:
        if np is None:
            raise SystemExit(
                "NumPy is required for --align-only but is unavailable."
            )
        session_dir = args.align_only[0].expanduser().resolve()
        source_force = args.align_only[1].expanduser().resolve()
        if not session_dir.is_dir():
            raise SystemExit(f"Capture session does not exist: {session_dir}")
        if not source_force.is_file():
            raise SystemExit(f"Force export does not exist: {source_force}")
        stat = source_force.stat()
        record = archive_force_export(
            source_force,
            session_dir,
            stat.st_ctime_ns,
            time.time_ns(),
        )
        summary = process_force_exports(
            session_dir,
            [record],
            args,
        )
        if summary["best_alignment"] is None:
            raise SystemExit("Force alignment could not be completed.")
        print(f"Aligned capture data: {session_dir}")
        return 0

    if PySpin is None:
        raise SystemExit(
            "PySpin is not available in this Python environment. Install the "
            "Teledyne Spinnaker SDK and its matching Python wrapper."
        )

    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    disk_usage = shutil.disk_usage(output_root)
    free_gb = disk_usage.free / 1024**3
    print(
        f"Capture storage: {output_root} | free={free_gb:.1f} GiB | "
        f"cameras={args.cameras}",
        flush=True,
    )
    if not args.no_save_images:
        # 2048x1536 raw Bayer BMP is about 3 MiB per frame per camera. Adding a
        # camera raises the sustained write load proportionally, so print the
        # requirement before the run rather than discovering it as frame loss.
        estimated_mb_per_s = args.cameras * 3.0 * 24.0
        print(
            f"Estimated sustained write load at 24 fps: "
            f"~{estimated_mb_per_s:.0f} MiB/s across {args.cameras} cameras",
            flush=True,
        )
    if (
        not args.no_save_images
        and args.minimum_free_gb > 0
        and free_gb < args.minimum_free_gb
    ):
        raise SystemExit(
            f"Only {free_gb:.1f} GiB is free on {output_root.anchor}; "
            f"at least {args.minimum_free_gb:g} GiB is required for "
            "full-resolution image capture. Choose another --output drive "
            "or free disk space."
        )

    session_dir = create_session_directory(output_root)
    print(f"Session directory: {session_dir}")

    system = None
    camera_list = None
    cameras = []
    serial_numbers = []
    camera_pixel_formats = []
    serial_port = None
    acquisition_started = []

    stop_event = threading.Event()
    heartbeat_stop_event = threading.Event()
    fatal_errors = queue.Queue()
    serial_write_lock = threading.Lock()
    stats_lock = threading.Lock()
    sensor_stats = SensorStats()
    sensor_frame_queue = queue.Queue(maxsize=args.sensor_queue_frames)
    sensor_reader_done = threading.Event()
    camera_stats = []
    sensor_last_activity = [0.0]
    camera_last_activity = [0.0] * args.cameras
    workers = []
    stopped_normally = False
    force_records = []
    force_records_lock = threading.Lock()
    force_monitor_stop_event = threading.Event()
    force_monitor_thread = None
    force_summary = None

    if not args.no_force_export_monitor:
        export_dir = args.force_export_dir.expanduser().resolve()
        if export_dir.is_dir():
            force_monitor_thread = threading.Thread(
                target=force_export_worker,
                args=(
                    export_dir,
                    session_dir,
                    force_records,
                    force_records_lock,
                    force_monitor_stop_event,
                ),
                name="intellimesur-export-monitor",
                daemon=True,
            )
            force_monitor_thread.start()
            print(f"Watching IntelliMESUR exports: {export_dir}")
        else:
            print(
                "Force export monitor disabled for this run; directory "
                f"does not exist: {export_dir}"
            )

    try:
        (
            system,
            camera_list,
            cameras,
            serial_numbers,
            camera_pixel_formats,
        ) = initialize_cameras(args)
        camera_stats = [
            CameraStats(index, serial_number)
            for index, serial_number in enumerate(serial_numbers)
        ]
        for stats, pixel_format in zip(camera_stats, camera_pixel_formats):
            stats.source_pixel_format = pixel_format
            stats.saved_pixel_format = (
                "BGR8" if args.demosaic_on_save else pixel_format
            )

        for camera in cameras:
            camera.BeginAcquisition()
            acquisition_started.append(camera)
        wait_for_cameras_armed(cameras)
        print(f"All {len(cameras)} cameras are acquiring and armed.")

        for camera, stats in zip(cameras, camera_stats):
            worker = threading.Thread(
                target=camera_worker,
                args=(
                    camera,
                    stats,
                    session_dir,
                    args.image_format,
                    not args.no_save_images,
                    args.demosaic_on_save,
                    args.image_timeout_ms,
                    stats_lock,
                    camera_last_activity,
                    stop_event,
                    fatal_errors,
                ),
                name=f"camera-{stats.serial}",
                daemon=True,
            )
            worker.start()
            workers.append(worker)

        serial_port = serial.Serial(
            args.port,
            args.baud,
            timeout=0.2,
            write_timeout=1.0,
        )
        print(f"Arduino serial: {args.port} at {args.baud} baud")
        try:
            serial_port.set_buffer_size(
                rx_size=args.serial_rx_buffer,
                tx_size=4096,
            )
            print(
                "Arduino serial receive buffer: "
                f"{args.serial_rx_buffer} bytes"
            )
        except (AttributeError, OSError, serial.SerialException) as exc:
            print(f"Serial buffer configuration warning: {exc}")
        time.sleep(0.2)
        serial_port.reset_input_buffer()
        wait_for_arduino_handshake(
            serial_port,
            serial_write_lock,
            args.handshake_timeout,
        )
        capture_start = time.monotonic()

        heartbeat_thread = threading.Thread(
            target=heartbeat_worker,
            args=(
                serial_port,
                serial_write_lock,
                heartbeat_stop_event,
                args.heartbeat_interval,
            ),
            name="arduino-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        workers.append(heartbeat_thread)

        sensor_writer_thread = threading.Thread(
            target=sensor_writer_worker,
            args=(
                sensor_frame_queue,
                sensor_reader_done,
                session_dir,
                sensor_stats,
                stats_lock,
                stop_event,
                fatal_errors,
                not args.no_open_cell_interpolation,
                args.adc_ref,
                args.vdrive,
                args.open_cell_voltage,
                args.point_open_voltage,
                args.point_neighbor_delta,
                args.line_open_voltage,
                args.line_open_fraction,
                args.line_neighbor_delta,
                args.interpolation_radius,
                args.interpolation_min_neighbors,
            ),
            name="sensor-writer",
            daemon=True,
        )
        sensor_writer_thread.start()
        workers.append(sensor_writer_thread)

        sensor_reader_thread = threading.Thread(
            target=sensor_serial_reader_worker,
            args=(
                serial_port,
                sensor_frame_queue,
                sensor_reader_done,
                sensor_stats,
                stats_lock,
                sensor_last_activity,
                stop_event,
                fatal_errors,
            ),
            name="sensor-serial-reader",
            daemon=True,
        )
        sensor_reader_thread.start()
        workers.append(sensor_reader_thread)

        print("Capture running. Press Ctrl+C to stop.")
        last_report = time.monotonic()
        while True:
            try:
                error = fatal_errors.get_nowait()
            except queue.Empty:
                error = None
            if error:
                raise RuntimeError(error)

            with stats_lock:
                sensor_count = sensor_stats.frames
                camera_counts = [item.complete_images for item in camera_stats]

            if args.frames and sensor_count >= args.frames:
                stopped_normally = True
                print(f"Requested {args.frames} sensor frames captured.")
                break

            now = time.monotonic()
            sensor_reference = sensor_last_activity[0] or capture_start
            if now - sensor_reference > args.stall_timeout:
                raise TimeoutError(
                    f"No Arduino sensor frame for {args.stall_timeout:g} s"
                )
            for index, serial_number in enumerate(serial_numbers):
                camera_reference = camera_last_activity[index] or capture_start
                if now - camera_reference > args.stall_timeout:
                    raise TimeoutError(
                        f"No image from camera {serial_number} for "
                        f"{args.stall_timeout:g} s"
                    )

            if now - last_report >= 1.0:
                print(
                    f"sensor={sensor_count} | cameras={camera_counts}",
                    flush=True,
                )
                last_report = now
            time.sleep(0.05)

    except KeyboardInterrupt:
        stopped_normally = True
        print("Stopping capture...")
    except Exception as exc:
        print(f"Capture failed: {exc}")
    finally:
        heartbeat_stop_event.set()
        if serial_port is not None and serial_port.is_open:
            try:
                write_serial_command(
                    serial_port,
                    serial_write_lock,
                    COMMAND_STOP,
                )
            except Exception:
                pass

        if stopped_normally and camera_stats:
            # STOP can arrive while the Arduino is scanning. In that case it
            # completes and triggers one final frame before processing STOP on
            # the next loop. Wait until every receiver count (the sensor plus
            # each camera) has stopped changing so that this in-flight frame is
            # not cut off unevenly.
            print("Draining the final in-flight frame...")
            deadline = time.monotonic() + args.drain_timeout
            stable_since = time.monotonic()
            previous_counts = None
            while time.monotonic() < deadline:
                with stats_lock:
                    counts = (
                        sensor_stats.frames,
                        *(item.complete_images for item in camera_stats),
                    )
                now = time.monotonic()
                if counts != previous_counts:
                    previous_counts = counts
                    stable_since = now
                elif now - stable_since >= 0.25:
                    break
                time.sleep(0.01)

            if previous_counts is not None:
                print(
                    "Drained counts: sensor="
                    f"{previous_counts[0]} | cameras="
                    f"{list(previous_counts[1:])}"
                )

        stop_event.set()
        for worker in workers:
            worker.join(timeout=max(2.0, args.image_timeout_ms / 1000.0 + 1.0))

        still_running = [worker.name for worker in workers if worker.is_alive()]
        if still_running:
            print(
                "Worker shutdown warning; still running: "
                + ", ".join(still_running)
            )

        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass

        for camera in reversed(acquisition_started):
            try:
                camera.EndAcquisition()
            except Exception as exc:
                print(f"EndAcquisition warning: {exc}")

        for camera in reversed(cameras):
            try:
                set_enum(camera.GetNodeMap(), "TriggerMode", "Off", required=False)
            except Exception:
                pass
            try:
                camera.DeInit()
            except Exception as exc:
                print(f"Camera cleanup warning: {exc}")

        camera = None
        worker = None
        acquisition_started.clear()
        cameras.clear()
        workers.clear()

        if camera_list is not None:
            camera_list.Clear()
            camera_list = None
        gc.collect()
        if system is not None:
            try:
                system.ReleaseInstance()
            except PySpin.SpinnakerException as exc:
                print(f"System cleanup warning: {exc}")
            system = None

        if force_monitor_thread is not None:
            with force_records_lock:
                exports_seen = len(force_records)
            if (
                stopped_normally
                and exports_seen == 0
                and args.force_export_wait > 0
            ):
                print(
                    "Waiting up to "
                    f"{args.force_export_wait:g} s for IntelliMESUR export..."
                )
                deadline = time.monotonic() + args.force_export_wait
                while time.monotonic() < deadline:
                    with force_records_lock:
                        if force_records:
                            break
                    time.sleep(0.05)
            force_monitor_stop_event.set()
            force_monitor_thread.join(timeout=2.0)
            if force_monitor_thread.is_alive():
                print("Force export monitor shutdown warning")

        with force_records_lock:
            completed_force_records = list(force_records)
        try:
            force_summary = process_force_exports(
                session_dir,
                completed_force_records,
                args,
            )
            if not completed_force_records and not args.no_force_export_monitor:
                print("No new IntelliMESUR automatic export was detected.")
        except Exception as exc:
            print(f"Could not process force exports: {exc}")
            force_summary = {
                "exports_detected": len(completed_force_records),
                "error": str(exc),
            }

        try:
            save_session_metadata(
                session_dir,
                args,
                serial_numbers,
                sensor_stats,
                camera_stats,
                force_summary,
            )
        except Exception as exc:
            print(f"Could not save session metadata: {exc}")

    print(f"Capture data: {session_dir}")
    if camera_stats:
        camera_counts = [item.complete_images for item in camera_stats]
        pairing_clean = (
            sensor_stats.frames > 0
            and sensor_stats.dropped_frames == 0
            and all(
                item.complete_images == sensor_stats.frames
                and item.incomplete_images == 0
                and item.dropped_images == 0
                for item in camera_stats
            )
        )
        print(
            f"Final counts: sensor={sensor_stats.frames}, "
            f"cameras={camera_counts}, sequential pairing clean={pairing_clean}"
        )
        if not args.no_open_cell_interpolation:
            print(
                "Sensor correction: "
                f"frames_with_open_cells={sensor_stats.frames_with_open_cells}, "
                f"interpolated_cells="
                f"{sensor_stats.interpolated_cell_samples}, "
                f"unrepaired_cells={sensor_stats.unrepaired_cell_samples}, "
                f"bad_rows={sensor_stats.detected_bad_rows}, "
                f"bad_columns={sensor_stats.detected_bad_columns}"
            )
        print(
            "Sensor reader: "
            f"bad_frames={sensor_stats.bad_frames}, "
            f"duplicate_copies={sensor_stats.duplicate_frames}, "
            f"dropped_frames={sensor_stats.dropped_frames}, "
            f"queue_peak={sensor_stats.reader_queue_peak}/"
            f"{args.sensor_queue_frames}"
        )
    return 0 if stopped_normally else 1


if __name__ == "__main__":
    raise SystemExit(main())
