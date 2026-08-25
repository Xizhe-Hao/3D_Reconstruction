#!/usr/bin/env python3
"""Re-run sensor anomaly repair and optionally fill lost sensor frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from capture_3blackfly_sensor_force import (
    MATRIX_COLS,
    MATRIX_ROWS,
    # OPTIONAL OPEN-CONTACT REPAIR: spatial high-voltage repair function.
    interpolate_open_cells,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)

    # BEGIN OPTIONAL OPEN-CONTACT REPAIR CLI
    parser.add_argument("--open-cell-voltage", type=float, default=4.8)
    parser.add_argument("--point-open-voltage", type=float, default=4.5)
    parser.add_argument("--point-neighbor-delta", type=float, default=0.75)
    parser.add_argument("--line-open-voltage", type=float, default=4.5)
    parser.add_argument("--line-open-fraction", type=float, default=0.75)
    parser.add_argument("--line-neighbor-delta", type=float, default=0.75)
    parser.add_argument("--interpolation-radius", type=int, default=3)
    parser.add_argument("--interpolation-min-neighbors", type=int, default=3)
    # END OPTIONAL OPEN-CONTACT REPAIR CLI

    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help=(
            "Create sensor_corrected_filled.csv on the common camera timeline. "
            "Missing frames are temporal estimates and are explicitly marked."
        ),
    )
    return parser.parse_args()


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


# BEGIN OPTIONAL OPEN-CONTACT REPAIR OFFLINE PASS
# This pass re-runs spatial high-voltage detection on sensor_raw.csv and writes
# sensor_corrected_v2.csv. It is separate from missing-frame interpolation.
def write_repaired(session, args):
    source = session / "sensor_raw.csv"
    rows = read_csv(source)
    if not rows:
        raise RuntimeError(f"No rows in {source}")
    cell_names = [
        f"r{row + 1}c{column + 1}_adc8"
        for row in range(MATRIX_ROWS)
        for column in range(MATRIX_COLS)
    ]
    header = ["host_received_ns", "frame_index", "device_millis"]
    corrected_path = session / "sensor_corrected_v2.csv"
    anomaly_path = session / "sensor_anomalies_v2.csv"
    corrected_rows = []
    line_frames = 0
    corrected_cells = 0
    unrepaired_cells = 0

    with corrected_path.open(
        "w", newline="", encoding="utf-8"
    ) as corrected_handle, anomaly_path.open(
        "w", newline="", encoding="utf-8"
    ) as anomaly_handle:
        corrected_writer = csv.writer(corrected_handle)
        corrected_writer.writerow(header + cell_names)
        anomaly_writer = csv.writer(anomaly_handle)
        anomaly_writer.writerow(
            header
            + [
                "bad_cell_count",
                "interpolated_cell_count",
                "unrepaired_cell_count",
                "bad_rows",
                "bad_columns",
                "bad_cells",
                "unrepaired_cells",
            ]
        )
        for row in rows:
            values = [int(row[name]) for name in cell_names]
            (
                corrected,
                bad,
                unrepaired,
                bad_rows,
                bad_columns,
            ) = interpolate_open_cells(
                values,
                5.0,
                5.0,
                args.open_cell_voltage,
                args.point_open_voltage,
                args.point_neighbor_delta,
                args.line_open_voltage,
                args.line_open_fraction,
                args.line_neighbor_delta,
                args.interpolation_radius,
                args.interpolation_min_neighbors,
            )
            common = [
                int(row["host_received_ns"]),
                int(row["frame_index"]),
                int(row["device_millis"]),
            ]
            corrected_writer.writerow(common + corrected)
            corrected_rows.append((common, corrected))
            if bad:
                anomaly_writer.writerow(
                    common
                    + [
                        len(bad),
                        len(bad) - len(unrepaired),
                        len(unrepaired),
                        ";".join(f"R{index + 1}" for index in bad_rows),
                        ";".join(f"C{index + 1}" for index in bad_columns),
                        ";".join(cell_names[index] for index in bad),
                        ";".join(cell_names[index] for index in unrepaired),
                    ]
                )
            if bad_rows or bad_columns:
                line_frames += 1
            corrected_cells += len(bad) - len(unrepaired)
            unrepaired_cells += len(unrepaired)

    print(
        f"Spatial repair: frames={len(rows)}, line_fault_frames={line_frames}, "
        f"corrected_cells={corrected_cells}, "
        f"unrepaired_cells={unrepaired_cells}"
    )
    print(f"Corrected sensor: {corrected_path}")
    print(f"Anomaly log: {anomaly_path}")
    return cell_names, corrected_rows
# END OPTIONAL OPEN-CONTACT REPAIR OFFLINE PASS


def common_camera_timeline(session):
    camera_dirs = sorted(
        path for path in session.glob("camera_*") if path.is_dir()
    )
    if len(camera_dirs) != 3:
        raise RuntimeError("Expected three camera directories")
    camera_maps = []
    for camera_dir in camera_dirs:
        rows = read_csv(camera_dir / "frames.csv")
        camera_maps.append(
            {
                int(row["capture_sequence_index"]): int(row["host_received_ns"])
                for row in rows
                if int(row.get("complete", "1")) == 1
            }
        )
    common = set(camera_maps[0])
    for mapping in camera_maps[1:]:
        common &= set(mapping)
    sequence = sorted(common)
    timestamps = {
        index: int(np.median([mapping[index] for mapping in camera_maps]))
        for index in sequence
    }
    return sequence, timestamps


def write_filled_timeline(session, cell_names, corrected_rows):
    # TEMPORAL FRAME-LOSS REPAIR, NOT OPEN-CONTACT REPAIR.
    # Keep this when using a good sensor if serial frames can still be missing.
    sequence, camera_ns = common_camera_timeline(session)
    row_map = {common[1]: (common, values) for common, values in corrected_rows}
    exact_indices = np.asarray(sorted(row_map), dtype=np.int64)
    exact_values = np.asarray(
        [row_map[int(index)][1] for index in exact_indices],
        dtype=np.float64,
    )
    exact_millis = np.asarray(
        [row_map[int(index)][0][2] for index in exact_indices],
        dtype=np.float64,
    )
    output = session / "sensor_corrected_filled.csv"
    estimated_count = 0

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "host_received_ns",
                "frame_index",
                "device_millis",
                "frame_source",
                *cell_names,
            ]
        )
        for frame_index in sequence:
            exact = row_map.get(frame_index)
            if exact is not None:
                common, values = exact
                writer.writerow([*common, "exact", *values])
                continue

            position = np.searchsorted(exact_indices, frame_index)
            if position == 0 or position == len(exact_indices):
                writer.writerow(
                    [
                        camera_ns[frame_index],
                        frame_index,
                        "",
                        "unavailable",
                        *([""] * len(cell_names)),
                    ]
                )
                continue
            left_index = exact_indices[position - 1]
            right_index = exact_indices[position]
            weight = (frame_index - left_index) / (right_index - left_index)
            values = np.rint(
                exact_values[position - 1] * (1.0 - weight)
                + exact_values[position] * weight
            ).astype(np.uint8)
            device_millis = round(
                exact_millis[position - 1] * (1.0 - weight)
                + exact_millis[position] * weight
            )
            writer.writerow(
                [
                    camera_ns[frame_index],
                    frame_index,
                    device_millis,
                    "temporal_interpolation",
                    *values.tolist(),
                ]
            )
            estimated_count += 1

    print(
        f"Filled timeline: frames={len(sequence)}, "
        f"temporal_interpolations={estimated_count}"
    )
    print(f"Filled sensor: {output}")


def main():
    args = parse_args()
    session = args.session.expanduser().resolve()
    if not session.is_dir():
        raise SystemExit(f"Session does not exist: {session}")
    cell_names, corrected_rows = write_repaired(session, args)
    if args.fill_missing:
        write_filled_timeline(session, cell_names, corrected_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
