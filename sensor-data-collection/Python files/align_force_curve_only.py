#!/usr/bin/env python3
"""Globally align an emailed IntelliMESUR CSV to a sensor capture by shape.

This tool deliberately ignores source-file creation, modification, download,
and email timestamps. It searches the complete tactile recording for the
force-run placement that maximizes absolute Pearson correlation against
multiple 16x16 sensor features.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from capture_3blackfly_sensor_force import (
    DEFAULT_ADC_REFERENCE_V,
    DEFAULT_FORCE_MIN_CORRELATION,
    DEFAULT_SENSOR_DRIVE_V,
    ForceExportRecord,
    build_sensor_features,
    load_sensor_alignment_data,
    moving_average,
    parse_intellimesur_export,
    pearson_correlation,
    write_force_alignment_outputs,
)
from export_multimodal_mp4 import interpolate_force, load_force


DEFAULT_COARSE_STEP_S = 0.10
DEFAULT_REFINE_STEP_S = 0.005
DEFAULT_AMBIGUITY_EXCLUSION_S = 1.0
DEFAULT_MINIMUM_PEAK_MARGIN = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Align an IntelliMESUR CSV to a completed camera/sensor session "
            "using curve shape only. Filesystem timestamps are ignored."
        )
    )
    parser.add_argument(
        "session",
        type=Path,
        nargs="?",
        help="Capture session directory; prompted when omitted.",
    )
    parser.add_argument(
        "force_csv",
        type=Path,
        nargs="?",
        help="Emailed IntelliMESUR CSV; prompted when omitted.",
    )
    parser.add_argument(
        "--adc-ref",
        type=float,
        default=DEFAULT_ADC_REFERENCE_V,
    )
    parser.add_argument(
        "--vdrive",
        type=float,
        default=DEFAULT_SENSOR_DRIVE_V,
    )
    parser.add_argument(
        "--coarse-step",
        type=float,
        default=DEFAULT_COARSE_STEP_S,
        help="Global-search spacing in seconds.",
    )
    parser.add_argument(
        "--refine-step",
        type=float,
        default=DEFAULT_REFINE_STEP_S,
        help="Fine-search spacing around the strongest coarse peaks.",
    )
    parser.add_argument(
        "--ambiguity-exclusion",
        type=float,
        default=DEFAULT_AMBIGUITY_EXCLUSION_S,
        help=(
            "Ignore candidates this close to the best start when measuring "
            "the independent runner-up peak."
        ),
    )
    parser.add_argument(
        "--minimum-peak-margin",
        type=float,
        default=DEFAULT_MINIMUM_PEAK_MARGIN,
        help="Minimum best-minus-runner-up score for high confidence.",
    )
    parser.add_argument(
        "--minimum-correlation",
        type=float,
        default=DEFAULT_FORCE_MIN_CORRELATION,
        help="Minimum score for medium confidence.",
    )
    args = parser.parse_args()
    if args.adc_ref <= 0 or args.vdrive <= 0:
        parser.error("--adc-ref and --vdrive must be positive")
    if args.coarse_step <= 0 or args.refine_step <= 0:
        parser.error("--coarse-step and --refine-step must be positive")
    if args.refine_step > args.coarse_step:
        parser.error("--refine-step cannot exceed --coarse-step")
    if args.ambiguity_exclusion < 0:
        parser.error("--ambiguity-exclusion must be non-negative")
    if not 0 <= args.minimum_peak_margin <= 1:
        parser.error("--minimum-peak-margin must be between 0 and 1")
    if not 0 <= args.minimum_correlation <= 1:
        parser.error("--minimum-correlation must be between 0 and 1")
    if args.session is None:
        args.session = Path(
            input(
                "Drag the capture session folder here, then press Enter:\n> "
            ).strip().strip('"')
        )
    if args.force_csv is None:
        args.force_csv = Path(
            input(
                "Drag the emailed IntelliMESUR CSV here, then press Enter:\n> "
            ).strip().strip('"')
        )
    return args


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def unique_path(directory, source_name):
    candidate = directory / source_name
    if not candidate.exists():
        return candidate
    source = Path(source_name)
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"{source.stem}_{suffix}{source.suffix}"


def import_force_csv(source, session):
    force_dir = session / "force"
    force_dir.mkdir(parents=True, exist_ok=True)
    destination = unique_path(force_dir, source.name)
    shutil.copyfile(source, destination)
    return ForceExportRecord(
        source_path=str(source.resolve()),
        archived_path=str(destination.resolve()),
        detected_ns=0,
        stable_ns=0,
        source_creation_ns=0,
        source_write_ns=0,
        source_size=int(destination.stat().st_size),
    )


def evaluate_start(
    start_ns,
    force,
    sensor,
    features,
    smooth_window,
):
    force_time_s = force["time_s"] - force["time_s"][0]
    duration_s = float(force_time_s[-1])
    relative_s = (sensor["host_ns"] - int(start_ns)) / 1e9
    in_run = (relative_s >= 0.0) & (relative_s <= duration_s)
    overlap = int(np.count_nonzero(in_run))
    if overlap < 15:
        return None

    force_at_sensor = np.interp(
        relative_s[in_run],
        force_time_s,
        force["load"],
    )
    force_at_sensor = moving_average(
        force_at_sensor,
        min(smooth_window, overlap),
    )

    best = None
    for feature_name, feature_values in features.items():
        sensor_values = feature_values[in_run]
        correlations = {
            "level": pearson_correlation(sensor_values, force_at_sensor),
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
                    "overlap_sensor_frames": overlap,
                }
    return best


def separated_peaks(candidates, exclusion_ns, limit=5):
    peaks = []
    for candidate in sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True,
    ):
        start_ns = candidate["estimated_force_start_ns"]
        if all(
            abs(start_ns - peak["estimated_force_start_ns"]) > exclusion_ns
            for peak in peaks
        ):
            peaks.append(candidate)
            if len(peaks) >= limit:
                break
    return peaks


def find_curve_only_alignment(force, sensor, args):
    features, sensor_dt_s, smooth_window = build_sensor_features(sensor)
    force_time_s = force["time_s"] - force["time_s"][0]
    force_duration_s = float(force_time_s[-1])
    sensor_start_ns = int(sensor["host_ns"][0])
    sensor_end_ns = int(sensor["host_ns"][-1])
    latest_start_ns = sensor_end_ns - int(round(force_duration_s * 1e9))
    if latest_start_ns <= sensor_start_ns:
        raise ValueError(
            "The force run must be shorter than the tactile recording and "
            "fully contained inside it."
        )

    coarse_step_ns = max(1, int(round(args.coarse_step * 1e9)))
    refine_step_ns = max(1, int(round(args.refine_step * 1e9)))
    coarse_starts = np.arange(
        sensor_start_ns,
        latest_start_ns + 1,
        coarse_step_ns,
        dtype=np.int64,
    )
    if coarse_starts[-1] != latest_start_ns:
        coarse_starts = np.append(coarse_starts, latest_start_ns)

    coarse_results = []
    for start_ns in coarse_starts:
        result = evaluate_start(
            int(start_ns),
            force,
            sensor,
            features,
            smooth_window,
        )
        if result is not None:
            coarse_results.append(result)
    if not coarse_results:
        raise RuntimeError("No valid curve-alignment candidates were found.")

    coarse_top = sorted(
        coarse_results,
        key=lambda item: item["score"],
        reverse=True,
    )[:10]
    refined_starts = set()
    for candidate in coarse_top:
        center = candidate["estimated_force_start_ns"]
        lower = max(sensor_start_ns, center - coarse_step_ns)
        upper = min(latest_start_ns, center + coarse_step_ns)
        refined_starts.update(
            int(value)
            for value in np.arange(
                lower,
                upper + 1,
                refine_step_ns,
                dtype=np.int64,
            )
        )
        refined_starts.add(int(upper))

    refined_results = []
    for start_ns in sorted(refined_starts):
        result = evaluate_start(
            start_ns,
            force,
            sensor,
            features,
            smooth_window,
        )
        if result is not None:
            refined_results.append(result)

    all_results = coarse_results + refined_results
    exclusion_ns = int(round(args.ambiguity_exclusion * 1e9))
    peaks = separated_peaks(all_results, exclusion_ns)
    best = dict(peaks[0])
    runner_up = peaks[1] if len(peaks) > 1 else None
    runner_up_score = float(runner_up["score"]) if runner_up else None
    peak_margin = (
        float(best["score"] - runner_up_score)
        if runner_up_score is not None
        else 1.0
    )

    if (
        best["score"] >= 0.70
        and peak_margin >= args.minimum_peak_margin
    ):
        confidence = "high"
    elif best["score"] >= args.minimum_correlation:
        confidence = (
            "ambiguous"
            if peak_margin < args.minimum_peak_margin
            else "medium"
        )
    else:
        confidence = "low"

    best.update(
        {
            "alignment_mode": "curve_only_global",
            "uses_file_timestamps": False,
            "confidence": confidence,
            "runner_up_score": runner_up_score,
            "peak_margin": peak_margin,
            "minimum_peak_margin": float(args.minimum_peak_margin),
            "force_duration_s": force_duration_s,
            "force_samples": int(len(force_time_s)),
            "sensor_samples": int(len(sensor["host_ns"])),
            "sensor_period_s": float(sensor_dt_s),
            "global_search_start_ns": sensor_start_ns,
            "global_search_end_ns": latest_start_ns,
            "coarse_step_s": float(args.coarse_step),
            "refine_step_s": float(args.refine_step),
            "coarse_candidates": int(len(coarse_results)),
            "refined_candidates": int(len(refined_results)),
            "alternative_peaks": [
                {
                    "score": float(item["score"]),
                    "signed_correlation": float(
                        item["signed_correlation"]
                    ),
                    "feature": item["feature"],
                    "metric": item["metric"],
                    "estimated_force_start_ns": int(
                        item["estimated_force_start_ns"]
                    ),
                }
                for item in peaks[1:]
            ],
        }
    )
    return best, features


def install_canonical_outputs(session, result):
    shutil.copy2(
        result["alignment_path"],
        session / "force_alignment.json",
    )
    shutil.copy2(
        result["sensor_force_path"],
        session / "sensor_force_aligned.csv",
    )
    shutil.copy2(
        result["force_samples_path"],
        session / "force_samples_aligned.csv",
    )


def update_video_alignment(session):
    alignment_path = session / "multimodal_video_alignment.csv"
    if not alignment_path.is_file():
        raise FileNotFoundError(
            "Run the acquisition workflow through MP4 export before force "
            f"alignment: missing {alignment_path}"
        )
    force = load_force(session)
    if force is None:
        raise RuntimeError("The canonical aligned force file is invalid.")

    with alignment_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "capture_host_ns",
        "force_in_run",
        "force_time_s",
        "force_load",
        "force_distance",
    }
    if not required.issubset(fieldnames):
        raise ValueError(
            "The multimodal alignment table has an unsupported schema."
        )

    temporary = alignment_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            (
                force_time,
                force_load,
                force_distance,
                force_in_run,
            ) = interpolate_force(force, int(row["capture_host_ns"]))
            row.update(
                {
                    "force_in_run": force_in_run,
                    "force_time_s": force_time,
                    "force_load": force_load,
                    "force_distance": force_distance,
                }
            )
            writer.writerow(row)
    temporary.replace(alignment_path)

    metadata_path = session / "multimodal_video_export.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["force_file"] = "force_samples_aligned.csv"
        metadata["force_alignment"] = "force_alignment.json"
        metadata["force_alignment_mode"] = "curve_only_global"
        write_json_atomic(metadata_path, metadata)
    return len(rows)


def update_session_metadata(session, record, alignment):
    metadata_path = session / "session.json"
    metadata = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    metadata["force_export"] = {
        "monitor_enabled": False,
        "imported_after_capture": True,
        "source_force_export": record.source_path,
        "archived_force_export": record.archived_path,
        "alignment_mode": "curve_only_global",
        "uses_file_timestamps": False,
        "best_alignment_file": "force_alignment.json",
        "sensor_force_file": "sensor_force_aligned.csv",
        "force_samples_file": "force_samples_aligned.csv",
        "best_alignment": alignment,
    }
    write_json_atomic(metadata_path, metadata)


def main():
    args = parse_args()
    session = args.session.expanduser().resolve()
    source_force = args.force_csv.expanduser().resolve()
    if not session.is_dir():
        raise SystemExit(f"Capture session does not exist: {session}")
    if not source_force.is_file():
        raise SystemExit(f"Force CSV does not exist: {source_force}")

    record = import_force_csv(source_force, session)
    force = parse_intellimesur_export(Path(record.archived_path))
    sensor_args = SimpleNamespace(
        adc_ref=args.adc_ref,
        vdrive=args.vdrive,
    )
    sensor = load_sensor_alignment_data(session, sensor_args)
    alignment, features = find_curve_only_alignment(
        force,
        sensor,
        args,
    )
    output_stem = f"curve_only_{Path(record.archived_path).stem}"
    result = write_force_alignment_outputs(
        session,
        force,
        sensor,
        record,
        alignment,
        features,
        output_stem,
    )
    install_canonical_outputs(session, result)
    aligned_video_frames = update_video_alignment(session)
    update_session_metadata(session, record, alignment)

    import_record = {
        "imported_utc": utc_now(),
        "source": asdict(record),
        "uses_file_timestamps": False,
        "alignment": alignment,
        "updated_video_alignment_frames": aligned_video_frames,
    }
    write_json_atomic(
        session / "force_curve_alignment_import.json",
        import_record,
    )

    print(
        "Curve-only force alignment: "
        f"score={alignment['score']:.3f}, "
        f"confidence={alignment['confidence']}, "
        f"feature={alignment['feature']}, "
        f"metric={alignment['metric']}, "
        f"peak_margin={alignment['peak_margin']:.3f}"
    )
    print("Filesystem timestamps used: no")
    print(f"Updated video alignment rows: {aligned_video_frames}")
    print(f"Force import completed: {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
