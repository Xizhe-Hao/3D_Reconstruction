#!/usr/bin/env python3
"""Attach a force export to a capture session without guessing the clock offset.

    python align_force.py <session dir> <force csv>

RUN WITH THE **CAPTURE** ENVIRONMENT (.venv), the same one the capture uses.

Why this exists
---------------
`capture_3blackfly_sensor_force.py --align-only` anchors on the CSV's creation
time, assumes the run ended `--force-export-delay` seconds earlier, and then
correlates within `--force-search-window` (5 s by default) of that guess. That
works when IntelliMESUR writes the export straight into a watched folder on this
machine. It cannot work here: IntelliMESUR runs on a tablet, so the filename
carries the tablet's clock, and the file's creation time is whenever the CSV was
downloaded -- minutes to hours off. The search never reaches the truth and the
alignment silently reports `score=0.000, confidence=low`, which reads like a
failed measurement rather than a wrong starting point.

The offset is recoverable from the data instead of the clocks: the tactile array
and the force gauge see the same press, so sliding one against the other over
the whole plausible range finds it. This does that, converts the answer into the
`--force-export-delay` the existing tool wants, and hands over -- so the actual
alignment, its features and its outputs stay owned by one implementation.

It also says how much of the force run overlapped the capture, which is the
thing worth knowing before trusting a session: a run that started before the
cameras did is not salvageable by any amount of alignment.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

OPEN_ADC8 = 15          # at or below this the cell is reading its open-circuit rail
DEAD_FRACTION = 0.5     # a cell open in more than half the frames is not a sensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path, help="capture_YYYYmmdd_HHMMSS directory")
    parser.add_argument("force_csv", type=Path, help="IntelliMESUR export")
    parser.add_argument("--search-window", type=float, default=15.0,
                        help="seconds handed to the real aligner around the offset "
                             "found here")
    parser.add_argument("--min-correlation", type=float, default=0.35,
                        help="refuse to align below this; a low peak means the two "
                             "recordings do not describe the same press")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the offset and the command, change nothing")
    return parser.parse_args()


def load_sensor(session: Path):
    """Host timestamps and one scalar per frame, from the cells that still work."""
    with (session / "sensor_raw.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    cells = [i for i, name in enumerate(header) if name.endswith("_adc8")]
    stamps = np.array([int(row[0]) for row in rows[1:]], dtype=np.int64)
    adc = np.array([[int(row[i]) for i in cells] for row in rows[1:]], dtype=float)

    # Dead rows and columns would otherwise contribute a constant, which dilutes
    # the correlation without adding signal. Found from the data rather than
    # configured, so this keeps working as the array is repaired or degrades.
    healthy = (adc <= OPEN_ADC8).mean(axis=0) < DEAD_FRACTION
    if healthy.sum() < 16:
        raise SystemExit(f"only {healthy.sum()} live cells in {session.name}; "
                         "the tactile array is too damaged to align against")
    signal = adc[:, healthy].mean(axis=1)
    return stamps, signal, int(healthy.sum()), adc.shape[1]


def load_force(path: Path):
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines(True)
    start = next((i for i, line in enumerate(lines) if line.startswith("Reading\t")), None)
    if start is None:
        raise SystemExit(f"{path} has no IntelliMESUR Reading table")
    reader = list(csv.DictReader(lines[start:], delimiter="\t"))
    names = [name for name in (reader[0] or {}) if name]
    time_field = next(name for name in names if name.startswith("Time"))
    load_field = next(name for name in names if name.startswith("Load"))
    times = np.array([float(r[time_field]) for r in reader if r[time_field]])
    loads = np.array([float(r[load_field]) for r in reader if r[load_field]])
    return times - times[0], loads


def best_offset(sensor_t, sensor_signal, force_t, force_load, step=0.02):
    """Slide the force curve across the capture and keep the best-supported peak.

    Ranked by the correlation's t statistic, not the correlation itself. A short
    overlap correlates well by accident -- on one session a 4.5 s tail scored
    r=0.87 against the true 43.5 s alignment's r=0.74 -- and picking the larger
    r would have thrown away 90% of the run. t = r*sqrt((n-2)/(1-r^2)) prefers
    the peak the data actually supports.
    """
    baseline = np.median(sensor_signal[:min(60, len(sensor_signal))])
    response = sensor_signal - baseline
    duration = float(force_t[-1])
    best = (-1.0, 0.0, 0.0, 0)
    for offset in np.arange(-duration, sensor_t[-1], step):
        inside = (sensor_t >= offset) & (sensor_t <= offset + duration)
        count = int(inside.sum())
        if count < 100:
            continue
        measured = response[inside]
        expected = np.interp(sensor_t[inside] - offset, force_t, force_load)
        if measured.std() < 1e-9 or expected.std() < 1e-9:
            continue
        score = float(np.corrcoef(measured, expected)[0, 1])
        if not np.isfinite(score) or score <= 0.0:
            continue
        support = score * np.sqrt((count - 2) / max(1.0 - score ** 2, 1e-9))
        if support > best[0]:
            best = (support, score, float(offset), count)
    return best[1:]


def main() -> int:
    args = parse_args()
    if not (args.session / "sensor_raw.csv").is_file():
        raise SystemExit(f"{args.session} has no sensor_raw.csv")

    stamps, signal, live, total = load_sensor(args.session)
    sensor_t = (stamps - stamps[0]) / 1e9
    force_t, force_load = load_force(args.force_csv)
    duration = float(force_t[-1])

    print(f"sensor : {len(sensor_t)} frames / {sensor_t[-1]:.1f} s"
          f"   ({live}/{total} live cells)")
    print(f"force  : {len(force_t)} readings / {duration:.1f} s"
          f"   {force_load.min():.1f} to {force_load.max():.1f} N")

    score, offset, overlap = best_offset(sensor_t, signal, force_t, force_load)
    covered = overlap / 24.0
    print(f"\nbest correlation r={score:+.3f} at offset {offset:+.2f} s"
          f"   ({covered:.1f} s of the {duration:.1f} s force run overlaps)")

    if covered < 0.9 * duration:
        print(f"\nWARNING: only {covered / duration * 100:.0f}% of the force run "
              f"happened while the cameras were recording.")
        print("Start the capture FIRST, then run the force test inside that window --")
        print("force samples with no images are not usable as shape ground truth.")
    if score < args.min_correlation:
        print(f"\nrefusing: r={score:.3f} is below --min-correlation "
              f"{args.min_correlation}. These two recordings do not appear to")
        print("describe the same press. Check that the CSV belongs to this session.")
        return 1

    # Convert the offset into the anchor the real aligner expects. It computes
    # coarse_start = ctime - (duration + delay), so delay is whatever separates
    # the download from the true start of the run.
    ctime = args.force_csv.stat().st_ctime
    force_start = stamps[0] / 1e9 + offset
    delay = ctime - force_start - duration

    command = [sys.executable,
               str(Path(__file__).with_name("capture_3blackfly_sensor_force.py")),
               "--align-only", str(args.session), str(args.force_csv),
               "--force-export-delay", f"{delay:.2f}",
               "--force-search-window", str(args.search_window)]
    print(f"\n--force-export-delay {delay:.2f}  (CSV downloaded "
          f"{delay + duration:.0f} s after the run began)")
    if args.dry_run:
        print("\n" + " ".join(f'"{part}"' if " " in part else part for part in command))
        return 0
    print()
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
