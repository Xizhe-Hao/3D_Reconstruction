#!/usr/bin/env python3
"""One-minute check of a bench run: tactile response, force curve, contact map.

Bench runs are captured with --no-save-images, so there are no MP4s and
replay_capture_2d.py cannot open them. This reads only sensor_raw.csv and the
raw IntelliMESUR export in <session>/force/, and answers the questions to ask
before moving to the next run:

  * did the sensor run at the expected rate, without dropped frames?
  * is the unloaded baseline quiet, and did it come back after the run?
  * did every press show up?
  * is the contact centred on the intended pad?
  * was the force gauge tared, and is its calibration still valid?

The two streams are drawn on their own clocks; aligning them is a separate step.

Usage:
    python bench_quicklook.py <capture session folder>
    python bench_quicklook.py <session> --force <export.csv> --show
    python bench_quicklook.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROWS = COLS = 16
PITCH_MM = 7.5
EXPECTED_FPS = 6.0  # the rig runs at 6 fps on purpose: 24 fps video is too large for the reconstruction
TAP_MAX_S = 4.0  # a sync tap (4 mm, 1 s hold at 300 mm/min) touches for ~2.6 s
TAP_PAIR_GAP_S = 6.0  # taps are ~3 s apart; normal presses rest 15 s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("session", nargs="?", type=Path, help="capture session folder")
    parser.add_argument("--force", type=Path, help="IntelliMESUR export; default: the raw export in <session>/force/")
    parser.add_argument("--baseline-s", type=float, default=3.0, help="unloaded seconds at the start")
    parser.add_argument("--out", type=Path, help="PNG path; default <session>/bench_quicklook.png")
    parser.add_argument("--show", action="store_true", help="also open the figure")
    parser.add_argument("--selftest", action="store_true", help="synthetic run, headless")
    return parser.parse_args()


# ---------------------------------------------------------------- loading


def load_sensor(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Device time (s), frame index, and ADC frames (n, 16, 16) from sensor_raw.csv."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        cells = [header.index(f"r{r}c{c}_adc8") for r in range(1, ROWS + 1) for c in range(1, COLS + 1)]
        i_ms, i_idx = header.index("device_millis"), header.index("frame_index")
        millis, index, frames = [], [], []
        for row in reader:
            if not row:
                continue
            millis.append(float(row[i_ms]))
            index.append(int(row[i_idx]))
            frames.append([float(row[i]) for i in cells])
    millis = np.asarray(millis)
    return (millis - millis[0]) / 1000.0, np.asarray(index), np.asarray(frames).reshape(-1, ROWS, COLS)


def find_force_export(session: Path) -> Path | None:
    """The raw IntelliMESUR export (starts with 'sep='), not the derived alignment CSVs."""
    candidates = []
    for path in sorted((session / "force").glob("*.csv")):
        try:
            with path.open(encoding="utf-8-sig", errors="replace") as handle:
                if handle.readline().startswith("sep="):
                    candidates.append(path)
        except OSError:
            continue
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def load_force(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Header fields, time (s), load (N), distance (mm) from an IntelliMESUR export."""
    meta, data = {}, []
    in_table = False
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            fields = [f.strip() for f in line.rstrip("\r\n").split("\t")]
            if not in_table:
                if fields[0] == "Reading":
                    columns = fields
                    in_table = True
                elif ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip()
                continue
            if len(fields) >= 4 and fields[0]:
                try:
                    data.append([float(f) for f in fields[:4]])
                except ValueError:
                    continue
    if not data:
        raise RuntimeError(f"no data rows in {path}")
    data = np.asarray(data)
    col = {name.split(" [")[0]: i for i, name in enumerate(columns) if name}
    return meta, data[:, col["Time"]], data[:, col["Load"]], data[:, col["Distance"]]


# ---------------------------------------------------------------- analysis


def segments(active: np.ndarray, t: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [start, stop) where active is True."""
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b > a]


def double_taps(segs: list[tuple[int, int]], t: np.ndarray) -> list[tuple[float, float]]:
    """Start times of pairs of short presses close together."""
    taps = [(t[a], t[min(b, len(t)) - 1]) for a, b in segs if t[min(b, len(t)) - 1] - t[a] < TAP_MAX_S]
    pairs, i = [], 0
    while i < len(taps) - 1:
        if taps[i + 1][0] - taps[i][1] < TAP_PAIR_GAP_S:
            pairs.append((taps[i][0], taps[i + 1][1]))
            i += 2
        else:
            i += 1
    return pairs


def analyse_sensor(t: np.ndarray, index: np.ndarray, frames: np.ndarray, baseline_s: float) -> dict:
    dt = np.diff(t)
    fps = 1.0 / float(np.median(dt)) if dt.size else float("nan")
    dropped = int(np.sum(np.diff(index) - 1)) if index.size > 1 else 0
    base_mask = t <= baseline_s
    if base_mask.sum() < 5:
        base_mask = np.arange(len(t)) < max(5, len(t) // 20)
    baseline = np.median(frames[base_mask], axis=0)
    delta = frames - baseline
    # The press direction depends on the wiring: pick the sign of the biggest excursion.
    sign = 1.0 if np.percentile(delta.sum(axis=(1, 2)), 99.5) >= -np.percentile(delta.sum(axis=(1, 2)), 0.5) else -1.0
    response = np.clip(sign * delta, 0, None)
    total = response.sum(axis=(1, 2))
    noise = float(np.std(total[base_mask]))
    level = float(np.mean(total[base_mask]))
    threshold = level + max(5.0 * noise, 0.02 * float(total.max() - level), 1.0)
    segs = segments(total > threshold, t)
    peak = int(np.argmax(total))
    weights = response[peak]
    rows, cols = np.mgrid[0:ROWS, 0:COLS]
    if weights.sum() > 0:
        centroid_rc = (float((rows * weights).sum() / weights.sum()), float((cols * weights).sum() / weights.sum()))
    else:
        centroid_rc = (float("nan"), float("nan"))
    peak_cell = np.unravel_index(int(np.argmax(weights)), weights.shape)
    end_mask = t >= t[-1] - baseline_s
    return {
        "fps": fps, "dropped": dropped, "frames": len(t), "duration_s": float(t[-1]),
        "total": total, "threshold": threshold, "noise": noise, "segments": segs,
        "double_taps": double_taps(segs, t), "peak_index": peak, "peak_map": weights,
        "centroid_rc": centroid_rc, "peak_cell": (int(peak_cell[0]), int(peak_cell[1])),
        "end_drift": float(np.mean(total[end_mask]) - level), "sign": sign,
    }


def median_filter(x: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or len(x) < window:
        return x
    half = window // 2
    padded = np.pad(x, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1), axis=1)


def analyse_force(t: np.ndarray, load: np.ndarray, distance: np.ndarray, meta: dict) -> dict:
    start_load = float(np.median(load[: max(5, len(load) // 100)]))
    # The crosshead's starts and stops put +-3-5 N spikes on the load; a 0.15 s median removes them.
    rate = 1.0 / float(np.median(np.diff(t)))
    excursion = median_filter(load, int(round(0.15 * rate)) | 1) - start_load
    sign = 1.0 if excursion.max() >= -excursion.min() else -1.0
    excursion = sign * excursion
    magnitude = np.abs(load - start_load)
    segs = segments(excursion > max(1.0, 0.15 * float(excursion.max())), t)
    due = meta.get("Calibration Due Date")
    expired = None
    if due:
        try:
            expired = datetime.strptime(due, "%b %d, %Y").date() < date.today()
        except ValueError:
            pass
    return {
        "start_load": start_load, "end_load": float(np.median(load[-max(5, len(load) // 100):])),
        "peak_load": float(load[np.argmax(magnitude)]), "travel": float(distance.max() - distance.min()),
        "rate_hz": rate, "segments": segs,
        "double_taps": double_taps(segs, t), "cal_due": due, "cal_expired": expired,
        "sensor": meta.get("Force Sensor Model"), "duration_s": float(t[-1] - t[0]),
    }


# ---------------------------------------------------------------- report


def report_lines(s: dict, f: dict | None) -> list[str]:
    r, c = s["centroid_rc"]
    pr, pc = s["peak_cell"]
    lines = [
        f"tactile: {s['frames']} frames, {s['duration_s']:.1f} s, {s['fps']:.1f} fps, dropped {s['dropped']}",
        f"  baseline noise (sum) {s['noise']:.1f}, end-of-run drift {s['end_drift']:+.1f} "
        f"({s['end_drift'] / max(s['noise'], 1e-9):+.1f} sigma)",
        f"  presses {len(s['segments'])}, sync double-taps {len(s['double_taps'])}",
        f"  peak cell R{pr + 1}C{pc + 1}; centroid R{r + 1:.2f} C{c + 1:.2f} "
        f"= (u, v) ({(c + 0.5) * PITCH_MM:.1f}, {(r + 0.5) * PITCH_MM:.1f}) mm",
    ]
    if f is not None:
        lines += [
            f"force: {f['sensor']}, {f['duration_s']:.1f} s at {f['rate_hz']:.0f} Hz, peak {f['peak_load']:+.1f} N, "
            f"travel {f['travel']:.2f} mm",
            f"  load at start {f['start_load']:+.1f} N, at end {f['end_load']:+.1f} N",
            f"  presses {len(f['segments'])}, sync double-taps {len(f['double_taps'])}",
        ]
    return lines


def warnings(s: dict, f: dict | None) -> list[str]:
    out = []
    if abs(s["fps"] - EXPECTED_FPS) > 1.0:
        out.append(f"tactile rate {s['fps']:.1f} fps, expected {EXPECTED_FPS:g} (the rig is meant to run at this rate)")
    if s["dropped"]:
        out.append(f"{s['dropped']} tactile frames dropped")
    if abs(s["end_drift"]) > 5 * max(s["noise"], 1e-9):
        out.append("baseline did not return after the run (drift > 5 sigma): creep or the sample moved")
    if f is not None:
        if abs(f["start_load"]) > 0.3:
            out.append(f"force not tared: {f['start_load']:+.1f} N before the run")
        if f["cal_expired"]:
            out.append(f"force sensor calibration expired ({f['cal_due']})")
    else:
        out.append("no IntelliMESUR export in <session>/force/")
    return out


def plot(path: Path, t, s: dict, force, f: dict | None, title: str, show: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 3, width_ratios=[2.2, 2.2, 1.4])
    ax_s = fig.add_subplot(grid[0, :2])
    ax_f = fig.add_subplot(grid[1, :2])
    ax_m = fig.add_subplot(grid[0, 2])
    ax_t = fig.add_subplot(grid[1, 2])

    ax_s.plot(t, s["total"], lw=0.8, color="tab:blue")
    ax_s.axhline(s["threshold"], color="0.6", lw=0.6, ls="--")
    for a, b in s["segments"]:
        ax_s.axvspan(t[a], t[min(b, len(t)) - 1], color="tab:blue", alpha=0.08)
    for a, b in s["double_taps"]:
        ax_s.axvspan(a, b, color="tab:orange", alpha=0.3)
    ax_s.set(title="tactile response (sum over cells, ADC counts; orange = sync double-tap)",
             xlabel="sensor device time [s]", ylabel="response")

    if f is not None:
        ft, load, dist = force
        ax_f.plot(ft, load, lw=0.8, color="tab:red")
        for a, b in f["double_taps"]:
            ax_f.axvspan(a, b, color="tab:orange", alpha=0.3)
        ax_f.set(xlabel="force export time [s] (own clock, not aligned)", ylabel="load [N]")
        ax_d = ax_f.twinx()
        ax_d.plot(ft, dist, lw=0.6, color="0.4")
        ax_d.set_ylabel("crosshead distance [mm]")
    else:
        ax_f.text(0.5, 0.5, "no force export", ha="center", va="center", transform=ax_f.transAxes)

    image = ax_m.imshow(s["peak_map"], cmap="magma", origin="upper")
    r, c = s["centroid_rc"]
    ax_m.plot(c, r, "o", mfc="none", mec="cyan", ms=10)
    ax_m.set_xticks([0, 7, 15], ["C1", "C8", "C16"])
    ax_m.set_yticks([0, 7, 15], ["R1", "R8", "R16"])
    ax_m.set_title(f"peak frame at {t[s['peak_index']]:.1f} s (o = centroid)")
    fig.colorbar(image, ax=ax_m, fraction=0.046)

    ax_t.axis("off")
    text = report_lines(s, f)
    warns = warnings(s, f)
    text += ["", "WARNINGS:" if warns else "no warnings"] + [f"- {w}" for w in warns]
    ax_t.text(0, 1, "\n".join(text), va="top", family="monospace", fontsize=7, wrap=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- entry points


def run(session: Path, force_path: Path | None, baseline_s: float, out: Path | None, show: bool) -> int:
    t, index, frames = load_sensor(session / "sensor_raw.csv")
    s = analyse_sensor(t, index, frames, baseline_s)
    force_path = force_path or find_force_export(session)
    f = force = None
    if force_path is not None:
        meta, ft, load, dist = load_force(force_path)
        force = (ft, load, dist)
        f = analyse_force(ft, load, dist, meta)
    for line in report_lines(s, f):
        print(line)
    warns = warnings(s, f)
    for w in warns:
        print(f"WARNING: {w}")
    plot(out or session / "bench_quicklook.png", t, s, force, f, session.name, show)
    return 1 if warns else 0


def selftest() -> int:
    """Synthetic 6 fps run: sync double-taps, three presses on R8C8, matching force export."""
    import tempfile

    rng = np.random.default_rng(0)
    fps, duration = 6.0, 80.0
    t = np.arange(0, duration, 1 / fps)

    def profile(times):
        load = np.zeros_like(times)
        for start in (4.0, 7.0, 70.0, 73.0):  # double-taps
            load += 15 * np.exp(-0.5 * ((times - start - 0.8) / 0.4) ** 2)
        for start in (15.0, 30.0, 45.0):  # presses with a hold
            load += 30 * np.clip(np.minimum(times - start, start + 5 - times), 0, 1)
        return load

    load = profile(t)
    rows, cols = np.mgrid[0:ROWS, 0:COLS]
    footprint = np.exp(-(((rows - 7) ** 2 + (cols - 7) ** 2) / 2.0))
    frames = 200 - (load[:, None, None] * footprint * 3) + rng.normal(0, 0.7, (len(t), ROWS, COLS))
    frames = np.clip(np.round(frames), 0, 255)

    with tempfile.TemporaryDirectory() as tmp:
        session = Path(tmp) / "capture_selftest"
        (session / "force").mkdir(parents=True)
        with (session / "sensor_raw.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["host_received_ns", "frame_index", "device_millis"]
                            + [f"r{r}c{c}_adc8" for r in range(1, 17) for c in range(1, 17)])
            for i, (ti, frame) in enumerate(zip(t, frames)):
                writer.writerow([int(ti * 1e9), i, int(ti * 1000) + 31] + frame.astype(int).ravel().tolist())
        ft = np.arange(0, duration, 0.01)
        with (session / "force" / "export.csv").open("w") as handle:
            handle.write("sep=\t\nselftest\n\nSystem Information:\nForce Sensor Model: FS05-50\n"
                         "Calibration Due Date: Jan 1, 2099\n\nReading\tLoad [N]\tDistance [mm]\tTime [s]\t\n")
            for i, (ti, li) in enumerate(zip(ft, profile(ft)), 1):
                handle.write(f"{i}\t{li:.1f}\t{100 - li / 5:.2f}\t{ti:.3f}\t\n")
        s = analyse_sensor(*load_sensor(session / "sensor_raw.csv"), 3.0)
        meta, ft2, load2, dist2 = load_force(session / "force" / "export.csv")
        f = analyse_force(ft2, load2, dist2, meta)
        plot(Path(tmp) / "q.png", t, s, (ft2, load2, dist2), f, "selftest", False)

    checks = {
        "fps": abs(s["fps"] - fps) < 0.5,
        "presses": len(s["segments"]) == 7,
        "tactile double-taps": len(s["double_taps"]) == 2,
        "force double-taps": len(f["double_taps"]) == 2,
        "peak cell R8C8": s["peak_cell"] == (7, 7),
        "no warnings": not warnings(s, f),
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(checks.values()) else 1


def main() -> int:
    args = parse_args()
    if args.selftest:
        return selftest()
    if args.session is None:
        print("give a capture session folder (or --selftest)")
        return 2
    return run(args.session, args.force, args.baseline_s, args.out, args.show)


if __name__ == "__main__":
    sys.exit(main())
