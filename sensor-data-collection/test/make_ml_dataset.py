#!/usr/bin/env python3
"""Export every dead-weight session under ROOT as one frame-level table for ML.

    python make_ml_dataset.py C:/tactile_data/weight_cal [--min-marks 10] [--out DIR]

Writes ROOT/ml_dataset.csv and ROOT/ml_dataset.npz with one row per frame:

    session        capture folder name
    pad            the pad the weight was centred on (R1C1 ...), the label for "where"
    pad_row, pad_col   the same as 0-based indices
    frame_index, t_s   frame number and seconds since the session started
    mass_g         grams on the pad in this frame (0 when nothing is on), the label for "how much"
    load_N         mass_g * g / 1000
    step, rep, order   which placement this frame belongs to (0 = nothing on), its repetition and
                   the order the weights were applied in that repetition
    phase          idle / loading / hold / unloading (hold = last 5 s of a placement, the plateau)
    loaded         1 while a weight is on
    d_r1c1 .. d_r4c4   the 16 cell responses: baseline-subtracted, pressure POSITIVE, in 8-bit-
                   equivalent counts (12-bit raw / 16, baseline = median of the session's first 5 s)
    raw_r1c1 .. raw_r4c4   the 16 raw ADC values as recorded (12-bit, 0-4095; pressing LOWERS them)

The on/off edges of every placement are taken from the loaded pad's own
response (weight_cal.refine_edges), not from the key presses, so the labels
sit on the signal. Sessions with fewer than --min-marks marks are skipped
(aborted test runs). The npz holds X (N,4,4) float32 responses, y_mass (N,),
y_pad (N,) as "R1C1" strings, and the same per-frame columns as arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Python files"))
from tactile_preprocess import baseline, load_raw, response_sign, session_meta  # noqa: E402
from weight_cal import G, PLATEAU_LAST_S, intervals, read_events, refine_edges  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--min-marks", type=int, default=10, help="skip sessions with fewer on/off marks")
    p.add_argument("--settle-s", type=float, default=3.0)
    p.add_argument("--release-s", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def export_session(s: Path, args):
    meta_raw = json.loads((s / "session.json").read_text(encoding="utf-8"))
    pad = meta_raw.get("pad_under_test")
    m = re.fullmatch(r"R(\d+)C(\d+)", pad or "")
    if not m:
        return None
    events = read_events(s)
    if len(events) < args.min_marks:
        return None
    pr, pc = int(m.group(1)) - 1, int(m.group(2)) - 1
    ns, idx, frames = load_raw(s)  # 8-bit-equivalent counts
    t = (ns - ns[0]) / 1e9
    meta = session_meta(s, *frames.shape[1:])
    base = baseline(frames, t)
    sign = response_sign(frames, base, meta)
    delta = sign * (frames - base)
    bits = int(meta_raw.get("adc_bits") or 8)
    raw = np.rint(frames * (2 ** (bits - 8))).astype(int)
    last_s = float(meta_raw.get("plateau_last_s") or PLATEAU_LAST_S)
    step_info = {r["step"]: (int(r["rep"]), r["order"]) for r in events}

    n = len(t)
    mass = np.zeros(n)
    step = np.zeros(n, dtype=int)
    rep = np.zeros(n, dtype=int)
    order = np.full(n, "", dtype=object)
    phase = np.full(n, "idle", dtype=object)
    for st, mg, on_ns, off_ns in intervals(events):
        a, b = refine_edges(t, delta[:, pr, pc], (on_ns - ns[0]) / 1e9,
                            (off_ns - ns[0]) / 1e9 if off_ns else t[-1])
        on = (t >= a) & (t <= b)
        mass[on], step[on] = mg, st
        rep[on], order[on] = step_info.get(st, (0, ""))
        hold_from = max(a + args.settle_s, b - last_s)
        phase[on & (t < hold_from)] = "loading"
        phase[on & (t >= hold_from)] = "hold"
        rel = (t > b) & (t < b + args.release_s)
        phase[rel], step[rel] = "unloading", st
    rows = []
    for k in range(n):
        rows.append([s.name, pad, pr, pc, int(idx[k]), f"{t[k]:.3f}", f"{mass[k]:g}", f"{mass[k] * G / 1000:.4f}",
                     step[k], rep[k], order[k], phase[k], int(mass[k] > 0)]
                    + [f"{v:.3f}" for v in delta[k].ravel()] + raw[k].ravel().tolist())
    return rows, delta.astype(np.float32), mass, np.full(n, pad), t, phase


def main() -> None:
    args = parse_args()
    out = args.out or args.root
    sessions = sorted(d for d in args.root.glob("capture_*")
                      if (d / "weights" / "events.csv").exists() and (d / "session.json").exists())
    rows_all, X, y, pads, phases, used = [], [], [], [], [], []
    for s in sessions:
        res = export_session(s, args)
        if res is None:
            print(f"skipped {s.name}")
            continue
        rows, delta, mass, pad, t, phase = res
        rows_all += rows
        X.append(delta)
        y.append(mass)
        pads.append(pad)
        phases.append(phase)
        used.append(s.name)
        print(f"{s.name}: pad {pad[0]}, {len(mass)} frames, {int((mass > 0).sum())} loaded, "
              f"masses {sorted({int(v) for v in mass if v > 0})} g")
    if not rows_all:
        raise SystemExit("nothing exported")
    shape = X[0].shape[1:]
    cells = [f"r{r + 1}c{c + 1}" for r in range(shape[0]) for c in range(shape[1])]
    header = (["session", "pad", "pad_row", "pad_col", "frame_index", "t_s", "mass_g", "load_N", "step", "rep",
               "order", "phase", "loaded"] + [f"d_{c}" for c in cells] + [f"raw_{c}" for c in cells])
    with (out / "ml_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(header)
        w.writerows(rows_all)
    Xa, ya, pa, pha = np.concatenate(X), np.concatenate(y), np.concatenate(pads), np.concatenate(phases)
    sess = np.concatenate([np.full(len(x), n) for x, n in zip(X, used)])
    np.savez_compressed(out / "ml_dataset.npz", X=Xa, y_mass=ya.astype(np.float32), y_pad=pa.astype(str),
                        phase=pha.astype(str), session=sess.astype(str), cells=np.array(cells))
    hold = pha == "hold"
    print(f"\n{len(ya)} frames from {len(used)} sessions -> {out / 'ml_dataset.csv'} and ml_dataset.npz")
    print(f"loaded frames {int((ya > 0).sum())}, of which plateau (hold) {int(hold.sum())}; "
          f"pads {sorted(set(map(str, pa)))}; masses {sorted({int(v) for v in ya if v > 0})} g")


if __name__ == "__main__":
    main()
