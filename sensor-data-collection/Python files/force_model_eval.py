#!/usr/bin/env python3
"""How far does ML get on the force runs, and how much is the sensor itself?

    python force_model_eval.py C:/tactile_data/force_cal/capture_20261016_224016 ... --out C:/tactile_data/force_cal/model_eval

Every session needs force_alignment/frames.csv (align_force_presses.py). The
frames are cleaned with tactile_preprocess (baseline, deadband) and three
models predict the Mark-10 load from them:

    curve     one monotonic curve of the peak cell's counts (a classic calibration)
    frame     gradient boosting on the current frame around the peak cell
    history   the same plus the history features (lags, max since contact, ...)

under three splits, from easiest to hardest:

    within    5 time blocks per session; a block is predicted from everything else
    same-cell R2C7 one day -> R2C7 the other (sessions that share the peak cell)
    other     leave one session out: a cell / day the model has never seen

Errors are mean absolute errors in N on static frames (|dF/dt| below
--static-rate), so the clock residual of the alignment does not count as model
error. The hardware section measures what no model can undo: frames where the
sensor reads nothing under load (dead zone), counts per newton per force band
(the resolution one 8-bit count buys) and the loading/unloading gap at the same
force (hysteresis).

Sessions of the 16x16 M16B board and the 4x4 ESP32 board can be mixed: the
grid, the frame rate and the press direction come from each session's data.

Outputs in --out: eval.json, predictions.csv, model_eval.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from tactile_preprocess import (baseline, clean, frame_features, frame_rate, history_features, load_raw,
                                response_sign, session_meta)

BANDS = [(0.1, 0.8, "<0.8 N"), (0.8, 3.0, "0.8-3 N"), (3.0, 7.0, "3-7 N"), (7.0, 40.0, ">7 N")]
MODELS = ("curve", "frame", "history")
COLORS = {"curve": "#2a78d6", "frame": "#eb6834", "history": "#1baf7a"}  # categorical slots 1-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("sessions", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--static-rate", type=float, default=0.15, help="N/s; slower counts as static")
    p.add_argument("--deadband", type=float, default=4.0)
    p.add_argument("--blocks", type=int, default=5)
    return p.parse_args()


def load_session(session: Path, deadband: float, static_rate: float):
    ns, idx, frames = load_raw(session)
    t = (ns - ns[0]) / 1e9
    base = baseline(frames, t)
    sign = response_sign(frames, base, session_meta(session, *frames.shape[1:]))
    delta = clean(frames, base, deadband, sign=sign)  # pressure positive on either board
    feats = frame_features(delta)
    feats.update(history_features(feats["peak"], frame_rate(ns)))
    row_of = {f: i for i, f in enumerate(idx)}
    with (session / "force_alignment" / "frames.csv").open(newline="") as handle:
        labels = [r for r in csv.DictReader(handle) if r["load_N"] not in ("", "nan")]
    rows = np.array([row_of[int(r["frame_index"])] for r in labels])
    load = np.array([float(r["load_N"]) for r in labels])
    tl = t[rows]
    rate = np.gradient(load) / np.maximum(np.gradient(tl), 1e-3)  # frames can share a host timestamp
    rate = np.convolve(rate, np.ones(7) / 7, mode="same")
    recent_max = np.array([load[(tl >= x - 60) & (tl <= x)].max() for x in tl])
    return {
        "name": session.name.replace("capture_", ""),
        "t": tl,
        "load": load,
        "static": np.abs(rate) < static_rate,
        "unloading_side": load < 0.9 * recent_max,
        "X": {k: v[rows] for k, v in feats.items()},
        "peak_cell": np.unravel_index(np.abs(delta).max(axis=0).argmax(), delta.shape[1:]),
        "grid": delta.shape[1:],
    }


def design(sessions, names):
    return np.column_stack([np.concatenate([s["X"][n] for s in sessions]) for n in names])


def fit_predict(train, test, frame_names, hist_names):
    y = np.concatenate([s["load"] for s in train])
    out = {}
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(np.concatenate([s["X"]["peak"] for s in train]), y)
    out["curve"] = iso.predict(np.concatenate([s["X"]["peak"] for s in test]))
    for model, names in (("frame", frame_names), ("history", frame_names + hist_names)):
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=40, random_state=0)
        gbm.fit(design(train, names), y)
        out[model] = np.clip(gbm.predict(design(test, names)), 0, None)
    return out


def blocks_of(s, n):
    edges = np.linspace(0, len(s["load"]), n + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        keep = np.zeros(len(s["load"]), bool)
        keep[a:b] = True
        yield keep


def subset(s, keep):
    return {**s, "t": s["t"][keep], "load": s["load"][keep], "static": s["static"][keep],
            "unloading_side": s["unloading_side"][keep], "X": {k: v[keep] for k, v in s["X"].items()}}


def run_splits(sessions, frame_names, hist_names, n_blocks):
    """Yield (split, test session part, predictions) for every fold."""
    for i, s in enumerate(sessions):
        for keep in blocks_of(s, n_blocks):
            train = [o for j, o in enumerate(sessions) if j != i] + [subset(s, ~keep)]
            test = subset(s, keep)
            yield "within", test, fit_predict(train, [test], frame_names, hist_names)
    by_cell: dict[tuple, list] = {}
    for s in sessions:
        by_cell.setdefault(s["peak_cell"], []).append(s)
    for group in by_cell.values():
        for s in group if len(group) > 1 else ():
            train = [o for o in group if o is not s]
            yield "same-cell", s, fit_predict(train, [s], frame_names, hist_names)
    for s in sessions:
        train = [o for o in sessions if o is not s]
        yield "other", s, fit_predict(train, [s], frame_names, hist_names)


def mae_table(records):
    """records: list of dicts with split, load, static, pred[model]. MAE on static, loaded frames."""
    table = {}
    for split in ("within", "same-cell", "other"):
        rs = [r for r in records if r["split"] == split]
        if not rs:
            continue
        y = np.concatenate([r["load"] for r in rs])
        m = np.concatenate([r["static"] for r in rs]) & (y > BANDS[0][0])
        table[split] = {}
        for model in MODELS:
            p = np.concatenate([r["pred"][model] for r in rs])
            row = {"all": float(np.mean(np.abs(p - y)[m])), "n": int(m.sum())}
            for lo, hi, label in BANDS:
                b = m & (y >= lo) & (y < hi)
                row[label] = float(np.mean(np.abs(p - y)[b])) if b.sum() >= 10 else None
            table[split][model] = row
    return table


def hardware(sessions):
    """Dead zone, counts per newton and hysteresis, per session, static frames only."""
    report = {}
    for s in sessions:
        y, peak, st, down = s["load"], s["X"]["peak"], s["static"], s["unloading_side"]
        r, c = s["peak_cell"]
        entry = {"peak_cell": f"R{r + 1}C{c + 1}"}
        loaded = st & (y >= 0.3)
        dead = loaded & (peak == 0)
        entry["dead_zone"] = {
            "frames_loaded": int(loaded.sum()), "frames_reading_zero": int(dead.sum()),
            "max_force_reading_zero_N": float(y[dead].max()) if dead.any() else None,
            "min_force_reading_contact_N": float(y[st & (peak > 0) & ~down].min()) if (st & (peak > 0) & ~down).any() else None,
        }
        levels = []
        up = st & ~down & (peak > 0)
        for lo, hi in ((0.8, 1.5), (1.5, 3), (3, 5), (5, 8), (8, 12), (12, 25)):
            b = up & (y >= lo) & (y < hi)
            if b.sum() >= 15:
                levels.append((float(np.median(y[b])), float(np.median(peak[b]))))
        slopes = []
        for (f0, p0), (f1, p1) in zip(levels, levels[1:]):
            slope = (p1 - p0) / (f1 - f0)
            slopes.append({"from_N": round(f0, 2), "to_N": round(f1, 2), "counts_per_N": round(slope, 1),
                           "N_per_8bit_count": round(1 / slope, 3) if slope > 0 else None,
                           "N_per_10bit_count": round(0.25 / slope, 3) if slope > 0 else None})
        entry["sensitivity"] = slopes
        gaps = []
        for lo, hi in ((0.4, 0.8), (0.8, 1.5), (1.5, 3), (3, 8), (8, 25)):
            a, d = st & ~down & (y >= lo) & (y < hi), st & down & (y >= lo) & (y < hi)
            if a.sum() >= 10 and d.sum() >= 10:
                gaps.append({"band_N": f"{lo}-{hi}", "loading_counts": float(np.median(peak[a])),
                             "unloading_counts": float(np.median(peak[d]))})
        entry["hysteresis"] = gaps
        report[s["name"]] = entry
    return report


def plot(path, table, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [s for s in ("within", "same-cell", "other") if s in table]
    fig, axes = plt.subplots(1, len(splits) + 1, figsize=(4.2 * (len(splits) + 1), 4.2))
    labels = [b[2] for b in BANDS]
    x = np.arange(len(labels))
    top = max(v for sp in table.values() for row in sp.values() for k, v in row.items()
              if k in labels and v is not None)
    for ax, split in zip(axes, splits):
        for k, model in enumerate(MODELS):
            vals = [table[split][model][lab] or 0 for lab in labels]
            ax.bar(x + (k - 1) * 0.26, vals, 0.24, color=COLORS[model], label=model)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, top * 1.1)
        ax.set_title(f"{split}: MAE by force band")
        ax.set_ylabel("mean |error| (N)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    ax = axes[-1]
    rs = [r for r in records if r["split"] == ("same-cell" if "same-cell" in table else "other")]
    y = np.concatenate([r["load"] for r in rs])
    m = np.concatenate([r["static"] for r in rs])
    for model in ("curve", "history"):
        p = np.concatenate([r["pred"][model] for r in rs])
        ax.scatter(y[m], p[m], s=8, alpha=0.35, color=COLORS[model], label=model, linewidths=0)
    lim = float(y.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#888", lw=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mark-10 load (N)", ylabel="predicted (N)",
           title=f"{rs[0]['split']}: predicted vs true")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    sessions = [load_session(s, args.deadband, args.static_rate) for s in args.sessions]
    frame_names = list(frame_features(np.zeros((1, 4, 4))))  # feature names only; any grid gives the same
    hist_names = list(history_features(np.zeros(1)))
    records = []
    for split, test, pred in run_splits(sessions, frame_names, hist_names, args.blocks):
        records.append({"split": split, "session": test["name"], "t": test["t"], "load": test["load"],
                        "static": test["static"], "pred": {k: v for k, v in pred.items()}})
    table = mae_table(records)
    hw = hardware(sessions)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval.json").write_text(json.dumps({"mae_static_N": table, "hardware": hw,
                                                    "sessions": [s["name"] for s in sessions],
                                                    "grids": {s["name"]: list(map(int, s["grid"])) for s in sessions}},
                                                   indent=2))
    with (args.out / "predictions.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["split", "session", "t_s", "load_N", "static", *[f"pred_{m}_N" for m in MODELS]])
        for r in records:
            for k in range(len(r["load"])):
                w.writerow([r["split"], r["session"], f"{r['t'][k]:.3f}", f"{r['load'][k]:.3f}", int(r["static"][k]),
                            *[f"{r['pred'][m][k]:.3f}" for m in MODELS]])
    plot(args.out / "model_eval.png", table, records)

    for split, rows in table.items():
        print(f"\n{split}  (MAE in N, static frames with load > 0.1 N)")
        print(f"  {'model':8s} {'all':>6s} " + " ".join(f"{b[2]:>8s}" for b in BANDS) + "      n")
        for model, row in rows.items():
            cells = " ".join(f"{row[b[2]]:8.2f}" if row[b[2]] is not None else f"{'-':>8s}" for b in BANDS)
            print(f"  {model:8s} {row['all']:6.2f} {cells} {row['n']:6d}")
    print("\nhardware")
    for name, entry in hw.items():
        dz = entry["dead_zone"]
        print(f"  {name} {entry['peak_cell']}: {dz['frames_reading_zero']}/{dz['frames_loaded']} static frames >= 0.3 N "
              f"read zero (up to {dz['max_force_reading_zero_N']} N); first contact on loading at "
              f"{dz['min_force_reading_contact_N']} N")
        for sl in entry["sensitivity"]:
            print(f"    {sl['from_N']:5.2f}->{sl['to_N']:5.2f} N: {sl['counts_per_N']:6.1f} counts/N, "
                  f"{sl['N_per_8bit_count']} N per 8-bit count, {sl['N_per_10bit_count']} N per 10-bit count")
        for g in entry["hysteresis"]:
            print(f"    hysteresis {g['band_N']:>8s} N: loading {g['loading_counts']:5.0f}, unloading {g['unloading_counts']:5.0f} counts")
    print(f"\nwrote {args.out / 'eval.json'}, predictions.csv, model_eval.png")


if __name__ == "__main__":
    args = parse_args()
    main()
