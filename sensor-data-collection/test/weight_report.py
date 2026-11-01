#!/usr/bin/env python3
"""Report of a dead-weight calibration session recorded with weight_cal.py.

    python weight_report.py <session> [--settle-s 3] [--out DIR]

For every step (one weight on the pad): the summed response of the array and
the pad's own plateau level in 8-bit-equivalent counts, the footprint (how
many cells the weight touched and what share the pad carried), noise, creep
over the hold, time to settle, and the residual after the weight came off.
Per mass: repeatability across the repetitions, separation from the
neighbouring masses, counts per newton and the force resolution that follows
from the noise. Hysteresis is the plateau difference between steps that
followed a heavier weight and steps that followed a lighter one.

The weights sit directly on the pad, so a wide base can also load the
neighbours: the quantity that is known exactly is the total load, and the
primary calibration is therefore the SUMMED response -> N. The pad-only curve
is scored alongside. Both are monotone fits trained on all but one repetition
and scored on the held-out one, which is the number that says how well dead
weights can calibrate this pad.

Needs <session>/sensor_raw.csv, session.json and weights/events.csv.
Writes <session>/report/weight_report.png, weight_steps.csv, weight_report.json.
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
from tactile_preprocess import baseline, frame_rate, load_raw, response_sign, session_meta  # noqa: E402
from weight_cal import G, SPREAD_COUNTS, intervals, read_events, refine_edges  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--settle-s", type=float, default=3.0, help="seconds after 'on' never counted as plateau")
    p.add_argument("--last-s", type=float, default=None,
                   help="plateau = the last seconds of each hold (default: session.json plateau_last_s, else 5)")
    p.add_argument("--tail-s", type=float, default=2.0, help="seconds at each end of the hold used for creep")
    p.add_argument("--platen-g", type=float, default=None,
                   help="override the tare mass (platen) in session.json; 0 when weights sit directly on the pad")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def window(t, a, b):
    return np.flatnonzero((t >= a) & (t < b))


def main() -> None:
    args = parse_args()
    s = args.session
    ns, idx, frames = load_raw(s)
    t = (ns - ns[0]) / 1e9
    fps = frame_rate(ns)
    meta = session_meta(s, *frames.shape[1:])
    raw_meta = json.loads((s / "session.json").read_text(encoding="utf-8")) if (s / "session.json").exists() else {}
    platen_g = args.platen_g if args.platen_g is not None else float(raw_meta.get("platen_g") or 0.0)
    last_s = args.last_s if args.last_s is not None else float(raw_meta.get("plateau_last_s") or 5.0)
    base = baseline(frames, t)
    sign = response_sign(frames, base, meta)
    delta = sign * (frames - base)  # pressure positive, no deadband: 0.1 N must stay visible
    steps = intervals(read_events(s))
    if not steps:
        raise SystemExit("weights/events.csv has no on/off pairs")
    end_t = t[-1] + 1

    def hold_window(t_on, t_off):
        return window(t, max(t_on + args.settle_s, t_off - last_s), t_off)

    # the loaded pad: what the session says (weight_cal --pads), else the cell with the largest plateau response
    hold_idx = []
    for step, mass, on_ns, off_ns in steps:
        t_on = (on_ns - ns[0]) / 1e9
        t_off = (off_ns - ns[0]) / 1e9 if off_ns else end_t
        hold_idx.extend(hold_window(t_on, t_off))
    mean_map = delta[np.array(hold_idx, dtype=int)].mean(axis=0) if hold_idx else delta.mean(axis=0)
    peak_rc = np.unravel_index(int(mean_map.argmax()), mean_map.shape)
    stated = raw_meta.get("pad_under_test")
    m = re.fullmatch(r"R(\d+)C(\d+)", stated or "")
    if m and 0 <= int(m.group(1)) - 1 < delta.shape[1] and 0 <= int(m.group(2)) - 1 < delta.shape[2]:
        pr, pc = int(m.group(1)) - 1, int(m.group(2)) - 1
    else:
        pr, pc = peak_rc
    pad_name = f"R{pr + 1}C{pc + 1}"
    peak_name = f"R{peak_rc[0] + 1}C{peak_rc[1] + 1}"
    pad = delta[:, pr, pc]
    total = delta.sum(axis=(1, 2))

    # per-step table; the on/off edges come from the pad's own response, the marks only bound the search
    rows = []
    prev_mass = None
    marks = [((on_ns - ns[0]) / 1e9, (off_ns - ns[0]) / 1e9 if off_ns else end_t) for _, _, on_ns, off_ns in steps]
    edges = [refine_edges(t, pad, a, b) for a, b in marks]
    mark_shift = [(e[0] - m[0], e[1] - m[1]) for e, m in zip(edges, marks)]
    for k, (step, mass, on_ns, off_ns) in enumerate(steps):
        t_on, t_off = edges[k]
        next_on = edges[k + 1][0] if k + 1 < len(steps) else end_t
        hold = hold_window(t_on, t_off)
        if hold.size < 3:
            prev_mass = mass
            continue
        load_n = (mass + platen_g) * G / 1000.0
        level = float(np.median(pad[hold]))
        head = window(t, t_on + args.settle_s, t_on + args.settle_s + args.tail_s)
        tail = window(t, t_off - args.tail_s, t_off)
        creep = float(np.median(pad[tail]) - np.median(pad[head])) if head.size and tail.size else np.nan
        rise = window(t, t_on, t_off)
        reached = rise[pad[rise] >= 0.9 * level] if level > 0 else rise[:1]
        settle = float(t[reached[0]] - t_on) if reached.size else np.nan
        after = window(t, t_off + 2.0, min(t_off + 8.0, next_on))
        residual = float(np.median(pad[after])) if after.size else np.nan
        hold_map = delta[hold].mean(axis=0)
        foot = hold_map > SPREAD_COUNTS
        foot[pr, pc] = True  # the pad is always part of its own footprint
        cells = int((hold_map > SPREAD_COUNTS).sum())
        w = np.where(foot, hold_map, 0.0)
        rr, cc = np.indices(hold_map.shape)
        centroid = f"R{(w * rr).sum() / w.sum() + 1:.2f}C{(w * cc).sum() / w.sum() + 1:.2f}" if w.sum() > 0 else ""
        share = float(hold_map[pr, pc] / w.sum()) if w.sum() > 0 else np.nan
        # summed response = the footprint cells only; idle cells would add their drift and noise
        foot_sum = float(np.median(delta[hold][:, foot].sum(axis=1)))
        rows.append({
            "step": step, "mass_g": mass, "load_N": load_n, "t_on_s": t_on, "hold_s": t_off - t_on,
            "pad_counts": level, "pad_noise": float(pad[hold].std()), "sum_counts": foot_sum,
            "cells": cells, "pad_share": share, "centroid": centroid,
            "creep_counts": creep, "settle_s": settle, "residual_after_counts": residual,
            "prev_mass_g": prev_mass, "frames": int(hold.size),
        })
        prev_mass = mass

    # unloaded noise and drift from the gaps between steps (refined edges, so the weight is really off)
    gaps = []
    for k in range(len(steps) - 1):
        gaps.append(window(t, edges[k][1] + 3.0, edges[k + 1][0] - 1.0))
    gap_idx = np.concatenate([g for g in gaps if g.size]) if gaps else np.arange(0)
    # robust (MAD) so the hand reaching in to place the next weight does not count as noise
    unl = pad[gap_idx] if gap_idx.size > 5 else pad[t <= 5]
    noise = float(1.4826 * np.median(np.abs(unl - np.median(unl))))
    drift = (float(np.median(pad[gaps[-1]]) - np.median(pad[gaps[0]]))
             if len(gaps) >= 2 and gaps[0].size and gaps[-1].size else np.nan)

    # per-mass summary
    masses = sorted({r["mass_g"] for r in rows})
    per_mass = []
    for m in masses:
        rs = [r for r in rows if r["mass_g"] == m]
        lv = np.array([r["sum_counts"] for r in rs])  # primary: summed response
        pv = np.array([r["pad_counts"] for r in rs])
        heavier = [r["sum_counts"] for r in rs if r["prev_mass_g"] is not None and r["prev_mass_g"] > m]
        lighter = [r["sum_counts"] for r in rs if r["prev_mass_g"] is not None and r["prev_mass_g"] < m]
        per_mass.append({
            "mass_g": m, "load_N": rs[0]["load_N"], "n": len(rs), "mean": float(lv.mean()),
            "std_rep": float(lv.std(ddof=1)) if len(lv) > 1 else np.nan,
            "pad_mean": float(pv.mean()), "pad_std_rep": float(pv.std(ddof=1)) if len(pv) > 1 else np.nan,
            "cells_mean": float(np.mean([r["cells"] for r in rs])),
            "pad_share_mean": float(np.nanmean([r["pad_share"] for r in rs])),
            "creep_mean": float(np.nanmean([r["creep_counts"] for r in rs])),
            "hysteresis": (float(np.mean(heavier) - np.mean(lighter)) if heavier and lighter else np.nan),
        })
    for a, b in zip(per_mass, per_mass[1:]):
        gap = b["mean"] - a["mean"]
        spread = np.hypot(np.nan_to_num(a["std_rep"], nan=noise), np.nan_to_num(b["std_rep"], nan=noise))
        b["counts_per_N"] = gap / (b["load_N"] - a["load_N"]) if b["load_N"] > a["load_N"] else np.nan
        b["separation_sigma"] = gap / spread if spread > 0 else np.inf
        b["force_resolution_N"] = noise / b["counts_per_N"] if b.get("counts_per_N") else np.nan
    reps = {}
    for r in rows:  # repetition = how many times this mass has been seen so far
        reps[r["step"]] = sum(1 for q in rows if q["mass_g"] == r["mass_g"] and q["step"] <= r["step"])

    # held-out repetition: monotone counts -> N curve
    def iso_predict(train_x, train_y, x):
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(train_x, train_y)
        return iso.predict(x)

    maes = {}
    for key, col in (("sum", "sum_counts"), ("pad", "pad_counts")):
        errs = []
        for rep in sorted(set(reps.values())):
            test = [r for r in rows if reps[r["step"]] == rep]
            train = [r for r in rows if reps[r["step"]] != rep]
            if len(train) < 3 or not test:
                continue
            pred = iso_predict([r[col] for r in train], [r["load_N"] for r in train], [r[col] for r in test])
            for r, p in zip(test, pred):
                r[f"pred_N_{key}"] = float(p)
                errs.append(p - r["load_N"])
        maes[key] = float(np.abs(np.array(errs)).mean()) if errs else np.nan
    mae = maes["sum"]
    span = (max(r["load_N"] for r in rows) - min(r["load_N"] for r in rows)) if rows else np.nan
    monotone = all(b["mean"] > a["mean"] for a, b in zip(per_mass, per_mass[1:]))
    spread = [m for m in per_mass if m["pad_share_mean"] < 0.7]

    # ---- print
    print(f"{s.name}: {len(rows)} steps, {len(masses)} masses, pad {pad_name}"
          + (f" (but the largest response is on {peak_name})" if peak_name != pad_name else "")
          + f", plateau = last {last_s:g} s, {fps:.1f} fps, platen {platen_g:g} g")
    print(f"unloaded noise {noise:.2f} counts; baseline drift first->last gap {drift:+.1f} counts")
    on_shift = np.array([s[0] for s in mark_shift])
    off_shift = np.array([s[1] for s in mark_shift])
    print(f"SPACE timing vs the signal: 'on' pressed {np.median(-on_shift):+.1f} s after the weight landed, "
          f"'off' pressed {np.median(-off_shift):+.1f} s after it left (medians; edges taken from the signal)")
    print("summed response (primary)                                            | pad only      | footprint")
    print(f"{'mass g':>7s} {'load N':>7s} {'n':>2s} {'mean':>7s} {'std':>6s} {'cts/N':>7s} {'sep/sd':>6s} "
          f"{'res N':>7s} {'creep':>6s} {'hyst':>6s} | {'mean':>6s} {'std':>5s} | cells share")
    for m in per_mass:
        print(f"{m['mass_g']:7g} {m['load_N']:7.3f} {m['n']:2d} {m['mean']:7.1f} {m['std_rep']:6.2f} "
              f"{m.get('counts_per_N', np.nan):7.1f} {m.get('separation_sigma', np.nan):6.1f} "
              f"{m.get('force_resolution_N', np.nan):7.3f} {m['creep_mean']:+6.1f} {m['hysteresis']:+6.1f} | "
              f"{m['pad_mean']:6.1f} {m['pad_std_rep']:5.2f} | {m['cells_mean']:5.1f} {m['pad_share_mean']:5.0%}")
    if np.isfinite(mae):
        print(f"masses in order: {'yes' if monotone else 'NO'}; held-out-repetition MAE: summed {maes['sum']:.3f} N "
              f"({maes['sum'] / span:.0%} of the {span:.2f} N span), pad only {maes['pad']:.3f} N")
    else:
        print("not enough repetitions for a held-out score")
    if spread:
        print("weights that spread beyond the pad (pad carries < 70 % of the response): "
              + ", ".join(f"{m['mass_g']:g} g ({m['cells_mean']:.1f} cells, {m['pad_share_mean']:.0%})" for m in spread)
              + " -> trust the summed curve for these, not the pad curve")
    bad = [r for r in rows if np.isfinite(r["residual_after_counts"]) and abs(r["residual_after_counts"]) > 3 * noise + 1]
    if bad:
        print(f"{len(bad)} step(s) did not return to baseline within 2-8 s after removal: "
              + ", ".join(f"#{r['step']} ({r['mass_g']:g} g, {r['residual_after_counts']:+.1f})" for r in bad[:8]))

    # ---- files
    out = args.out or s / "report"
    out.mkdir(parents=True, exist_ok=True)
    keys = ["step", "mass_g", "load_N", "t_on_s", "hold_s", "sum_counts", "pad_counts", "pad_noise",
            "cells", "pad_share", "centroid", "creep_counts", "settle_s", "residual_after_counts",
            "prev_mass_g", "pred_N_sum", "pred_N_pad", "frames"]
    with (out / "weight_steps.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(r.get(k), float) and not np.isfinite(r[k]) else r.get(k, "")) for k in keys})
    clean_json = json.loads(json.dumps({
        "session": str(s), "pad": pad_name, "pad_stated": stated, "peak_pad": peak_name,
        "platen_g": platen_g, "fps": fps, "last_s": last_s, "steps": len(rows),
        "noise_counts": noise, "drift_counts": drift, "monotone": monotone,
        "heldout_mae_N": maes, "span_N": span, "per_mass": per_mass,
    }, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else float(x)))
    (out / "weight_report.json").write_text(json.dumps(clean_json, indent=2))
    plot(out / "weight_report.png", s.name, t, pad, rows, per_mass, noise, pad_name, mae)
    print(f"wrote {out}")


def plot(path, name, t, pad, rows, per_mass, noise, pad_name, mae):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]
    a.plot(t, pad, lw=0.8, color="tab:blue")
    for r in rows:
        a.axvspan(r["t_on_s"], r["t_on_s"] + r["hold_s"], color="tab:orange", alpha=0.15)
        a.text(r["t_on_s"], pad.max() * 1.02, f"{r['mass_g']:g}", fontsize=7, rotation=90, va="bottom")
    a.set(xlabel="time [s]", ylabel=f"{pad_name} change [counts]", title=f"{name}: pad response, weights shaded (g)")
    a.grid(alpha=0.3)

    a = ax[0, 1]
    markers = "osD^v<>*"
    reps = {}
    for r in rows:
        reps[r["step"]] = sum(1 for q in rows if q["mass_g"] == r["mass_g"] and q["step"] <= r["step"])
    for rep in sorted(set(reps.values())):
        rs = [r for r in rows if reps[r["step"]] == rep]
        a.scatter([r["load_N"] for r in rs], [r["sum_counts"] for r in rs], s=26,
                  marker=markers[(rep - 1) % len(markers)], label=f"rep {rep} (sum)", alpha=0.85)
        a.scatter([r["load_N"] for r in rs], [r["pad_counts"] for r in rs], s=26, facecolors="none",
                  edgecolors="grey", marker=markers[(rep - 1) % len(markers)], alpha=0.7)
    a.errorbar([m["load_N"] for m in per_mass], [m["mean"] for m in per_mass],
               yerr=[0 if not np.isfinite(m["std_rep"]) else m["std_rep"] for m in per_mass],
               color="k", lw=1.2, capsize=3, label="summed: mean ± std")
    a.plot([m["load_N"] for m in per_mass], [m["pad_mean"] for m in per_mass], color="grey", lw=1, ls="--",
           label=f"{pad_name} only (hollow)")
    a.axhspan(-noise, noise, color="grey", alpha=0.2, label="unloaded noise")
    a.set(xlabel="load [N]", ylabel="plateau [counts]", title="calibration points: summed response and pad only")
    a.legend(fontsize=8, frameon=False)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    rs = [r for r in rows if "pred_N_sum" in r]
    if rs:
        x = np.arange(len(rs))
        a.bar(x - 0.2, [r["pred_N_sum"] - r["load_N"] for r in rs], 0.4, color="tab:red", label="summed curve")
        a.bar(x + 0.2, [r["pred_N_pad"] - r["load_N"] for r in rs], 0.4, color="grey", label="pad curve")
        a.set_xticks(x, [f"{r['mass_g']:g}" for r in rs], fontsize=7)
        a.set(xlabel="step (mass g)", ylabel="predicted - true [N]",
              title=f"held-out repetition error: MAE summed {mae:.3f} N")
        a.legend(fontsize=8, frameon=False)
    else:
        a.text(0.5, 0.5, "not enough repetitions", ha="center", transform=a.transAxes)
    a.grid(axis="y", alpha=0.3)

    a = ax[1, 1]
    x = np.arange(len(rows))
    a.bar(x - 0.2, [r["creep_counts"] for r in rows], 0.4, label="creep over hold", color="tab:green")
    a.bar(x + 0.2, [r["residual_after_counts"] for r in rows], 0.4, label="residual 2-8 s after off", color="tab:purple")
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(x, [f"{r['mass_g']:g}" for r in rows], fontsize=7)
    a.set(xlabel="step (mass g)", ylabel="counts", title="creep and recovery per step")
    a.legend(fontsize=8, frameon=False)
    a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
