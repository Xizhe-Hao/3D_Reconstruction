#!/usr/bin/env python3
"""One page per calibration root: how well does the weight model do on each pad?

    python weight_eval.py C:/tactile_data/weight_cal [--pads R1C1,R1C2,R1C3,R1C4] [--exclude R2C2]

Predictions are leave-one-repetition-out with the deployed model type (shared
Hill shape + gain per pad, see weight_live.py), so every number is for a
plateau the model did not see. Per pad: predicted vs true grams with the
mean +- std of each weight, bias and spread per weight, MAE. For the array: a
pad x weight MAE heat map, a confusion matrix of "which standard weight would
it be called", and a summary table.

Writes ROOT/weight_eval.png and weight_eval.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from weight_cal import G  # noqa: E402
from weight_models import Hill, load_points  # noqa: E402

STANDARD_G = (10.0, 20.0, 50.0, 100.0, 200.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--pads", default=None, help="pads to show (default: every calibrated pad)")
    p.add_argument("--exclude", default="", help="pads to leave out of the fit, e.g. R2C2")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exclude = {e.strip().upper() for e in args.exclude.split(",") if e.strip()}
    pts = load_points(args.root, exclude)
    if not pts:
        raise SystemExit("no pad session with report/weight_steps.csv; run weight_summary.py first")
    show = [p.strip().upper() for p in args.pads.split(",")] if args.pads else sorted({p[0] for p in pts})

    # leave one repetition out, deployed model type
    preds = []  # (pad, rep, true_g, pred_g)
    for rep in sorted({p[1] for p in pts}):
        m = Hill().fit([p for p in pts if p[1] != rep])
        for pad, r, F, c in pts:
            if r == rep:
                preds.append((pad, r, F / G * 1000.0, float(m.predict(pad, c)[0]) / G * 1000.0))
    masses = sorted({round(p[2]) for p in preds})

    # per pad x mass stats
    stats = {}
    for pad in show:
        for mg in masses:
            v = np.array([q[3] for q in preds if q[0] == pad and round(q[2]) == mg])
            if v.size:
                stats[(pad, mg)] = {"n": int(v.size), "mean": float(v.mean()),
                                    "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                                    "bias": float(v.mean() - mg), "mae": float(np.abs(v - mg).mean()),
                                    "nearest_ok": float(np.mean([min(STANDARD_G, key=lambda s: abs(s - x)) == mg for x in v]))}
    per_pad = {}
    for pad in show:
        e = np.array([q[3] - q[2] for q in preds if q[0] == pad])
        if e.size:
            per_pad[pad] = {"mae": float(np.abs(e).mean()), "bias": float(e.mean()), "std": float(e.std(ddof=1)),
                            "nearest_ok": float(np.mean([stats[(pad, mg)]["nearest_ok"] for mg in masses if (pad, mg) in stats]))}
    all_e = np.array([q[3] - q[2] for q in preds if q[0] in show])

    # confusion: true weight -> nearest standard weight of the prediction
    conf = np.zeros((len(masses), len(masses)), dtype=int)
    for pad, r, tg, pg in preds:
        if pad not in show:
            continue
        i = masses.index(round(tg))
        near = min(STANDARD_G, key=lambda s: abs(s - pg))
        if near in masses:
            conf[i, masses.index(near)] += 1

    # ---- print
    print(f"leave-one-repetition-out, shared Hill + gain per pad; {len(show)} pads, weights {masses} g")
    print(f"{'pad':5s} " + " ".join(f"{f'{mg:g} g':>14s}" for mg in masses) + f"   {'MAE':>5s} {'bias':>5s} {'std':>5s} named-right")
    for pad in show:
        cells = []
        for mg in masses:
            s = stats.get((pad, mg))
            cells.append(f"{s['mean']:6.0f}+-{s['std']:4.0f} g" if s else f"{'-':>14s}")
        pp = per_pad.get(pad)
        print(f"{pad:5s} " + " ".join(cells) + (f"   {pp['mae']:5.0f} {pp['bias']:+5.0f} {pp['std']:5.0f}   {pp['nearest_ok'] * 100:3.0f}%" if pp else ""))
    print(f"all   MAE {np.abs(all_e).mean():.0f} g, bias {all_e.mean():+.0f} g, std {all_e.std(ddof=1):.0f} g; "
          f"named the right standard weight {np.trace(conf) / max(conf.sum(), 1) * 100:.0f}% of the time")

    out = args.out or args.root
    with (out / "weight_eval.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["pad", "mass_g", "n", "pred_mean_g", "pred_std_g", "bias_g", "mae_g", "nearest_correct"])
        for (pad, mg), s in sorted(stats.items()):
            w.writerow([pad, mg, s["n"], f"{s['mean']:.1f}", f"{s['std']:.1f}", f"{s['bias']:.1f}", f"{s['mae']:.1f}", f"{s['nearest_ok']:.2f}"])
    plot(out / "weight_eval.png", show, masses, preds, stats, per_pad, conf, all_e)
    print(f"wrote {out / 'weight_eval.png'}, weight_eval.csv")


def plot(path, show, masses, preds, stats, per_pad, conf, all_e):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(show)
    fig = plt.figure(figsize=(4.3 * n + 1, 13.5))
    gs = GridSpec(3, n, figure=fig, height_ratios=(1.25, 1.0, 1.1), hspace=0.45, wspace=0.3)
    lim = max(masses) * 1.25
    markers = "osD^v"
    for i, pad in enumerate(show):
        a = fig.add_subplot(gs[0, i])
        a.plot([0, lim], [0, lim], color="k", lw=0.8)
        a.fill_between([0, lim], [0 - 19, lim - 19], [0 + 19, lim + 19], color="grey", alpha=0.12, label="+-19 g (model)")
        for rep in sorted({q[1] for q in preds if q[0] in show}):
            qq = [q for q in preds if q[0] == pad and q[1] == rep]
            a.scatter([q[2] for q in qq], [q[3] for q in qq], s=34, marker=markers[(rep - 1) % 5], color="tab:blue",
                      alpha=0.75, label=f"rep {rep}", zorder=3)
        xs = [mg for mg in masses if (pad, mg) in stats]
        a.errorbar(xs, [stats[(pad, mg)]["mean"] for mg in xs], yerr=[stats[(pad, mg)]["std"] for mg in xs],
                   fmt="_", color="tab:red", ms=16, mew=2, capsize=5, lw=1.5, label="mean +- std", zorder=4)
        pp = per_pad.get(pad, {})
        a.set(xlim=(0, lim), ylim=(0, lim), xlabel="true weight [g]", ylabel="predicted [g]")
        a.set_title(f"{pad}   MAE {pp.get('mae', np.nan):.0f} g   bias {pp.get('bias', np.nan):+.0f} g   "
                    f"std {pp.get('std', np.nan):.0f} g", fontsize=10)
        a.set_xticks(masses)
        a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=7, frameon=False, loc="upper left")

        a = fig.add_subplot(gs[1, i])
        x = np.arange(len(masses))
        bias = [stats[(pad, mg)]["bias"] if (pad, mg) in stats else np.nan for mg in masses]
        std = [stats[(pad, mg)]["std"] if (pad, mg) in stats else 0 for mg in masses]
        colors = ["tab:green" if abs(b) <= 19 else "tab:orange" if abs(b) <= 40 else "tab:red" for b in bias]
        a.bar(x, bias, 0.6, yerr=std, capsize=4, color=colors, alpha=0.85)
        a.axhline(0, color="k", lw=0.8)
        for k, mg in enumerate(masses):
            s = stats.get((pad, mg))
            if s:
                a.text(k, (s["bias"] + (s["std"] if s["bias"] >= 0 else -s["std"])) * 1.05 + (3 if s["bias"] >= 0 else -8),
                       f"{s['bias']:+.0f}+-{s['std']:.0f}", ha="center", fontsize=8)
        a.set_xticks(x, [f"{mg:g} g" for mg in masses])
        a.set(ylabel="predicted - true [g]", title="bias per weight (bar) and spread (whisker)", xlabel="weight")
        a.grid(axis="y", alpha=0.3)

    # bottom row: MAE heat map, confusion, summary text
    a = fig.add_subplot(gs[2, 0:max(1, n // 2)])
    mat = np.array([[stats[(pad, mg)]["mae"] if (pad, mg) in stats else np.nan for mg in masses] for pad in show])
    im = a.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(40, np.nanmax(mat)), aspect="auto")
    a.set_xticks(range(len(masses)), [f"{mg:g} g" for mg in masses])
    a.set_yticks(range(len(show)), show)
    for i in range(len(show)):
        for j in range(len(masses)):
            if np.isfinite(mat[i, j]):
                a.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=10,
                       color="w" if mat[i, j] > 0.6 * max(40, np.nanmax(mat)) else "k")
    a.set_title("MAE per pad and weight [g]", fontsize=10)
    fig.colorbar(im, ax=a, shrink=0.8)

    a = fig.add_subplot(gs[2, max(1, n // 2):max(2, n - 1)] if n >= 3 else gs[2, 1:])
    im = a.imshow(conf, cmap="Blues", aspect="auto")
    a.set_xticks(range(len(masses)), [f"{mg:g}" for mg in masses])
    a.set_yticks(range(len(masses)), [f"{mg:g}" for mg in masses])
    a.set(xlabel="named as [g]", ylabel="true weight [g]", title="which standard weight the prediction is closest to")
    for i in range(len(masses)):
        for j in range(len(masses)):
            if conf[i, j]:
                a.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=10,
                       color="w" if conf[i, j] > conf.max() * 0.6 else "k")

    if n >= 3:
        a = fig.add_subplot(gs[2, n - 1])
        a.axis("off")
        lines = [f"{'pad':5s} {'MAE':>5s} {'bias':>5s} {'std':>5s} {'named':>6s}"]
        for pad in show:
            pp = per_pad.get(pad)
            if pp:
                lines.append(f"{pad:5s} {pp['mae']:5.0f} {pp['bias']:+5.0f} {pp['std']:5.0f} {pp['nearest_ok'] * 100:5.0f}%")
        lines.append("")
        lines.append(f"all   {np.abs(all_e).mean():5.0f} {all_e.mean():+5.0f} {all_e.std(ddof=1):5.0f} "
                     f"{np.trace(conf) / max(conf.sum(), 1) * 100:5.0f}%")
        lines += ["", "grams; leave-one-repetition-out;", "shared Hill shape + gain per pad", "(the model weight_live.py uses)"]
        a.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10, transform=a.transAxes)
    fig.suptitle("dead-weight model: prediction quality per pad", fontsize=13, y=0.995)
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
