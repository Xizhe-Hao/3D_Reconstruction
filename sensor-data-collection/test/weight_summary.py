#!/usr/bin/env python3
"""All pads of a dead-weight calibration root on one page: gain map, dead pads, shared curve.

    python weight_summary.py C:/tactile_data/weight_cal [--refresh]

Runs weight_report.py for every capture_* session under ROOT that has
weights/events.csv (skipped when report/weight_report.json already exists,
unless --refresh), then:

- a table pad x mass of the plateau (summed response, 8-bit-equivalent counts)
  with the repetition std, and flags: DEAD (no mass above 3x noise), WEAK
  (largest mass below 10 counts), SPREAD (pad carries < 70 % of its own
  response), MISPLACED (largest response on another cell);
- maps of the array: gain at the heaviest mass, repetition std, noise;
- a shared calibration: one monotone curve shape for the whole array with a
  gain per pad (counts = gain_pad * h(load)), fitted by alternating isotonic
  regression; the held-out-pad score (fit on 15 pads, predict the 16th's
  loads from its own counts with its gain re-estimated from ONE mass) says
  whether a pad can be calibrated from a single weight once the shape is
  known. Also the per-pad-only score for comparison.

Writes ROOT/weight_summary.png, weight_summary.csv, weight_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--refresh", action="store_true", help="re-run weight_report.py on every session")
    p.add_argument("--rows", type=int, default=None, help="array rows (default: from the sessions)")
    p.add_argument("--cols", type=int, default=None)
    return p.parse_args()


def ensure_report(session: Path, refresh: bool):
    out = session / "report" / "weight_report.json"
    if out.exists() and not refresh:
        return out
    r = subprocess.run([sys.executable, str(HERE / "weight_report.py"), str(session)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{session.name}: weight_report.py failed:\n{r.stderr.strip()[-800:]}", file=sys.stderr)
        return None
    return out if out.exists() else None


def pad_rc(name):
    m = re.fullmatch(r"R(\d+)C(\d+)", name or "")
    return (int(m.group(1)) - 1, int(m.group(2)) - 1) if m else None


def iso_fit(x, y, increasing=True):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(x, y)


def shared_shape(points, n_iter=20):
    """points: list of (pad, load_N, counts). Returns (gain per pad, h: load -> counts (unit gain)).

    Alternating fit: h = isotonic(load -> counts / gain_pad) over all pads,
    gain_pad = least squares of counts on h(load) per pad; h is normalised so
    the median gain is 1.
    """
    pads = sorted({p for p, _, _ in points})
    gain = {p: 1.0 for p in pads}
    loads = np.array([l for _, l, _ in points])
    counts = np.array([c for _, _, c in points])
    pad_of = np.array([p for p, _, _ in points], dtype=object)
    h = None
    for _ in range(n_iter):
        g = np.array([gain[p] for p in pad_of])
        h = iso_fit(loads, counts / g)
        hl = h.predict(loads)
        for p in pads:
            m = pad_of == p
            denom = float((hl[m] ** 2).sum())
            gain[p] = float((counts[m] * hl[m]).sum() / denom) if denom > 0 else 1.0
        med = float(np.median(list(gain.values()))) or 1.0
        gain = {p: v / med for p, v in gain.items()}
    g = np.array([gain[p] for p in pad_of])
    h = iso_fit(loads, counts / g)
    return gain, h


def invert(h, loads_grid, counts):
    """load for given unit-gain counts, through the monotone h (interpolated on a grid)."""
    hl = h.predict(loads_grid)
    return np.interp(counts, hl, loads_grid)


def main() -> None:
    args = parse_args()
    sessions = sorted(d for d in args.root.glob("capture_*") if (d / "weights" / "events.csv").exists())
    if not sessions:
        raise SystemExit(f"no capture_* session with weights/events.csv under {args.root}")
    reports = []
    for s in sessions:
        path = ensure_report(s, args.refresh)
        if path is None:
            continue
        rep = json.loads(path.read_text())
        rep["session_path"] = s
        steps = list(csv.DictReader((s / "report" / "weight_steps.csv").open()))
        rep["steps_rows"] = steps
        reports.append(rep)
    if not reports:
        raise SystemExit("no usable report")

    # one entry per pad: the latest session of that pad
    by_pad = {}
    for rep in reports:
        rc = pad_rc(rep["pad"])
        if rc is None:
            continue
        by_pad[rc] = rep  # sorted by name = by time, so the last one wins
    rows_n = args.rows or max(r for r, _ in by_pad) + 1
    cols_n = args.cols or max(c for _, c in by_pad) + 1
    masses = sorted({m["mass_g"] for rep in by_pad.values() for m in rep["per_mass"]})
    heaviest = masses[-1]

    # table and flags
    table = []
    gain_map = np.full((rows_n, cols_n), np.nan)
    std_map = np.full((rows_n, cols_n), np.nan)
    noise_map = np.full((rows_n, cols_n), np.nan)
    for rc in sorted(by_pad):
        rep = by_pad[rc]
        pm = {m["mass_g"]: m for m in rep["per_mass"]}
        noise = rep["noise_counts"] or 0.0
        flags = []
        top = pm.get(heaviest)
        if all((m["mean"] or 0) < 3 * max(noise, 0.3) for m in pm.values()):
            flags.append("DEAD")
        elif top and (top["mean"] or 0) < 10:
            flags.append("WEAK")
        if top and top.get("pad_share_mean") is not None and top["pad_share_mean"] < 0.7:
            flags.append("SPREAD")
        if rep.get("peak_pad") and rep["peak_pad"] != rep["pad"]:
            flags.append(f"MISPLACED(peak {rep['peak_pad']})")
        if not rep.get("monotone", True):
            flags.append("NOT MONOTONE")
        row = {"pad": rep["pad"], "session": rep["session_path"].name, "noise": noise,
               "drift": rep.get("drift_counts"), "flags": " ".join(flags)}
        for m in masses:
            e = pm.get(m)
            row[f"{m:g}g"] = e["mean"] if e else np.nan
            row[f"{m:g}g_std"] = e["std_rep"] if e else np.nan
        table.append(row)
        if top:
            gain_map[rc] = top["mean"]
            std_map[rc] = top["std_rep"] if top["std_rep"] is not None else np.nan
        noise_map[rc] = noise

    # shared shape + per-pad gain, held-out pad
    points, by_pad_pts = [], {}
    for rc, rep in by_pad.items():
        for r in rep["steps_rows"]:
            if r["sum_counts"] == "":
                continue
            pt = (rep["pad"], float(r["load_N"]), float(r["sum_counts"]))
            points.append(pt)
            by_pad_pts.setdefault(rep["pad"], []).append(pt)
    shared = {}
    if len(by_pad_pts) >= 3:
        gain, h = shared_shape(points)
        grid = np.linspace(0, max(l for _, l, _ in points) * 1.05, 400)
        err_one, err_own = [], []
        for pad, pts in by_pad_pts.items():
            others = [p for p in points if p[0] != pad]
            if len({p[0] for p in others}) < 2:
                continue
            g_o, h_o = shared_shape(others, n_iter=12)
            hl = h_o.predict(grid)
            # this pad's gain from ONE mass (the heaviest, first repetition), then predict all its steps
            top_pts = [p for p in pts if p[1] == max(q[1] for q in pts)]
            g_pad = top_pts[0][2] / max(float(h_o.predict([top_pts[0][1]])[0]), 1e-6)
            for _, load, cnt in pts:
                pred = float(np.interp(cnt / max(g_pad, 1e-6), hl, grid))
                err_one.append(pred - load)
            # own-pad isotonic, leave one repetition out, for comparison
            loads = np.array([p[1] for p in pts])
            cnts = np.array([p[2] for p in pts])
            for k in range(len(pts)):
                m = np.ones(len(pts), bool)
                m[k] = False
                if m.sum() < 3:
                    continue
                own = iso_fit(cnts[m], loads[m])
                err_own.append(float(own.predict([cnts[k]])[0]) - loads[k])
        shared = {
            "gain_per_pad": gain,
            "shape_counts_at_unit_gain": {f"{l:.3f}": float(c) for l, c in zip(grid[::40], h.predict(grid[::40]))},
            "heldout_pad_one_weight_mae_N": float(np.abs(err_one).mean()) if err_one else None,
            "own_pad_leave_one_out_mae_N": float(np.abs(err_own).mean()) if err_own else None,
            "span_N": float(max(l for _, l, _ in points)),
        }

    # ---- print
    print(f"{len(by_pad)} pads under {args.root}; masses {', '.join(f'{m:g} g' for m in masses)}; "
          f"plateau = summed response, 8-bit-equivalent counts (mean +- repetition std)")
    head = f"{'pad':5s} " + " ".join(f"{f'{m:g}g':>12s}" for m in masses) + f" {'noise':>6s}  flags"
    print(head)
    for row in table:
        cells = " ".join(
            (f"{row[f'{m:g}g']:6.1f}+-{row[f'{m:g}g_std']:4.1f}" if np.isfinite(row[f"{m:g}g"]) and
             np.isfinite(row[f"{m:g}g_std"] or np.nan) else
             (f"{row[f'{m:g}g']:6.1f}      " if np.isfinite(row[f"{m:g}g"]) else f"{'-':>12s}")) for m in masses)
        print(f"{row['pad']:5s} {cells} {row['noise']:6.2f}  {row['flags']}")
    finite = gain_map[np.isfinite(gain_map)]
    if finite.size:
        print(f"gain at {heaviest:g} g: median {np.median(finite):.1f} counts, range {finite.min():.1f}-{finite.max():.1f} "
              f"(x{finite.max() / max(finite.min(), 1e-6):.1f}); pads flagged: "
              f"{sum(1 for r in table if r['flags'])}/{len(table)}")
    if shared:
        print(f"shared shape + per-pad gain: held-out pad calibrated from ONE {heaviest:g} g weight -> MAE "
              f"{shared['heldout_pad_one_weight_mae_N']:.3f} N; own-pad curve leave-one-out MAE "
              f"{shared['own_pad_leave_one_out_mae_N']:.3f} N (span {shared['span_N']:.2f} N)")
        gl = shared["gain_per_pad"]
        print("relative gain per pad: " + "  ".join(f"{p} {g:.2f}" for p, g in sorted(gl.items())))

    # ---- files
    with (args.root / "weight_summary.csv").open("w", newline="") as handle:
        keys = ["pad", "session"] + [k for m in masses for k in (f"{m:g}g", f"{m:g}g_std")] + ["noise", "drift", "flags"]
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        for row in table:
            w.writerow({k: ("" if isinstance(row.get(k), float) and not np.isfinite(row[k]) else row.get(k, "")) for k in keys})
    (args.root / "weight_summary.json").write_text(json.dumps(
        {"root": str(args.root), "masses_g": masses, "pads": table, "shared": shared},
        indent=2, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)))
    plot(args.root / "weight_summary.png", by_pad, masses, gain_map, std_map, noise_map, shared, points, rows_n, cols_n)
    print(f"wrote {args.root / 'weight_summary.png'}, weight_summary.csv, weight_summary.json")


def plot(path, by_pad, masses, gain_map, std_map, noise_map, shared, points, rows_n, cols_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for a, data, title in ((ax[0, 0], gain_map, f"plateau at {masses[-1]:g} g [counts]"),
                           (ax[0, 1], std_map, "repetition std [counts]"),
                           (ax[0, 2], noise_map, "unloaded noise [counts]")):
        im = a.imshow(data, cmap="viridis", interpolation="nearest")
        a.set_xticks(range(cols_n), [f"C{c + 1}" for c in range(cols_n)])
        a.set_yticks(range(rows_n), [f"R{r + 1}" for r in range(rows_n)])
        a.set_title(title)
        for r in range(rows_n):
            for c in range(cols_n):
                v = data[r, c]
                a.text(c, r, "-" if not np.isfinite(v) else f"{v:.1f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=a, shrink=0.8)

    a = ax[1, 0]
    for rc in sorted(by_pad):
        rep = by_pad[rc]
        a.plot([m["load_N"] for m in rep["per_mass"]], [m["mean"] for m in rep["per_mass"]], "-o", ms=3, lw=1,
               label=rep["pad"])
    a.set(xlabel="load [N]", ylabel="summed plateau [counts]", title="every pad's curve")
    a.legend(fontsize=6, ncol=2, frameon=False)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    if shared:
        gain = shared["gain_per_pad"]
        for pad, pts in {}.items():
            pass
        for p, l, c in points:
            a.scatter(l, c / gain[p], s=8, color="tab:blue", alpha=0.5)
        grid = np.array([float(k) for k in shared["shape_counts_at_unit_gain"]])
        a.plot(grid, list(shared["shape_counts_at_unit_gain"].values()), color="k", lw=1.5, label="shared shape")
        a.set(xlabel="load [N]", ylabel="counts / pad gain", title="all pads after dividing by their gain")
        a.legend(frameon=False)
    a.grid(alpha=0.3)

    a = ax[1, 2]
    if shared:
        gl = shared["gain_per_pad"]
        gmap = np.full((rows_n, cols_n), np.nan)
        for p, g in gl.items():
            rc = pad_rc(p)
            if rc:
                gmap[rc] = g
        im = a.imshow(gmap, cmap="coolwarm", vmin=0, vmax=2, interpolation="nearest")
        for r in range(rows_n):
            for c in range(cols_n):
                v = gmap[r, c]
                a.text(c, r, "-" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=8)
        a.set_xticks(range(cols_n), [f"C{c + 1}" for c in range(cols_n)])
        a.set_yticks(range(rows_n), [f"R{r + 1}" for r in range(rows_n)])
        a.set_title("relative gain per pad (median = 1)")
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
