#!/usr/bin/env python3
"""Quantitative report of an aligned bench run: drift, hold behaviour, accuracy, maps.

    python bench_report.py <capture session>

Needs <session>/force_alignment/frames.csv from align_force_presses.py and the
session's sensor_raw.csv. Writes <session>/report/:

    report_1_quantitative.png  timeline (force, crosshead, tactile), drift per depth
                               and cycle, hold changes, calibration curves per cycle,
                               held-out force prediction and its error
    report_2_maps.png          2D tactile maps (depth x cycle, one colour scale)
                               and 3D bars of the deepest press
    presses.csv                one row per press with the numbers behind the plots

Presses are grouped by depth (rounded to --depth-bin mm) and by cycle: a new
cycle starts whenever the depth goes down from one press to the next, which
matches programs that step through increasing depths and repeat.

The prediction panel is deliberately simple: a cubic curve from total tactile
response to force, fitted on every cycle except one and tested on that one. It
shows what the signal supports, not the best a model could do.

Works for any grid (16x16 M16B or 4x4 ESP32 sessions); the press direction is
taken from the data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python files"))
from tactile_preprocess import load_raw, response_sign, session_meta  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("session", type=Path)
    p.add_argument("--depth-bin", type=float, default=2.5, help="mm; depths are rounded to this")
    p.add_argument("--hold-skip-s", type=float, default=0.5, help="ignore the start of a hold (settling)")
    p.add_argument("--window-frames", type=int, default=3, help="frames averaged at hold start / end")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


# ---------------------------------------------------------------- data


def load_session(session: Path):
    _, idx, frames = load_raw(session)  # grid size from the header
    by_index = {i: k for k, i in enumerate(idx)}

    lab = list(csv.DictReader((session / "force_alignment" / "frames.csv").open(newline="")))
    k = np.array([by_index[int(r["frame_index"])] for r in lab])
    base = np.median(frames[:max(10, k[0])], axis=0) if k[0] > 10 else np.median(frames[:30], axis=0)
    sign = response_sign(frames, base, session_meta(session, *frames.shape[1:]))
    resp = np.clip(sign * (frames[k] - base), 0, None)
    d = {
        "t": np.array([float(r["t_s"]) for r in lab]),
        "load": np.array([float(r["load_N"]) for r in lab]),
        "dist": np.array([float(r["distance_mm"]) for r in lab]),
        "ind": np.array([float(r["indentation_mm"]) if r["indentation_mm"] else np.nan for r in lab]),
        "press": np.array([int(r["press"]) for r in lab]),
        "phase": np.array([r["phase"] for r in lab]),
        "resp": resp,
    }
    d["total"] = resp.sum(axis=(1, 2))
    return d


def press_table(d, args):
    t, press, phase = d["t"], d["press"], d["phase"]
    rows = []
    for p in range(1, press.max() + 1):
        s = np.flatnonzero(press == p)
        if s.size < 3:
            continue
        depth = float(np.nanmax(d["ind"][s]))
        hold = s[phase[s] == "hold"]
        hold = hold[t[hold] >= t[hold[0]] + args.hold_skip_s] if hold.size else hold
        w = args.window_frames
        row = {"press": p, "t_s": float(t[s[0]]), "depth_mm": depth,
               "depth_bin": float(np.round(depth / args.depth_bin) * args.depth_bin),
               "hold_s": float(t[hold[-1]] - t[hold[0]]) if hold.size > 1 else 0.0,
               "peak_force_N": float(np.nanmax(d["load"][s])), "peak_tactile": float(d["total"][s].max()),
               "peak_frame": int(s[np.argmax(d["total"][s])])}
        if hold.size >= 2 * w:
            a, b = hold[:w], hold[-w:]
            row.update(force_start=float(d["load"][a].mean()), force_end=float(d["load"][b].mean()),
                       tactile_start=float(d["total"][a].mean()), tactile_end=float(d["total"][b].mean()),
                       start_frame=int(a[w // 2]))
        rows.append(row)
    cycle, prev = 0, None
    for row in rows:  # a new cycle whenever the depth goes down
        if prev is not None and row["depth_bin"] < prev:
            cycle += 1
        row["cycle"] = cycle
        prev = row["depth_bin"]
    return rows


def loo_prediction(d, rows):
    """Cubic total->force, trained on all cycles but one, tested on that one."""
    cyc_of_press = {r["press"]: r["cycle"] for r in rows}
    cycle = np.array([cyc_of_press.get(p, -1) for p in d["press"]])
    use = (d["press"] > 0) & (d["load"] > 0.5) & np.isfinite(d["load"])
    pred = np.full(len(d["t"]), np.nan)
    for c in sorted(set(cycle[use])):
        tr, te = use & (cycle != c), use & (cycle == c)
        if tr.sum() < 20 or te.sum() == 0:
            continue
        coef = np.polyfit(d["total"][tr] / 1000, d["load"][tr], 3)
        pred[te] = np.polyval(coef, d["total"][te] / 1000)
    return pred, cycle, use


def rmse(a):
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


# ---------------------------------------------------------------- figures


def page_quantitative(d, rows, pred, cycle, use, path, title):
    import matplotlib.pyplot as plt

    cycles = sorted({r["cycle"] for r in rows})
    cmap = plt.get_cmap("viridis")
    ccol = {c: cmap(i / max(1, len(cycles) - 1)) for i, c in enumerate(cycles)}
    depths = sorted({r["depth_bin"] for r in rows if "tactile_start" in r})
    dcol = {dep: plt.get_cmap("tab10")(i) for i, dep in enumerate(depths)}

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 1, 1, 1, 1.1], hspace=.38, wspace=.18)

    # 1. timeline
    ax = fig.add_subplot(gs[0, :])
    ax.plot(d["t"], d["load"], color="tab:red", lw=.8, label="force [N]")
    ax.set_ylabel("force [N]", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(d["t"], d["dist"], color="0.45", lw=.8, label="crosshead [mm]")
    ax2.set_ylabel("crosshead distance [mm]", color="0.45")
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.06))
    ax3.plot(d["t"], d["total"], color="tab:blue", lw=.8, alpha=.8)
    ax3.set_ylabel("tactile response (sum)", color="tab:blue")
    for r in rows:
        ax.annotate(f"{r['press']}\n{r['depth_mm']:.0f}mm", (r["t_s"], r["peak_force_N"]), fontsize=7,
                    ha="center", va="bottom")
    ax.set(title=f"{title}: force, crosshead and tactile on one clock", xlabel="time [s]")
    ax.grid(alpha=.3)

    # 2. drift per depth: absolute and relative
    axa, axr = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for dep in depths:
        g = [r for r in rows if r["depth_bin"] == dep and "tactile_start" in r and r["hold_s"] < 20]
        if not g:
            continue
        x = [r["cycle"] for r in g]
        tv = np.array([r["tactile_start"] for r in g])
        fv = np.array([r["force_start"] for r in g])
        axa.plot(x, tv, "o-", color=dcol[dep], label=f"{dep:g} mm")
        ref_t, ref_f = tv[-3:].mean(), fv[-3:].mean()
        axr.plot(x, (tv / ref_t - 1) * 100, "o-", color=dcol[dep], label=f"tactile {dep:g} mm")
        axr.plot(x, (fv / ref_f - 1) * 100, "x--", color=dcol[dep], alpha=.6, label=f"force {dep:g} mm")
        axa.annotate(f"{(tv[-1] / tv[0] - 1) * 100:+.0f}%", (x[-1], tv[-1]), xytext=(6, 0),
                     textcoords="offset points", color=dcol[dep], fontsize=9, va="center")
    axa.set(title="drift: tactile at hold start, same depth, each cycle (label: first -> last)",
            xlabel="cycle", ylabel="tactile response")
    axa.legend(fontsize=8)
    axr.axhline(0, color="0.4", lw=.8)
    axr.axhspan(-5, 5, color="0.85", zorder=0)
    axr.set(title="relative to the mean of the last 3 cycles (o tactile, x force; grey = +-5 %)",
            xlabel="cycle", ylabel="% from stable value")
    axr.legend(fontsize=7, ncol=2)
    for a in (axa, axr):
        a.grid(alpha=.3)

    # 3. hold behaviour and calibration curves per cycle
    axh, axc = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    hr = [r for r in rows if "tactile_start" in r]
    x = np.arange(len(hr))
    fchg = [(r["force_end"] / r["force_start"] - 1) * 100 for r in hr]
    tchg = [(r["tactile_end"] / r["tactile_start"] - 1) * 100 for r in hr]
    axh.bar(x - .2, fchg, .4, color="tab:red", label="force")
    axh.bar(x + .2, tchg, .4, color="tab:blue", label="tactile")
    axh.set_xticks(x, [f"{r['press']}\n{r['depth_mm']:.0f}mm\n{r['hold_s']:.0f}s" for r in hr], fontsize=7)
    axh.axhline(0, color="0.4", lw=.8)
    axh.set(title=f"change during each hold (force median {np.median(fchg):+.0f} %, tactile median {np.median(tchg):+.0f} %)",
            ylabel="% change, hold end vs start")
    axh.legend()
    for c in cycles:
        sel = use & (cycle == c) & np.isin(d["phase"], ["loading", "hold"])
        axc.scatter(d["load"][sel], d["total"][sel], s=4, color=ccol[c], label=f"cycle {c}")
    axc.set(title="calibration curve per cycle (loading + hold)", xlabel="force [N]", ylabel="tactile response")
    axc.legend(fontsize=8, markerscale=3)
    for a in (axh, axc):
        a.grid(alpha=.3)

    # 4. prediction over time
    axp = fig.add_subplot(gs[3, :])
    axp.plot(d["t"], d["load"], color="0.2", lw=.8, label="measured force")
    axp.plot(d["t"], pred, color="tab:orange", lw=.8, label="predicted from tactile (held-out cycle)")
    err = pred - d["load"]
    per_cycle = {c: rmse(err[use & (cycle == c)]) for c in cycles}
    for c in cycles:
        sel = use & (cycle == c)
        if sel.any():
            axp.text(d["t"][sel].mean(), axp.get_ylim()[1] if False else np.nanmax(d["load"]) * 1.02,
                     f"cycle {c}\n{per_cycle[c]:.1f} N", ha="center", fontsize=8, color=ccol[c])
    axp.set(title=f"held-out force prediction (cubic of tactile sum): RMSE {rmse(err[use]):.2f} N "
                  f"= {rmse(err[use]) / np.nanmax(d['load'][use]) * 100:.0f} % of max force",
            xlabel="time [s]", ylabel="force [N]", ylim=(-2, np.nanmax(d["load"]) * 1.15))
    axp.legend(loc="upper right")
    axp.grid(alpha=.3)

    # 5. error by phase and by depth, predicted vs measured
    sub = gs[4, 0].subgridspec(1, 2, wspace=.3)
    axe, axd, axs = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1]), fig.add_subplot(gs[4, 1])
    phases = ["loading", "hold", "unloading"]
    vals = [rmse(err[use & (d["phase"] == ph)]) for ph in phases]
    axe.bar(phases, vals, color=["tab:blue", "tab:green", "tab:red"])
    for i, v in enumerate(vals):
        axe.text(i, v, f"{v:.2f} N", ha="center", va="bottom")
    dep_err = []
    for dep in depths:
        ps = [r["press"] for r in rows if r["depth_bin"] == dep]
        dep_err.append(rmse(err[use & np.isin(d["press"], ps)]))
    labels = [f"{x:g} mm" for x in depths]
    axd.bar(labels, dep_err, color=[dcol[x] for x in depths])
    for i, v in enumerate(dep_err):
        axd.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    axd.set(title="held-out force error by depth", ylabel="RMSE [N]")
    axd.grid(alpha=.3)
    axe.set(title="held-out force error by phase", ylabel="RMSE [N]")
    for ph, col in zip(phases, ["tab:blue", "tab:green", "tab:red"]):
        sel = use & (d["phase"] == ph)
        axs.scatter(d["load"][sel], pred[sel], s=4, color=col, label=ph)
    lim = [0, np.nanmax(d["load"][use]) * 1.05]
    axs.plot(lim, lim, color="0.3", lw=.8)
    axs.set(title="predicted vs measured force", xlabel="measured [N]", ylabel="predicted [N]", xlim=lim, ylim=lim)
    axs.legend(markerscale=3)
    for a in (axe, axs):
        a.grid(alpha=.3)

    fig.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)


def page_maps(d, rows, path, title):
    import matplotlib.pyplot as plt

    hr = [r for r in rows if "start_frame" in r and r["hold_s"] < 20]
    depths = sorted({r["depth_bin"] for r in hr})
    cycles = sorted({r["cycle"] for r in hr})
    vmax = max(d["resp"][r["start_frame"]].max() for r in hr)
    fig = plt.figure(figsize=(3.2 * len(cycles) + 1, 3.2 * len(depths) + 6.5))
    gs = fig.add_gridspec(len(depths) + 2, len(cycles), height_ratios=[1] * len(depths) + [.15, 1.9],
                          hspace=.45, wspace=.15)
    im = None
    for i, dep in enumerate(depths):
        for j, c in enumerate(cycles):
            ax = fig.add_subplot(gs[i, j])
            r = next((r for r in hr if r["depth_bin"] == dep and r["cycle"] == c), None)
            ax.set_xticks([])
            ax.set_yticks([])
            if r is None:
                ax.axis("off")
                continue
            m = d["resp"][r["start_frame"]]
            im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax, origin="upper")
            w = m.sum()
            rr, cc = np.mgrid[0:m.shape[0], 0:m.shape[1]]
            if w > 0:
                ax.plot((cc * m).sum() / w, (rr * m).sum() / w, "o", mfc="none", mec="cyan", ms=7)
            ax.set_title(f"#{r['press']}  {r['depth_mm']:.1f} mm  {r['force_start']:.1f} N\nsum {r['tactile_start']:.0f}",
                         fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{dep:g} mm", fontsize=11)
            if i == 0:
                ax.text(.5, 1.45, f"cycle {c}", transform=ax.transAxes, ha="center", fontsize=11)
    if im is not None:
        cax = fig.add_subplot(gs[len(depths), :])
        fig.colorbar(im, cax=cax, orientation="horizontal", label="ADC counts above baseline (one scale for all)")

    # 3D bars: deepest press, first and last cycle
    deepest = max(depths)
    pick = [r for r in hr if r["depth_bin"] == deepest]
    pick = [pick[0], pick[-1]] if len(pick) > 1 else pick
    n3 = len(pick) + 1
    sub = gs[len(depths) + 1, :].subgridspec(1, n3, wspace=.05)
    n_rows, n_cols = d["resp"].shape[1:]
    xx, yy = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    for k, r in enumerate(pick):
        ax = fig.add_subplot(sub[0, k], projection="3d")
        m = d["resp"][r["start_frame"]]
        ax.bar3d(xx.ravel(), yy.ravel(), 0, .8, .8, m.ravel(), color=plt.get_cmap("magma")(m.ravel() / vmax),
                 shade=True)
        ax.set(zlim=(0, vmax), xlabel="C", ylabel="R", zlabel="counts",
               title=f"3D, press #{r['press']} ({deepest:g} mm, cycle {r['cycle']}): sum {r['tactile_start']:.0f}")
        ax.invert_yaxis()
        ax.view_init(35, -60)
    # row / column profiles of the same presses
    ax = fig.add_subplot(sub[0, n3 - 1])
    for r, ls in zip(pick, ["-", "--"]):
        m = d["resp"][r["start_frame"]]
        ax.plot(np.arange(1, n_rows + 1), m.sum(axis=1), ls, color="tab:blue", label=f"rows, #{r['press']}")
        ax.plot(np.arange(1, n_cols + 1), m.sum(axis=0), ls, color="tab:orange", label=f"columns, #{r['press']}")
    ax.set(title="row / column sums (dead or weak lines show as dips)", xlabel="row or column number",
           ylabel="counts", xticks=range(1, max(n_rows, n_cols) + 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.suptitle(f"{title}: tactile maps at hold start, depth x cycle (o = centroid)", fontsize=14, y=.995)
    fig.savefig(path, dpi=70, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    args = parse_args()
    d = load_session(args.session)
    rows = press_table(d, args)
    pred, cycle, use = loo_prediction(d, rows)
    out = args.out or args.session / "report"
    out.mkdir(parents=True, exist_ok=True)
    title = args.session.name
    page_quantitative(d, rows, pred, cycle, use, out / "report_1_quantitative.png", title)
    page_maps(d, rows, out / "report_2_maps.png", title)
    keys = ["press", "cycle", "t_s", "depth_mm", "depth_bin", "hold_s", "peak_force_N", "peak_tactile",
            "force_start", "force_end", "tactile_start", "tactile_end"]
    with (out / "presses.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.3f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})
    err = pred - d["load"]
    print(f"{len(rows)} presses in {len({r['cycle'] for r in rows})} cycles; "
          f"held-out force RMSE {rmse(err[use]):.2f} N")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
