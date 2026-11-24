#!/usr/bin/env python3
"""Per-pad force calibration from weight_cal.py sessions, with crosstalk kept OUT of the force.

    python pad_calibration.py C:/tactile_data/weight_cal_v3            # every capture_*_RxCy under the root
    python pad_calibration.py <session> [<session> ...]                 # or chosen sessions
    python pad_calibration.py ROOT --max-mass 200                       # ignore heavier steps (saturated)

Why this replaces the summed-response curve of weight_report.py
--------------------------------------------------------------
weight_report.py calibrates the SUM of every cell that moved by more than a
few counts against the load. That assumes every moving cell moved because the
weight pressed on it. On this matrix that is false: the readings of a passive
row/column matrix without diodes are coupled, so loading one pad shifts other
pads, most strongly the ones sharing its row or column, even ones 30 mm away
that the weight cannot touch (2026-11-13: a weight on R1C1 moved all of column
C2 by 60-175 counts). Adding those cells to the sum puts the crosstalk into
the force. Here:

  1. Every step gets its OWN no-load reference: the median of the unloaded
     seconds just before that weight went on (not the start of the session),
     so slow drift and incomplete recovery from the previous step do not
     enter the response.
  2. The calibrated quantity is the TARGET PAD's own change only:
     reference - plateau (median of the last --last-s seconds of the hold),
     in native ADC counts, positive = more pressure.
  3. Every other cell's change in the same step is stored as crosstalk
     (counts and as a fraction of the target's change). It is reported, never
     added to the force.
  4. Each step is checked: the target must carry the largest change, its
     change must be well above its own noise, and it must come back near the
     reference after the weight is removed. Flagged steps are listed and left
     out of the fit (--keep-flagged keeps them).
  5. Per pad, response = a * F^b is fitted in log-log (Velostat conductance
     follows a power law of force over its useful range); b is also fitted
     once for all pads with a per-pad gain a. The inverse F = (r / a)^(1/b)
     is the calibration. A step whose raw reading sits at the ADC rail
     (--clip) is clipped and left out of the fit; a mass where half or more of
     the steps clipped marks the pad SATURATED above the next lighter mass.
     With several sessions of one pad only the most complete is used
     (--all-sessions pools them). Accuracy is scored leave-one-repetition-out,
     in newtons.

Writes <root>/pad_calibration/: steps.csv (one row per step, every cell's
change), pads.csv, model.json, curves.png, crosstalk.png.
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
from tactile_preprocess import baseline, baseline_seconds, load_raw, response_sign, session_meta  # noqa: E402
from weight_cal import G, intervals, read_events, refine_edges  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("paths", type=Path, nargs="+", help="a capture root or session folders")
    p.add_argument("--pre-s", type=float, default=4.0, help="unloaded seconds before each step used as its reference")
    p.add_argument("--gap-s", type=float, default=0.5, help="seconds left out right before the weight goes on")
    p.add_argument("--settle-s", type=float, default=3.0, help="seconds after placement never counted as plateau")
    p.add_argument("--last-s", type=float, default=None, help="plateau = last seconds of the hold (default: session)")
    p.add_argument("--recovery-s", type=float, nargs=2, default=(2.0, 8.0), metavar=("FROM", "TO"),
                   help="window after removal where the pad must be back near its reference")
    p.add_argument("--max-mass", type=float, default=None, help="leave out steps heavier than this (g)")
    p.add_argument("--skip-reps", type=int, default=0,
                   help="leave out the first N repetitions of every mass (break-in of a fresh pad)")
    p.add_argument("--keep-flagged", action="store_true", help="fit flagged steps too")
    p.add_argument("--snr", type=float, default=5.0, help="target change must exceed this many noise std")
    p.add_argument("--residual", type=float, default=0.2, help="max residual after removal, fraction of the change")
    p.add_argument("--clip", type=float, default=0.04,
                   help="a step whose raw ADC reading on the target comes within this fraction of the ADC range "
                        "of the rail (0 or full scale) is clipped: left out of the fit (default 0.04 = ~160 of 4096)")
    p.add_argument("--all-sessions", action="store_true",
                   help="pool every session of a pad (default: only its most complete one, latest on a tie)")
    p.add_argument("--out", type=Path, default=None, help="output folder (default <root>/pad_calibration)")
    return p.parse_args(argv)


# ---------------------------------------------------------------- per session

def sessions_from(paths):
    found = []
    for p in paths:
        if (p / "weights" / "events.csv").exists():
            found.append(p)
        else:
            found += sorted(d for d in p.glob("capture_*") if (d / "weights" / "events.csv").exists())
    if not found:
        raise SystemExit("no weight_cal sessions (capture_*/weights/events.csv) found")
    return found


def window(t, a, b):
    return np.flatnonzero((t >= a) & (t <= b))


def analyse_session(session: Path, args):
    meta_raw = json.loads((session / "session.json").read_text(encoding="utf-8"))
    m = re.fullmatch(r"R(\d+)C(\d+)", meta_raw.get("pad_under_test") or "")
    if not m:
        print(f"skip {session.name}: session.json has no pad_under_test")
        return []
    pr, pc = int(m.group(1)) - 1, int(m.group(2)) - 1
    ns, _, frames = load_raw(session)
    bits = int(meta_raw.get("adc_bits") or 8)
    frames = frames * 2 ** (bits - 8)                      # back to native counts (12-bit on the ESP32 board)
    t = (ns - ns[0]) / 1e9
    meta = session_meta(session, *frames.shape[1:])
    start = baseline(frames, t, baseline_seconds(meta_raw))
    sign = response_sign(frames, start, meta)
    pressure = sign * frames                               # larger = more pressure, any offset
    platen_g = float(meta_raw.get("platen_g") or 0.0)
    last_s = args.last_s or float(meta_raw.get("plateau_last_s") or 5.0)
    steps = intervals(read_events(session))
    end_t = t[-1]

    # the loaded stretch of each step from the target's own signal; the SPACE marks only bound the search
    target_vs_start = pressure[:, pr, pc] - sign * start[pr, pc]
    spans = []
    for step, mass, on_ns, off_ns in steps:
        a = (on_ns - ns[0]) / 1e9
        b = (off_ns - ns[0]) / 1e9 if off_ns else end_t
        spans.append(refine_edges(t, target_vs_start, a, b) if mass > 0 else (a, b))

    x = pressure[:, pr, pc]
    base_s = baseline_seconds(meta_raw)
    rows = []
    starts = []                                            # where each step's load really began (rise start)
    for k, ((step, mass, on_ns, off_ns), (a, b)) in enumerate(zip(steps, spans)):
        prev_off = spans[k - 1][1] if k else 0.0
        next_on = spans[k + 1][0] if k + 1 < len(spans) else end_t
        hold = window(t, max(a + args.settle_s, b - last_s), b - args.gap_s)
        if hold.size < 3:
            continue
        level = float(np.median(x[hold]))
        # Unloaded part of the rest before this step: the lower half of the target's values there (the
        # weight may already be on for the last seconds before the SPACE mark, so the marks cannot decide).
        cand = window(t, prev_off + args.recovery_s[0] if k else 0.0, a)
        if cand.size >= 5:
            quiet = cand[x[cand] <= np.median(x[cand])]
            ref_t = float(np.median(x[quiet]))
            # walk back from the 50 % edge to where the rise really started (5 % of the step)
            thr = ref_t + 0.05 * (level - ref_t)
            i = int(np.searchsorted(t, a))
            while i > cand[0] and x[i] > thr:
                i -= 1
            rise = float(t[i])
            ref_w = window(t, max(t[cand[0]], rise - args.gap_s - args.pre_s), rise - args.gap_s)
            if ref_w.size < 5:
                ref_w = quiet
            ref = np.median(pressure[ref_w], axis=0)
            ref_src = "pre-step"
            noise = float(np.std(x[ref_w]))
        else:                                              # nothing unloaded before it: the session start
            rise = a
            ref = sign * start
            ref_src = "session start"
            noise = float(np.std(x[t <= base_s]))
        starts.append(rise)
        change = np.median(pressure[hold], axis=0) - ref    # every cell, counts, + = pressure
        target = float(change[pr, pc])
        # headroom of the raw reading to the rail it moves towards under load (ADC floor when pressing lowers it)
        full = 2 ** bits - 1
        raw = float(np.median(frames[hold, pr, pc]))
        headroom = raw if sign < 0 else full - raw
        clipped = mass > 0 and headroom < args.clip * full
        rec_w = window(t, b + args.recovery_s[0], min(b + args.recovery_s[1], next_on - 2.0))
        residual = float(np.median(x[rec_w]) - ref[pr, pc]) if rec_w.size >= 3 else np.nan

        flags = []
        if mass > 0:
            others = np.abs(change).copy()
            others[pr, pc] = -np.inf
            qr, qc = np.unravel_index(int(others.argmax()), others.shape)
            if abs(change[qr, qc]) > target:
                flags.append(f"largest change on R{qr + 1}C{qc + 1} ({change[qr, qc]:+.0f} vs {target:+.0f})")
            if target < args.snr * max(noise, 0.5):
                flags.append(f"below noise ({target:.0f} < {args.snr:g} x {noise:.1f})")
            if np.isfinite(residual) and target > 0 and abs(residual) > args.residual * target:
                flags.append(f"did not recover (residual {residual:+.0f})")
        rows.append({
            "session": session.name, "pad": f"R{pr + 1}C{pc + 1}", "pr": pr, "pc": pc, "step": step,
            "mass_g": mass, "load_N": (mass + platen_g) * G / 1000.0, "rep": None,
            "t_on": a, "t_off": b, "reference": ref_src, "noise": noise, "target": target,
            "residual": residual, "change": change, "flags": flags, "raw": raw, "clipped": clipped,
        })
    # repetition number per mass, in the order the steps were done
    seen = {}
    for r in rows:
        seen[r["mass_g"]] = seen.get(r["mass_g"], 0) + 1
        r["rep"] = seen[r["mass_g"]]
    return rows


# ---------------------------------------------------------------- fitting

def fit_power(F, r):
    """log r = log a + b log F; returns (a, b) or None."""
    ok = (F > 0) & (r > 0)
    if ok.sum() < 2 or len(np.unique(F[ok])) < 2:
        return None
    b, la = np.polyfit(np.log(F[ok]), np.log(r[ok]), 1)
    return float(np.exp(la)), float(b)


def fit_shared(per_pad):
    """One exponent b for all pads, one gain a per pad: least squares in log-log."""
    names, X, y = list(per_pad), [], []
    for i, n in enumerate(names):
        F, r = per_pad[n]
        ok = (F > 0) & (r > 0)
        for f, v in zip(F[ok], r[ok]):
            row = np.zeros(len(names) + 1)
            row[i], row[-1] = 1.0, np.log(f)
            X.append(row)
            y.append(np.log(v))
    if len(y) <= len(names):
        return None, {}
    sol = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0]
    return float(sol[-1]), {n: float(np.exp(sol[i])) for i, n in enumerate(names)}


def invert(r, a, b):
    return (np.clip(r, 1e-9, None) / a) ** (1.0 / b)


# ---------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    sessions = sessions_from(args.paths)
    root = args.paths[0] if len(args.paths) == 1 and not (args.paths[0] / "weights").exists() else sessions[0].parent
    out = args.out or root / "pad_calibration"
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for s in sessions for r in analyse_session(s, args)]
    if not args.all_sessions:                              # one session per pad: the most loaded steps, then latest
        best, dropped = {}, []
        for s in sorted({r["session"] for r in rows}):
            n_loaded = sum(1 for r in rows if r["session"] == s and r["mass_g"] > 0)
            pad = next(r["pad"] for r in rows if r["session"] == s)
            if pad not in best or n_loaded >= best[pad][1]:
                if pad in best:
                    dropped.append(best[pad][0])
                best[pad] = (s, n_loaded)
            else:
                dropped.append(s)
        if dropped:
            print(f"using one session per pad (--all-sessions pools them); left out: {', '.join(sorted(dropped))}")
            keep = {v[0] for v in best.values()}
            rows = [r for r in rows if r["session"] in keep]
            sessions = [s for s in sessions if s.name in keep]
    if args.max_mass is not None:
        rows = [r for r in rows if r["mass_g"] <= args.max_mass]
    if args.skip_reps:
        rows = [r for r in rows if r["rep"] > args.skip_reps]
    if not rows:
        raise SystemExit("no usable steps")
    shapes = [r["change"].shape for r in rows]
    shape = max(set(shapes), key=shapes.count)
    other = sorted({r["session"] for r in rows if r["change"].shape != shape})
    if other:
        print(f"left out {len(other)} session(s) recorded with a different grid than {shape[0]}x{shape[1]}: "
              + ", ".join(other))
        rows = [r for r in rows if r["change"].shape == shape]

    # ---- steps.csv
    cell_names = [f"R{i + 1}C{j + 1}" for i in range(shape[0]) for j in range(shape[1])]
    with (out / "steps.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "pad", "step", "rep", "mass_g", "load_N", "reference", "noise", "target",
                    "residual", "raw_adc", "clipped", "flags"] + [f"d_{n}" for n in cell_names])
        for r in rows:
            w.writerow([r["session"], r["pad"], r["step"], r["rep"], r["mass_g"], f"{r['load_N']:.4f}",
                        r["reference"], f"{r['noise']:.2f}", f"{r['target']:.1f}", f"{r['residual']:.1f}",
                        f"{r['raw']:.0f}", int(r["clipped"]),
                        "; ".join(r["flags"])] + [f"{v:.1f}" for v in r["change"].ravel()])

    # ---- per pad
    pads = sorted({r["pad"] for r in rows}, key=lambda n: tuple(map(int, re.findall(r"\d+", n))))
    per_pad, report, flagged = {}, {}, []
    for n in pads:
        pr_rows = [r for r in rows if r["pad"] == n]
        loaded = [r for r in pr_rows if r["mass_g"] > 0]
        flagged += [r for r in loaded if r["flags"]]
        use = [r for r in loaded if args.keep_flagged or not r["flags"]]
        masses = sorted({r["mass_g"] for r in use})
        means = {m_: float(np.mean([r["target"] for r in use if r["mass_g"] == m_])) for m_ in masses}
        stds = {m_: float(np.std([r["target"] for r in use if r["mass_g"] == m_])) for m_ in masses}
        # Saturation = the raw reading sat at the ADC rail (no headroom left), decided per step. A mass where at
        # least half the steps clipped marks the pad saturated above the next lighter mass; clipped steps are a
        # lower bound only and never enter the fit.
        sat_masses = [m_ for m_ in sorted({r["mass_g"] for r in loaded})
                      if 2 * sum(r["clipped"] for r in loaded if r["mass_g"] == m_)
                      >= sum(1 for r in loaded if r["mass_g"] == m_)]
        saturated_above = None
        if sat_masses:
            lighter = [m_ for m_ in masses if m_ < sat_masses[0]]
            saturated_above = lighter[-1] if lighter else sat_masses[0]
        n_clipped = sum(r["clipped"] for r in use)
        use = [r for r in use if not r["clipped"]]
        F = np.array([r["load_N"] for r in use])
        rv = np.array([r["target"] for r in use])
        per_pad[n] = (F, rv)
        tare = [r["target"] for r in pr_rows if r["mass_g"] == 0]
        report[n] = {"masses_g": masses, "mean_counts": means, "std_counts": stds,
                     "saturated_above_g": saturated_above, "fit": fit_power(F, rv),
                     "noise": float(np.median([r["noise"] for r in pr_rows])),
                     "tare_drift": float(np.mean(np.abs(tare))) if tare else None,
                     "n_used": len(use), "n_flagged": sum(1 for r in loaded if r["flags"]),
                     "n_clipped": n_clipped}
        # leave-one-repetition-out error in N with the pad's own power law
        errs = []
        for rep in sorted({r["rep"] for r in use}):
            tr = [r for r in use if r["rep"] != rep]
            te = [r for r in use if r["rep"] == rep]
            f = fit_power(np.array([r["load_N"] for r in tr]), np.array([r["target"] for r in tr]))
            if f and te:
                errs += [abs(float(invert(r["target"], *f)) - r["load_N"]) for r in te]
        report[n]["loro_mae_N"] = float(np.mean(errs)) if errs else None

    b_shared, a_shared = fit_shared(per_pad)

    # ---- crosstalk: change of every other cell as a fraction of the target's, at each pad's heaviest used mass
    xt = {}
    for n in pads:
        top = report[n]["masses_g"][-1] if report[n]["masses_g"] else None
        sel = [r for r in rows if r["pad"] == n and r["mass_g"] == top and (args.keep_flagged or not r["flags"])]
        if not sel:
            sel = [r for r in rows if r["pad"] == n and r["mass_g"] == top]
        if sel:
            mean_map = np.mean([r["change"] for r in sel], axis=0)
            xt[n] = {"mass_g": top, "map_counts": mean_map, "pr": sel[0]["pr"], "pc": sel[0]["pc"]}
            tgt = mean_map[sel[0]["pr"], sel[0]["pc"]]
            frac = mean_map / tgt if tgt > 0 else np.full(shape, np.nan)
            same_row = np.delete(frac[sel[0]["pr"]], sel[0]["pc"])
            same_col = np.delete(frac[:, sel[0]["pc"]], sel[0]["pr"])
            mask = np.ones(shape, bool)
            mask[sel[0]["pr"], :] = mask[:, sel[0]["pc"]] = False
            report[n]["crosstalk"] = {"same_row_max": float(np.nanmax(np.abs(same_row))) if same_row.size else None,
                                      "same_col_max": float(np.nanmax(np.abs(same_col))) if same_col.size else None,
                                      "elsewhere_max": float(np.nanmax(np.abs(frac[mask]))) if mask.any() else None}

    # ---- pads.csv + model.json
    with (out / "pads.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        masses_all = sorted({m_ for n in pads for m_ in report[n]["masses_g"]})
        w.writerow(["pad", "noise", "tare_drift"] + [f"{m_:g}g_mean" for m_ in masses_all]
                   + [f"{m_:g}g_std" for m_ in masses_all]
                   + ["saturated_above_g", "a", "b", "a_shared", "loro_mae_N", "xt_row", "xt_col", "xt_else",
                      "used", "flagged"])
        for n in pads:
            rp, f, xtr = report[n], report[n]["fit"], report[n].get("crosstalk", {})
            w.writerow([n, f"{rp['noise']:.1f}", "" if rp["tare_drift"] is None else f"{rp['tare_drift']:.1f}"]
                       + [f"{rp['mean_counts'][m_]:.0f}" if m_ in rp["mean_counts"] else "" for m_ in masses_all]
                       + [f"{rp['std_counts'][m_]:.1f}" if m_ in rp["std_counts"] else "" for m_ in masses_all]
                       + [rp["saturated_above_g"] or "", f"{f[0]:.4g}" if f else "", f"{f[1]:.3f}" if f else "",
                          f"{a_shared[n]:.4g}" if n in a_shared else "",
                          "" if rp["loro_mae_N"] is None else f"{rp['loro_mae_N']:.3f}"]
                       + ["" if xtr.get(k) is None else f"{xtr[k]:.2f}" for k in ("same_row_max", "same_col_max",
                                                                                  "elsewhere_max")]
                       + [rp["n_used"], rp["n_flagged"]])
    model = {"units": "native ADC counts, positive = pressure, change vs the unloaded seconds before each step",
             "model": "counts = a * F_N ** b ; F_N = (counts / a) ** (1 / b)",
             "b_shared": b_shared,
             "pads": {n: {"a": report[n]["fit"][0] if report[n]["fit"] else None,
                          "b": report[n]["fit"][1] if report[n]["fit"] else None,
                          "a_shared": a_shared.get(n),
                          "valid_up_to_g": report[n]["saturated_above_g"] or (max(report[n]["masses_g"])
                                                                              if report[n]["masses_g"] else None),
                          "loro_mae_N": report[n]["loro_mae_N"], "noise_counts": report[n]["noise"]}
                      for n in pads},
             "sessions": [s.name for s in sessions]}
    (out / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")

    plot_curves(out / "curves.png", pads, per_pad, report, b_shared, a_shared)
    plot_crosstalk(out / "crosstalk.png", pads, xt)

    # ---- console
    print(f"{len(sessions)} session(s), {len(rows)} steps -> {out}")
    print(f"{'pad':5s} {'noise':>5s}  " + "  ".join(f"{m_:>6g}g" for m_ in masses_all)
          + "   fit b   LORO err   crosstalk row/col/else")
    for n in pads:
        rp, f, xtr = report[n], report[n]["fit"], report[n].get("crosstalk", {})
        cells = "  ".join(f"{rp['mean_counts'][m_]:7.0f}" if m_ in rp["mean_counts"] else "      -"
                          for m_ in masses_all)
        xs = "/".join("-" if xtr.get(k) is None else f"{xtr[k]:.0%}"
                      for k in ("same_row_max", "same_col_max", "elsewhere_max"))
        sat = f"  SATURATED above {rp['saturated_above_g']:g} g" if rp["saturated_above_g"] else ""
        err = "-" if rp["loro_mae_N"] is None else f"{rp['loro_mae_N']:.2f} N"
        print(f"{n:5s} {rp['noise']:5.1f}  {cells}   {f[1] if f else float('nan'):5.2f}   {err:>8s}   {xs}{sat}")
    if b_shared is not None:
        print(f"shared exponent b = {b_shared:.2f} (counts grow like F^{b_shared:.2f})")
    if flagged:
        print(f"{len(flagged)} step(s) left out{' (kept: --keep-flagged)' if args.keep_flagged else ''}:")
        for r in flagged:
            print(f"   {r['pad']} {r['mass_g']:g} g rep {r['rep']}: " + "; ".join(r["flags"]))


def plot_curves(path, pads, per_pad, report, b_shared, a_shared):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(pads)
    cols = min(4, n)
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.4 * cols, 2.8 * rows_), squeeze=False, sharex=True)
    fmax = max((float(F.max()) for F, _ in per_pad.values() if F.size), default=1.0)
    grid_F = np.linspace(0, fmax * 1.05, 100)
    for ax, name in zip(axes.ravel(), pads):
        F, r = per_pad[name]
        ax.plot(F, r, "o", ms=5, color="#2a6fb0", alpha=0.8)
        f = report[name]["fit"]
        if f:
            ax.plot(grid_F, f[0] * grid_F ** f[1], color="#1f2328", lw=1.5, label=f"own b={f[1]:.2f}")
        if b_shared is not None and name in a_shared:
            ax.plot(grid_F, a_shared[name] * grid_F ** b_shared, color="#8c959f", lw=1.2, ls="--", label="shared b")
        title = name + (f"  (sat. > {report[name]['saturated_above_g']:g} g)" if report[name]["saturated_above_g"] else "")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("load (N)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("pad change (counts)", fontsize=9)
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Target pad response vs load (each step vs its own pre-step reference)", fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_crosstalk(path, pads, xt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    names = [n for n in pads if n in xt]
    if not names:
        return
    cols = min(4, len(names))
    rows_ = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.2 * cols, 3.2 * rows_), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        m = xt[name]["map_counts"]
        lim = max(float(np.abs(m).max()), 1.0)
        ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * lim else "#1f2328")
        ax.add_patch(Rectangle((xt[name]["pc"] - 0.5, xt[name]["pr"] - 0.5), 1, 1, fill=False, lw=2.5,
                               ec="#1f2328"))
        ax.set_xticks(range(m.shape[1]), [f"C{j + 1}" for j in range(m.shape[1])], fontsize=7)
        ax.set_yticks(range(m.shape[0]), [f"R{i + 1}" for i in range(m.shape[0])], fontsize=7)
        ax.set_title(f"{name} loaded, {xt[name]['mass_g']:g} g", fontsize=9, loc="left")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Change of every cell while one pad is loaded (counts, + = pressure; box = loaded pad)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
