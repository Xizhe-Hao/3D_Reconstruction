#!/usr/bin/env python3
"""Where is the weight and how heavy is it? Train on the dead-weight sessions, then run live.

    python weight_live.py train C:/tactile_data/weight_cal [--exclude R2C2]
    python weight_live.py live  C:/tactile_data/weight_cal/force_model.json --board esp32
    python weight_live.py replay C:/tactile_data/weight_cal/force_model.json <session>

train   fits the shared Hill shape with a gain per calibrated pad on the
        summed plateaus of every pad session under ROOT (see weight_models.py)
        and writes ROOT/force_model.json: shape parameters, gains, noise, the
        detection threshold and the expected error.
live    opens the board, takes a 5 s no-load baseline (keep the array empty),
        then on every frame: cells above the threshold are grouped around
        their local maxima (a weight's spill into a neighbour belongs to the
        peak), each group's summed counts go through the inverse curve with
        the peak pad's gain, and the map shows "R1C2  97 g (~100 g)" on the
        loaded pads with the +- band from the calibration. Several weights on
        several pads are reported separately. Pads without a calibration use
        the median gain and are marked "uncal".
        Keys: B re-take the baseline, S save a PNG, Q quit.
replay  runs the same estimator over a recorded session and plots estimated
        grams against the weights that were really on (weights/events.csv),
        so you can see how the model does before trusting it live.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Python files"))
from live_sensor_2d import BOARDS, Esp32Source  # noqa: E402
from weight_cal import G, intervals, read_events, refine_edges  # noqa: E402
from weight_models import Hill, load_points  # noqa: E402

STANDARD_G = (10.0, 20.0, 50.0, 100.0, 200.0)


# ---------------------------------------------------------------- model


class ForceModel:
    def __init__(self, spec: dict):
        self.cmax, self.k, self.n = spec["hill"]
        self.gains = {k: float(v) for k, v in spec["gains"].items()}
        self.median_gain = float(spec.get("median_gain", 1.0))
        self.threshold = float(spec.get("threshold_counts", 5.0))
        self.err_N = float(spec.get("expected_error_N", 0.2))
        self.noise = float(spec.get("noise_counts", 0.5))
        self.pads_rc = {k: (int(k[1]) - 1, int(k[3]) - 1) for k in self.gains}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def gain(self, r, c):
        return self.gains.get(f"R{r + 1}C{c + 1}"), f"R{r + 1}C{c + 1}"

    def force(self, counts, gain):
        """Inverse Hill: F from summed counts and the pad's gain; None when saturated."""
        u = counts / max(gain, 1e-6) / self.cmax
        if u >= 0.97:
            return None
        if u <= 0:
            return 0.0
        return float(self.k * (u / (1 - u)) ** (1 / self.n))

    def detect(self, delta):
        """Weights on the array: [{'pad', 'rc', 'counts', 'cells', 'force_N', 'grams', 'calibrated', 'saturated'}].

        Cells above the threshold are assigned to the neighbouring local
        maximum with the largest response, so a weight's spill is added to
        its own sum and two weights on adjacent pads stay two detections.
        """
        above = delta > self.threshold
        rows, cols = delta.shape
        peaks = []
        for r in range(rows):
            for c in range(cols):
                if not above[r, c]:
                    continue
                nb = [delta[rr, cc] for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                      if 0 <= rr < rows and 0 <= cc < cols]
                if all(delta[r, c] >= v for v in nb):
                    peaks.append((r, c))
        sums = {p: float(delta[p]) for p in peaks}
        members = {p: [p] for p in peaks}
        for r in range(rows):
            for c in range(cols):
                if not above[r, c] or (r, c) in sums:
                    continue
                cand = [(delta[rr, cc], (rr, cc)) for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                        if (rr, cc) in sums]
                if cand:
                    _, p = max(cand)
                    sums[p] += float(delta[r, c])
                    members[p].append((r, c))
        out = []
        for p, s in sums.items():
            g, name = self.gain(*p)
            calibrated = g is not None
            g = g if calibrated else self.median_gain
            f = self.force(s, g)
            out.append({"pad": name, "rc": p, "counts": s, "cells": len(members[p]),
                        "force_N": f, "grams": None if f is None else f / G * 1000.0,
                        "calibrated": calibrated, "saturated": f is None})
        return sorted(out, key=lambda d: -d["counts"])

    def nearest_standard(self, grams):
        if grams is None:
            return None
        return min(STANDARD_G, key=lambda s: abs(s - grams))


def train(args):
    exclude = {e.strip().upper() for e in args.exclude.split(",") if e.strip()}
    pts = load_points(args.root, exclude)
    if not pts:
        raise SystemExit("no pad session with report/weight_steps.csv under ROOT; run weight_summary.py first")
    model = Hill().fit(pts)
    # expected error: leave one repetition out on the training pads
    errs = []
    for rep in sorted({p[1] for p in pts}):
        m = Hill().fit([p for p in pts if p[1] != rep])
        for pad, r, F, c in pts:
            if r == rep:
                errs.append(float(m.predict(pad, c)[0]) - F)
    noise = []
    for s in args.root.glob("capture_*"):
        j = s / "report" / "weight_report.json"
        if j.exists():
            noise.append(json.loads(j.read_text()).get("noise_counts") or 0.0)
    noise_c = float(np.median(noise)) if noise else 0.5
    spec = {
        "trained_utc": datetime.utcnow().isoformat(), "root": str(args.root),
        "hill": list(map(float, model.theta)),
        "shape": "counts = gain * cmax * F^n / (F^n + k^n), F in N, counts = 8-bit-equivalent summed response",
        "gains": {p: float(g) for p, g in sorted(model.g.items())},
        "median_gain": float(np.median(list(model.g.values()))),
        "noise_counts": noise_c, "threshold_counts": max(5.0, 8 * noise_c),
        "expected_error_N": float(np.abs(errs).mean()) if errs else 0.2,
        "pads": sorted(model.g), "points": len(pts), "excluded": sorted(exclude),
    }
    out = args.root / "force_model.json"
    out.write_text(json.dumps(spec, indent=2))
    cmax, k, n = spec["hill"]
    print(f"trained on {len(pts)} plateaus from {len(model.g)} pads: counts = gain * {cmax:.1f} * F^{n:.2f} / "
          f"(F^{n:.2f} + {k:.2f}^{n:.2f}); gains " + ", ".join(f"{p} {g:.2f}" for p, g in sorted(model.g.items())))
    print(f"threshold {spec['threshold_counts']:.1f} counts (noise {noise_c:.2f}); expected error "
          f"+-{spec['expected_error_N']:.2f} N (+-{spec['expected_error_N'] / G * 1000:.0f} g)\nwrote {out}")
    return 0


# ---------------------------------------------------------------- estimator over a stream


class Estimator:
    """Baseline, per-cell change, temporal smoothing and detection for one frame stream."""

    def __init__(self, model: ForceModel, shape, adc_max, lowers_adc, baseline_s=5.0, smooth_s=1.0, fps=10.0):
        self.model = model
        self.scale = adc_max / 255.0
        self.lowers = lowers_adc
        self.baseline_s = baseline_s
        self.keep = max(1, int(smooth_s * fps))
        self.samples, self.base, self.t0 = [], None, None
        self.recent = []

    def reset(self):
        self.samples, self.base, self.t0 = [], None, None

    def push(self, grid, t):
        g = grid.astype(float) / self.scale
        if self.base is None:
            if self.t0 is None:
                self.t0 = t
            self.samples.append(g)
            if t - self.t0 >= self.baseline_s and len(self.samples) >= 5:
                self.base = np.median(np.stack(self.samples), axis=0)
            return None
        delta = (self.base - g) if self.lowers else (g - self.base)
        self.recent.append(delta)
        del self.recent[:-self.keep]
        smooth = np.median(np.stack(self.recent), axis=0)
        return smooth, self.model.detect(smooth)


def describe(det, model: ForceModel):
    if det["saturated"]:
        return f"{det['pad']}: >= saturation (~2 N)" + ("" if det["calibrated"] else "  uncal")
    g = det["grams"]
    err = model.err_N / G * 1000.0
    near = model.nearest_standard(g)
    return (f"{det['pad']}: {g:5.0f} g  +-{err:.0f}   (~{near:g} g)"
            + ("" if det["calibrated"] else "  uncal") + (f"  {det['cells']} cells" if det["cells"] > 1 else ""))


# ---------------------------------------------------------------- live


def live(args):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    model = ForceModel.load(args.model)
    source = Esp32Source(args.port, bits=args.bits, **({"roi": tuple(args.roi)} if args.roi else {}))
    source.start()
    est = Estimator(model, source.shape, source.adc_max, BOARDS["esp32"]["pressure_lowers_adc"],
                    args.baseline_s, args.smooth_s)
    rows, cols = source.shape
    fig = plt.figure(figsize=(11, 6))
    try:
        fig.canvas.manager.set_window_title("Weight finder")
    except AttributeError:
        pass
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0), left=0.06, right=0.97, top=0.9, bottom=0.1, wspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    img = ax.imshow(np.zeros(source.shape), cmap="magma", vmin=0, vmax=60, interpolation="nearest")
    ax.set_xticks(range(cols), [f"C{c + 1}" for c in range(cols)])
    ax.set_yticks(range(rows), [f"R{r + 1}" for r in range(rows)])
    fig.colorbar(img, ax=ax, shrink=0.8).set_label("change vs baseline (counts)")
    cell_txt = [[ax.text(c, r, "", ha="center", va="center", fontsize=9, color="w", fontweight="bold")
                 for c in range(cols)] for r in range(rows)]
    for name, (r, c) in model.pads_rc.items():
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="#4de8ff", lw=1.2, ls="--"))
    ax.set_title("calibrated pads dashed; loaded pads labelled in grams", fontsize=9)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    head = ax2.text(0.0, 0.95, "", va="top", fontsize=15, fontweight="bold", transform=ax2.transAxes)
    body = ax2.text(0.0, 0.8, "", va="top", fontsize=13, family="monospace", transform=ax2.transAxes)
    fig.text(0.5, 0.03, "B re-take baseline (array empty)   S snapshot   Q quit", ha="center", fontsize=9, color="#444")
    state = {"closed": False}

    def on_key(e):
        if e.key == "b":
            est.reset()
        elif e.key == "s":
            p = Path(args.model).parent / f"weight_live_{datetime.now():%Y%m%d_%H%M%S}.png"
            fig.savefig(p, dpi=130)
            print(f"snapshot {p}")
        elif e.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda e: state.update(closed=True))
    plt.ion()
    plt.show()
    t0 = time.perf_counter()
    try:
        while not state["closed"] and plt.fignum_exists(fig.number):
            s = source.read_latest()
            if s is not None:
                _, _, grid = s
                res = est.push(grid, time.perf_counter() - t0)
                if res is None:
                    head.set_text(f"baseline {min(time.perf_counter() - t0 - (est.t0 or 0), args.baseline_s):.1f}"
                                  f"/{args.baseline_s:g} s - keep the array empty")
                    head.set_color("#b8860b")
                    body.set_text("")
                else:
                    delta, dets = res
                    img.set_data(delta)
                    for r in range(rows):
                        for c in range(cols):
                            cell_txt[r][c].set_text("")
                    for d in dets:
                        r, c = d["rc"]
                        cell_txt[r][c].set_text("sat" if d["saturated"] else f"{d['grams']:.0f} g")
                    if dets:
                        head.set_text(f"{len(dets)} weight(s) on the array")
                        head.set_color("#1f5fbf")
                        body.set_text("\n".join(describe(d, model) for d in dets))
                    else:
                        head.set_text("nothing on the array")
                        head.set_color("#2a9d3a")
                        body.set_text("")
                fig.canvas.draw_idle()
            plt.pause(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    return 0


# ---------------------------------------------------------------- replay


def replay(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tactile_preprocess import frame_rate, load_raw, session_meta

    model = ForceModel.load(args.model)
    s = Path(args.session)
    ns, idx, frames = load_raw(s)
    t = (ns - ns[0]) / 1e9
    meta = session_meta(s, *frames.shape[1:])
    raw_meta = json.loads((s / "session.json").read_text(encoding="utf-8"))
    pad = raw_meta.get("pad_under_test", "?")
    est = Estimator(model, frames.shape[1:], 255.0, meta["pressure_lowers_adc"], args.baseline_s, args.smooth_s,
                    frame_rate(ns))
    grams_t, grams, where_ok, n_det = [], [], [], []
    for k in range(len(t)):
        res = est.push(frames[k], t[k])
        if res is None:
            continue
        _, dets = res
        grams_t.append(t[k])
        n_det.append(len(dets))
        if dets:
            d = dets[0]
            grams.append(np.nan if d["saturated"] else d["grams"])
            where_ok.append(d["pad"] == pad)
        else:
            grams.append(0.0)
            where_ok.append(True)
    grams_t, grams = np.array(grams_t), np.array(grams)
    # truth from the marks (edges refined on the pad's own response)
    rc = (int(pad[1]) - 1, int(pad[3]) - 1)
    base = np.median(frames[t <= 5], axis=0)
    sign = -1.0 if meta["pressure_lowers_adc"] else 1.0
    cell = sign * (frames[:, rc[0], rc[1]] - base[rc[0], rc[1]])
    truth = np.zeros(len(t))
    rows = []
    for step, mass, on_ns, off_ns in intervals(read_events(s)):
        a, b = refine_edges(t, cell, (on_ns - ns[0]) / 1e9, (off_ns - ns[0]) / 1e9 if off_ns else t[-1])
        truth[(t >= a) & (t <= b)] = mass
        m = (grams_t >= b - 5) & (grams_t <= b)
        est_g = float(np.nanmedian(grams[m])) if m.any() else np.nan
        rows.append((step, mass, est_g))
    err = np.array([e - m for _, m, e in rows if np.isfinite(e)])
    print(f"{s.name} ({pad}): {len(rows)} steps; estimated vs true grams, last 5 s of each hold:")
    for step, mass, est_g in rows:
        print(f"  step {step:2d}  {mass:5.0f} g  ->  {est_g:6.0f} g   nearest {model.nearest_standard(est_g) if np.isfinite(est_g) else '-'}")
    print(f"MAE {np.abs(err).mean():.0f} g; pad identified correctly in {np.mean(where_ok) * 100:.0f}% of loaded frames"
          if err.size else "no steps")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t, truth, color="k", lw=1.2, label="true weight on the pad")
    ax.plot(grams_t, grams, color="tab:red", lw=0.9, label="estimated (largest detection)")
    ax.set(xlabel="time [s]", ylabel="grams", title=f"{s.name}: live estimator replay on {pad}; MAE {np.abs(err).mean():.0f} g")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    out = s / "report" / "weight_live_replay.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("train")
    a.add_argument("root", type=Path)
    a.add_argument("--exclude", default="")
    b = sub.add_parser("live")
    b.add_argument("model", type=Path)
    b.add_argument("--board", choices=("esp32",), default="esp32")
    b.add_argument("--port", default=None)
    b.add_argument("--bits", type=int, choices=(8, 12), default=None)
    b.add_argument("--roi", type=int, nargs=4, metavar=("ROW0", "COL0", "ROWS", "COLS"), default=None,
                   help="window of the 32x32 frame holding wired pads (default 0 0 4 4); e.g. 0 0 6 6")
    b.add_argument("--baseline-s", type=float, default=5.0)
    b.add_argument("--smooth-s", type=float, default=1.0, help="median over this many seconds of frames")
    c = sub.add_parser("replay")
    c.add_argument("model", type=Path)
    c.add_argument("session", type=Path)
    c.add_argument("--baseline-s", type=float, default=5.0)
    c.add_argument("--smooth-s", type=float, default=1.0)
    args = p.parse_args()
    return {"train": train, "live": live, "replay": replay}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
