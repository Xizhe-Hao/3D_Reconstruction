#!/usr/bin/env python3
"""Which force model for the dead-weight pads? Three candidates, two held-out tests, one figure.

    python weight_models.py C:/tactile_data/weight_cal [--exclude R2C2] [--out DIR]

Reads report/weight_steps.csv of every pad session under ROOT (run
weight_summary.py first) and fits, on the summed plateau counts c and the
load F:

  global     one monotone curve F = h(c) for every pad (ignores pad gain)
  per-pad    an independent monotone curve per pad (what "a model per pad" means)
  shared     one shape with a gain per pad: c = g_pad * s(F), inverted per pad;
             s is isotonic, g_pad a scalar (alternating least squares)
  hill       the same, with a smooth 3-parameter saturating shape
             s(F) = cmax * F^n / (F^n + k^n) fitted to all pads at once

Tests:
  LORO  leave one repetition out: fit on the other repetitions of every pad,
        predict the held-out repetition (what the model does on a pad it has
        seen; placement error is the floor)
  LOPO  leave one pad out: fit on the other pads, estimate the held-out pad's
        gain from ONE weight of its first repetition (--gain-from, default
        100 g), predict all its steps (what the model does on a new pad)

Writes ROOT/weight_models.png and weight_models.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--exclude", default="", help="pads to leave out of the shared fits, e.g. R2C2")
    p.add_argument("--gain-from", type=float, default=100.0, help="g; the one weight used for a new pad's gain")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


# ---------------------------------------------------------------- data


def load_points(root: Path, exclude):
    """[(pad, rep, load_N, counts)] from every pad session's weight_steps.csv (latest session per pad)."""
    by_pad = {}
    for s in sorted(root.glob("capture_*")):
        steps = s / "report" / "weight_steps.csv"
        meta = s / "session.json"
        if not steps.exists() or not meta.exists():
            continue
        pad = json.loads(meta.read_text(encoding="utf-8")).get("pad_under_test")
        if not pad or pad in exclude:
            continue
        rows = list(csv.DictReader(steps.open()))
        pts, seen = [], {}
        for r in rows:
            if r["sum_counts"] == "":
                continue
            mass = float(r["mass_g"])
            seen[mass] = seen.get(mass, 0) + 1
            pts.append((pad, seen[mass], float(r["load_N"]), float(r["sum_counts"])))
        if pts:
            by_pad[pad] = pts  # sorted by name = time; the latest session of a pad wins
    return [p for pts in by_pad.values() for p in pts]


# ---------------------------------------------------------------- models


def iso(x, y):
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)


class Global:
    name = "global"

    def fit(self, pts):
        self.h = iso([p[3] for p in pts], [p[2] for p in pts])
        return self

    def predict(self, pad, counts):
        return self.h.predict(np.atleast_1d(counts))

    def curve(self, pad, loads):  # counts expected at these loads (for drawing): invert h on a grid
        grid = np.linspace(0, max(self.h.X_thresholds_) * 1.1, 400)
        return np.interp(loads, self.h.predict(grid), grid)


class PerPad:
    name = "per-pad"

    def fit(self, pts):
        self.h = {}
        for pad in {p[0] for p in pts}:
            own = [p for p in pts if p[0] == pad]
            if len(own) >= 3:
                self.h[pad] = iso([p[3] for p in own], [p[2] for p in own])
        return self

    def predict(self, pad, counts):
        if pad not in self.h:
            return np.full(np.atleast_1d(counts).shape, np.nan)
        return self.h[pad].predict(np.atleast_1d(counts))

    def curve(self, pad, loads):
        if pad not in self.h:
            return np.full(len(loads), np.nan)
        grid = np.linspace(0, max(self.h[pad].X_thresholds_) * 1.1, 400)
        return np.interp(loads, self.h[pad].predict(grid), grid)


class Shared:
    """c = g_pad * s(F); s isotonic in F; alternating fit; inverted by interpolation on a grid."""

    name = "shared"

    def fit(self, pts, n_iter=25):
        pads = sorted({p[0] for p in pts})
        self.g = {p: 1.0 for p in pads}
        F = np.array([p[2] for p in pts])
        c = np.array([p[3] for p in pts])
        who = np.array([p[0] for p in pts], dtype=object)
        for _ in range(n_iter):
            gv = np.array([self.g[p] for p in who])
            self.s = iso(F, c / gv)
            sF = self.s.predict(F)
            for p in pads:
                m = who == p
                d = float((sF[m] ** 2).sum())
                self.g[p] = float((c[m] * sF[m]).sum() / d) if d > 0 else 1.0
            med = float(np.median(list(self.g.values()))) or 1.0
            self.g = {p: v / med for p, v in self.g.items()}
        gv = np.array([self.g[p] for p in who])
        self.s = iso(F, c / gv)
        self.grid = np.linspace(0, F.max() * 1.05, 600)
        self.sg = self.s.predict(self.grid)
        return self

    def gain_from(self, load, counts):
        """A new pad's gain from one (load, counts) pair."""
        return float(counts / max(float(self.s.predict([load])[0]), 1e-6))

    def predict(self, pad, counts, gain=None):
        g = self.g.get(pad, 1.0) if gain is None else gain
        return np.interp(np.atleast_1d(counts) / max(g, 1e-6), self.sg, self.grid)

    def curve(self, pad, loads, gain=None):
        g = self.g.get(pad, 1.0) if gain is None else gain
        return g * self.s.predict(loads)


class Hill(Shared):
    """c = g_pad * cmax * F^n / (F^n + k^n): smooth saturating shape shared by all pads."""

    name = "hill"

    @staticmethod
    def shape(F, cmax, k, n):
        F = np.maximum(F, 1e-9)
        return cmax * F ** n / (F ** n + k ** n)

    def fit(self, pts, n_iter=25):
        from scipy.optimize import curve_fit

        pads = sorted({p[0] for p in pts})
        self.g = {p: 1.0 for p in pads}
        F = np.array([p[2] for p in pts])
        c = np.array([p[3] for p in pts])
        who = np.array([p[0] for p in pts], dtype=object)
        self.theta = (max(c.max(), 1.0), max(np.median(F), 0.1), 1.0)
        for _ in range(n_iter):
            gv = np.array([self.g[p] for p in who])
            try:
                self.theta, _ = curve_fit(self.shape, F, c / gv, p0=self.theta,
                                          bounds=([1.0, 0.01, 0.2], [1e4, 50.0, 4.0]), maxfev=20000)
            except RuntimeError:
                pass
            sF = self.shape(F, *self.theta)
            for p in pads:
                m = who == p
                d = float((sF[m] ** 2).sum())
                self.g[p] = float((c[m] * sF[m]).sum() / d) if d > 0 else 1.0
            med = float(np.median(list(self.g.values()))) or 1.0
            self.g = {p: v / med for p, v in self.g.items()}
        self.grid = np.linspace(0, F.max() * 1.05, 600)
        self.sg = self.shape(self.grid, *self.theta)
        return self

    def gain_from(self, load, counts):
        return float(counts / max(float(self.shape(np.array([load]), *self.theta)[0]), 1e-6))

    def curve(self, pad, loads, gain=None):
        g = self.g.get(pad, 1.0) if gain is None else gain
        return g * self.shape(np.asarray(loads, dtype=float), *self.theta)


class PerPadHill:
    """An independent smooth Hill curve per pad (3 parameters from ~15 points): interpolates, unlike isotonic."""

    name = "per-pad hill"

    def fit(self, pts):
        from scipy.optimize import curve_fit

        self.theta = {}
        for pad in {p[0] for p in pts}:
            own = [p for p in pts if p[0] == pad]
            if len(own) < 4:
                continue
            F = np.array([p[2] for p in own])
            c = np.array([p[3] for p in own])
            try:
                th, _ = curve_fit(Hill.shape, F, c, p0=(max(c.max(), 1.0), max(np.median(F), 0.1), 1.0),
                                  bounds=([1.0, 0.01, 0.2], [1e4, 50.0, 4.0]), maxfev=20000)
                self.theta[pad] = th
            except RuntimeError:
                pass
        self.grid = np.linspace(0, max(p[2] for p in pts) * 1.05, 600)
        return self

    def predict(self, pad, counts):
        if pad not in self.theta:
            return np.full(np.atleast_1d(counts).shape, np.nan)
        return np.interp(np.atleast_1d(counts), Hill.shape(self.grid, *self.theta[pad]), self.grid)

    def curve(self, pad, loads):
        if pad not in self.theta:
            return np.full(len(loads), np.nan)
        return Hill.shape(np.asarray(loads, dtype=float), *self.theta[pad])


MODELS = (Global, PerPad, PerPadHill, Shared, Hill)


# ---------------------------------------------------------------- tests


def loro(pts, Model):
    """leave one repetition out -> list of (pad, rep, load, pred)."""
    out = []
    for rep in sorted({p[1] for p in pts}):
        train = [p for p in pts if p[1] != rep]
        test = [p for p in pts if p[1] == rep]
        if len({p[0] for p in train}) == 0:
            continue
        m = Model().fit(train)
        for pad, r, F, c in test:
            out.append((pad, r, F, float(m.predict(pad, c)[0])))
    return out


def lopo(pts, Model, gain_from_g):
    """leave one pad out; its gain from one weight of its first repetition -> (pad, rep, load, pred)."""
    out = []
    pads = sorted({p[0] for p in pts})
    if len(pads) < 2:
        return out
    for pad in pads:
        train = [p for p in pts if p[0] != pad]
        test = [p for p in pts if p[0] == pad]
        m = Model().fit(train)
        if isinstance(m, Shared):
            anchor = [p for p in test if abs(p[2] - gain_from_g * 9.80665 / 1000) < 1e-3 and p[1] == 1]
            anchor = anchor or [max(test, key=lambda p: p[2])]
            g = m.gain_from(anchor[0][2], anchor[0][3])
            for _, r, F, c in test:
                out.append((pad, r, F, float(m.predict(pad, c, gain=g)[0])))
        elif isinstance(m, Global):
            for _, r, F, c in test:
                out.append((pad, r, F, float(m.predict(pad, c)[0])))
        # PerPad has nothing to say about an unseen pad
    return out


def mae(res):
    e = np.array([r[3] - r[2] for r in res if np.isfinite(r[3])])
    return float(np.abs(e).mean()) if e.size else np.nan


def mae_by_pad(res):
    out = {}
    for pad in sorted({r[0] for r in res}):
        out[pad] = mae([r for r in res if r[0] == pad])
    return out


# ---------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    exclude = {e.strip().upper() for e in args.exclude.split(",") if e.strip()}
    pts = load_points(args.root, exclude)
    if not pts:
        raise SystemExit("no pad session with report/weight_steps.csv; run weight_summary.py first")
    pads = sorted({p[0] for p in pts})
    loads = sorted({p[2] for p in pts})
    span = max(loads) - min(loads)
    print(f"{len(pads)} pads ({', '.join(pads)}), {len(pts)} plateau points, loads "
          f"{', '.join(f'{l:.2f}' for l in loads)} N" + (f"; excluded {sorted(exclude)}" if exclude else ""))

    results, table = {}, {}
    for Model in MODELS:
        a = loro(pts, Model)
        b = lopo(pts, Model, args.gain_from)
        results[Model.name] = {"LORO": a, "LOPO": b}
        table[Model.name] = {"LORO": mae(a), "LOPO": mae(b) if b else np.nan,
                             "LORO_by_pad": mae_by_pad(a), "LOPO_by_pad": mae_by_pad(b) if b else {}}
    print(f"\n{'model':13s} {'LORO MAE':>9s} {'LOPO MAE':>9s}   (N; span {span:.2f} N; LOPO gain from one "
          f"{args.gain_from:g} g weight)")
    for name, row in table.items():
        print(f"{name:13s} {row['LORO']:9.3f} {row['LOPO']:9.3f}   "
              + "  ".join(f"{p} {v:.2f}" for p, v in row["LORO_by_pad"].items()))
    full = {Model.name: Model().fit(pts) for Model in MODELS}
    gains = {name: m.g for name, m in full.items() if isinstance(m, Shared)}
    if "hill" in full:
        cmax, k, n = full["hill"].theta
        print(f"hill shape: counts = gain * {cmax:.1f} * F^{n:.2f} / (F^{n:.2f} + {k:.2f}^{n:.2f})  "
              f"(half-saturation at {k:.2f} N)")
    for name, g in gains.items():
        print(f"{name} gains: " + "  ".join(f"{p} {v:.2f}" for p, v in sorted(g.items())))

    out = args.out or args.root
    out.mkdir(parents=True, exist_ok=True)
    (out / "weight_models.json").write_text(json.dumps({
        "pads": pads, "loads_N": loads, "excluded": sorted(exclude), "gain_from_g": args.gain_from,
        "mae_N": {k: {kk: (None if isinstance(vv, float) and not np.isfinite(vv) else vv) for kk, vv in v.items()}
                  for k, v in table.items()},
        "gains": gains, "hill_theta": list(map(float, full["hill"].theta)) if "hill" in full else None,
    }, indent=2, default=float))
    plot(out / "weight_models.png", pts, pads, loads, full, results, table, span)
    print(f"wrote {out / 'weight_models.png'}, weight_models.json")


def plot(path, pts, pads, loads, full, results, table, span):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"global": "#888888", "per-pad": "tab:orange", "per-pad hill": "tab:red", "shared": "tab:blue",
              "hill": "tab:green"}
    n = len(pads)
    cols = min(n, 4)
    rows_ = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(4.2 * cols, 3.6 * rows_ + 7.5))
    gs = fig.add_gridspec(rows_ + 2, cols, height_ratios=[1.0] * rows_ + [1.1, 1.1])
    Fgrid = np.linspace(0.02, max(loads) * 1.05, 300)
    for i, pad in enumerate(pads):
        a = fig.add_subplot(gs[i // cols, i % cols])
        own = [p for p in pts if p[0] == pad]
        for rep in sorted({p[1] for p in own}):
            rr = [p for p in own if p[1] == rep]
            a.scatter([p[2] for p in rr], [p[3] for p in rr], s=26, marker="osD^v"[(rep - 1) % 5], color="k",
                      alpha=0.75, label=f"rep {rep}" if i == 0 else None)
        for name, m in full.items():
            a.plot(Fgrid, m.curve(pad, Fgrid), color=colors[name], lw=1.5,
                   label=name if i == 0 else None)
        g = full["shared"].g.get(pad, np.nan)
        a.set_title(f"{pad}  (shared gain {g:.2f})", fontsize=10)
        a.set(xlabel="load [N]", ylabel="summed plateau [counts]")
        a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=7, frameon=False)

    for j, split in enumerate(("LORO", "LOPO")):
        a = fig.add_subplot(gs[rows_, j * (cols // 2):(j + 1) * (cols // 2)] if cols >= 2 else gs[rows_, :])
        lim = max(loads) * 1.1
        a.plot([0, lim], [0, lim], color="k", lw=0.8)
        for name, res in results.items():
            r = res[split]
            if not r:
                continue
            jit = (list(colors).index(name) - 2.0) * 0.012 * lim
            a.scatter([x[2] + jit for x in r], [x[3] for x in r], s=14, color=colors[name], alpha=0.7,
                      label=f"{name}: MAE {table[name][split]:.3f} N")
        a.set(xlim=(0, lim), ylim=(-0.1 * lim, lim), xlabel="true load [N]", ylabel="predicted [N]",
              title=("leave one repetition out (seen pads)" if split == "LORO"
                     else "leave one pad out (new pad, gain from one weight)"))
        a.legend(fontsize=8, frameon=False)
        a.grid(alpha=0.3)

    a = fig.add_subplot(gs[rows_ + 1, :])
    x = np.arange(len(loads))
    w = 0.8 / len(results)
    for k, (name, res) in enumerate(results.items()):
        r = res["LORO"]
        vals = []
        for F in loads:
            e = [abs(q[3] - q[2]) for q in r if abs(q[2] - F) < 1e-6 and np.isfinite(q[3])]
            vals.append(np.mean(e) if e else np.nan)
        a.bar(x + (k - len(results) / 2 + 0.5) * w, vals, w, color=colors[name], label=name)
    a.set_xticks(x, [f"{F:.2f} N" for F in loads])
    a.set(ylabel="MAE [N]", title="leave-one-repetition-out error by load (all pads)")
    a.legend(fontsize=8, frameon=False)
    a.grid(axis="y", alpha=0.3)
    fig.suptitle(f"dead-weight force models, {len(pads)} pads, span {span:.2f} N", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
