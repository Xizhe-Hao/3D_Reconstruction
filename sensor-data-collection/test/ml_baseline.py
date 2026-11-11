#!/usr/bin/env python3
"""Starting point for ML on the dead-weight dataset: two sklearn models, the two honest splits.

    python ml_baseline.py C:/tactile_data/weight_cal/ml_dataset.npz [--frames hold|loaded|all]

Targets:  mass_g (regression, grams) and pad (which cell carries the weight).
Features: the 16 baseline-subtracted cell responses of one frame (X[i].ravel()).
Splits:   LORO  leave one repetition out (train on the other repetitions of every pad)
          LOPO  leave one pad out (train on the other pads)
Frames:   hold = plateau frames only (default, what the physics model is scored on),
          loaded = every frame with a weight on (includes the creep after placement),
          all = also the empty frames (mass 0), for a detector.
Prints MAE in grams per model and split, and the pad accuracy of a classifier.
Numbers to beat (plateaus, 4 healthy pads, 2026-10-31): Hill shape + gain per pad,
MAE 22 g LORO; a new pad from one 100 g weight, about 20 g LOPO.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("npz", type=Path)
    p.add_argument("--frames", choices=("hold", "loaded", "all"), default="hold")
    p.add_argument("--exclude", default="R2C2", help="pads to drop (damaged), comma separated; '' keeps all")
    p.add_argument("--models", default="gbm,mlp")
    return p.parse_args()


def rep_of(session, y, pad):
    """Repetition index of every frame: the k-th time this mass appears in this session's loaded stretches."""
    rep = np.zeros(len(y), dtype=int)
    for s in np.unique(session):
        m = session == s
        idx = np.flatnonzero(m)
        seen = {}
        prev = 0.0
        for i in idx:
            if y[i] > 0 and prev == 0.0:  # a new placement starts
                seen[y[i]] = seen.get(y[i], 0) + 1
            rep[i] = seen.get(y[i], 0) if y[i] > 0 else 0
            prev = y[i]
    return rep


def make(name):
    if name == "gbm":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=10)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000,
                                                            random_state=0, early_stopping=True))
    raise SystemExit(f"unknown model {name}")


def main() -> None:
    args = parse_args()
    d = np.load(args.npz, allow_pickle=False)
    X, y, pad, phase, sess = d["X"], d["y_mass"], d["y_pad"].astype(str), d["phase"].astype(str), d["session"].astype(str)
    X = X.reshape(len(X), -1)
    keep = ~np.isin(pad, [e.strip() for e in args.exclude.split(",") if e.strip()])
    if args.frames == "hold":
        keep &= phase == "hold"
    elif args.frames == "loaded":
        keep &= y > 0
    X, y, pad, sess = X[keep], y[keep], pad[keep], sess[keep]
    rep = rep_of(sess, y, pad)
    print(f"{len(y)} frames ({args.frames}), pads {sorted(set(map(str, pad)))}, "
          f"masses {sorted({int(v) for v in y})} g")

    for name in args.models.split(","):
        for split in ("LORO", "LOPO"):
            groups = rep if split == "LORO" else pad
            errs = []
            for g in sorted(set(groups)):
                if split == "LORO" and g == 0:
                    continue
                tr, te = groups != g, groups == g
                if tr.sum() < 20 or te.sum() == 0:
                    continue
                model = make(name).fit(X[tr], y[tr])
                errs.append(np.abs(model.predict(X[te]) - y[te]))
            e = np.concatenate(errs) if errs else np.array([np.nan])
            print(f"{name:4s} {split}: MAE {np.nanmean(e):5.1f} g  (median {np.nanmedian(e):5.1f} g, n {e.size})")

    # which pad: a classifier on the 16 responses vs the trivial argmax
    loaded = y > 0
    if loaded.sum() > 50 and len(set(pad[loaded])) > 1:
        cells = [f"R{r + 1}C{c + 1}" for r in range(4) for c in range(4)]
        argmax_ok = np.mean([cells[k] == p for k, p in zip(X[loaded].argmax(axis=1), pad[loaded])])
        accs = []
        for g in sorted(set(rep[loaded])):
            tr = loaded & (rep != g)
            te = loaded & (rep == g)
            if g == 0 or te.sum() == 0:
                continue
            clf = HistGradientBoostingClassifier(max_iter=200).fit(X[tr], pad[tr])
            accs.append(np.mean(clf.predict(X[te]) == pad[te]))
        print(f"pad: argmax of the 16 cells {argmax_ok * 100:.1f}% correct; GBM classifier LORO "
              f"{np.mean(accs) * 100:.1f}%" if accs else f"pad: argmax {argmax_ok * 100:.1f}%")


if __name__ == "__main__":
    main()
