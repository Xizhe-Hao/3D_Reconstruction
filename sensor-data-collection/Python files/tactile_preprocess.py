#!/usr/bin/env python3
"""Shared clean-up and features for 16x16 tactile frames (force calibration / ML).

    from tactile_preprocess import load_raw, baseline, clean, frame_features, history_features

What the 2026-10-16 force runs (R5C5, R2C7 twice, R7C6) say about the signal:

- an unloaded cell's noise is 0.17 counts std -- 8-bit quantisation, nothing to
  filter away;
- crosstalk is a step, not a fraction of the press: once a cell makes contact
  the rest of its row reads +1.0-1.9 counts and its column +0.3-0.6, the same at
  30 counts as at 250, and direct neighbours +1-3 (more is real load sharing);
- a real contact jumps straight past ~30 counts: the pads snap shut at ~0.8 N.

So a deadband of DEADBAND counts on the baseline-subtracted frame removes noise
and ghosting without losing a contact, and temporal smoothing would only add
lag at 6 fps. What the deadband cannot fix -- the dead zone, hysteresis, creep
and saturation -- is left to the history features and the model.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROWS = COLS = 16
DEADBAND = 4.0          # counts; ghosting <= 3, real contact >= ~30
FPS = 6.0
LAGS_S = (0.5, 1.0, 2.0, 4.0)


def load_raw(session: Path):
    """(host_ns, frame_index, frames[N,16,16] float) from <session>/sensor_raw.csv."""
    with (Path(session) / "sensor_raw.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        head = next(reader)
        cells = [head.index(f"r{r}c{c}_adc8") for r in range(1, ROWS + 1) for c in range(1, COLS + 1)]
        i_ns, i_idx = head.index("host_received_ns"), head.index("frame_index")
        ns, idx, frames = [], [], []
        for row in reader:
            if row:
                ns.append(int(row[i_ns]))
                idx.append(int(row[i_idx]))
                frames.append([float(row[i]) for i in cells])
    return np.asarray(ns, dtype=np.int64), np.asarray(idx), np.asarray(frames).reshape(-1, ROWS, COLS)


def baseline(frames: np.ndarray, t: np.ndarray, seconds: float = 5.0) -> np.ndarray:
    """Per-cell no-load level: the median of the first `seconds` (keep the sensor unloaded)."""
    return np.median(frames[t <= seconds], axis=0)


def clean(frames: np.ndarray, base: np.ndarray, deadband: float = DEADBAND) -> np.ndarray:
    """Baseline-subtracted counts with |change| <= deadband set to 0. Works on one frame or a stack."""
    delta = frames - base
    return np.where(np.abs(delta) > deadband, delta, 0.0)


def frame_features(delta: np.ndarray) -> dict[str, np.ndarray]:
    """Current-frame features of a cleaned stack [N,16,16], around each frame's peak cell."""
    n = len(delta)
    flat = delta.reshape(n, -1)
    k = flat.argmax(axis=1)
    pr, pc = np.divmod(k, COLS)
    peak = flat[np.arange(n), k]
    padded = np.pad(delta, ((0, 0), (2, 2), (2, 2)))
    sum3 = np.array([padded[i, pr[i] + 1:pr[i] + 4, pc[i] + 1:pc[i] + 4].sum() for i in range(n)])
    sum5 = np.array([padded[i, pr[i]:pr[i] + 5, pc[i]:pc[i] + 5].sum() for i in range(n)])
    positive = np.clip(delta, 0, None)
    return {
        "peak": peak,
        "sum3": sum3,
        "sum5": sum5,
        "total": positive.sum(axis=(1, 2)),
        "n_active": (delta > 0).sum(axis=(1, 2)).astype(float),
        "peak_share": np.divide(peak, sum3, out=np.zeros(n), where=sum3 > 0),
    }


def history_features(peak: np.ndarray, fps: float = FPS) -> dict[str, np.ndarray]:
    """Causal features of the peak series that make hysteresis and creep learnable.

    The same reading means different forces on the way up and on the way down,
    and it creeps during a hold, so a model needs where the press has been:
    lagged values, the maximum since contact, how long contact has lasted and
    how far below that maximum the cell now is.
    """
    n = len(peak)
    out: dict[str, np.ndarray] = {}
    for s in LAGS_S:
        lag = int(round(s * fps))
        shifted = np.concatenate([np.full(lag, peak[0]), peak[:-lag]]) if lag < n else np.full(n, peak[0])
        out[f"lag_{s:g}s"] = shifted
        out[f"diff_{s:g}s"] = peak - shifted
    max_since = np.zeros(n)
    since_on = np.zeros(n)
    since_off = np.full(n, 60.0)
    run_max, on_t, off_t = 0.0, None, None
    for i, p in enumerate(peak):
        if p > 0:
            if on_t is None:
                on_t, run_max = i, 0.0
            run_max = max(run_max, p)
            off_t = None
        else:
            if on_t is not None:
                off_t = i
            on_t, run_max = None, 0.0
        max_since[i] = run_max
        since_on[i] = 0.0 if on_t is None else (i - on_t) / fps
        since_off[i] = 60.0 if off_t is None else min(60.0, (i - off_t) / fps)
    out["max_since_contact"] = max_since
    out["below_max"] = max_since - peak
    out["contact_s"] = since_on
    out["released_s"] = since_off
    return out
