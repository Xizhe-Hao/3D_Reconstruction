#!/usr/bin/env python3
"""Shared loading, clean-up and features for tactile frames (force calibration / ML).

    from tactile_preprocess import load_raw, session_meta, frame_rate, baseline, clean, frame_features, history_features

Nothing here assumes a board. The grid size comes from the sensor_raw.csv
header (r{R}c{C}_adc8 columns, row-major), the frame rate from the host
timestamps, and the press polarity from the data: the 16x16 M16B board's ADC
count rises under load, the 4x4 ESP32 + AD7466 board's falls. clean() returns
pressure-positive counts for both. Board facts the data cannot tell (ADC
reference, pad size, pitch, whether a rate was intended) come from the
session's session.json through session_meta(); sessions without one are the
16x16 board, which is every capture before 2026-10-30.

What the 2026-10-16 force runs on the 16x16 board (R5C5, R2C7 twice, R7C6) say
about the signal:

- an unloaded cell's noise is 0.17 counts std -- 8-bit quantisation, nothing to
  filter away;
- crosstalk is a step, not a fraction of the press: once a cell makes contact
  the rest of its row reads +1.0-1.9 counts and its column +0.3-0.6, the same at
  30 counts as at 250, and direct neighbours +1-3 (more is real load sharing);
- a real contact jumps straight past ~30 counts: the pads snap shut at ~0.8 N.

So a deadband of DEADBAND counts on the baseline-subtracted frame removes noise
and ghosting without losing a contact, and temporal smoothing would only add
lag at 6-8 fps. What the deadband cannot fix -- the dead zone, hysteresis, creep
and saturation -- is left to the history features and the model.

The 4x4 ESP32 board, unloaded for 12 s on 2026-10-30: per-cell noise 0.0-0.26
counts std, no drift, 8 fps free-running. Its crosstalk under load is not
measured yet, so the same deadband applies until a force run says otherwise.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROWS = COLS = 16        # the 16x16 M16B board; data files carry their own size
DEADBAND = 4.0          # counts; ghosting <= 3, real contact >= ~30 (16x16 board)
FPS = 6.0               # nominal rate of the camera rig; frame_rate() measures the real one
LAGS_S = (0.5, 1.0, 2.0, 4.0)

CELL_COLUMN = re.compile(r"^r(\d+)c(\d+)_adc8$")

# Board facts for sessions without a session.json (every capture before 2026-10-30).
M16B_DEFAULTS = {
    "board": "m16b", "protocol": "M16B v1", "rows": 16, "cols": 16,
    "adc_reference_v": 5.0, "sensor_drive_v": 5.0, "pressure_lowers_adc": False,
    "pad_mm": 5.0, "pitch_mm": 7.5, "expected_fps": 6.0,
}
META_KEYS = ("rows", "cols", "adc_reference_v", "sensor_drive_v", "pressure_lowers_adc",
             "pad_mm", "pitch_mm", "expected_fps")


def grid_columns(head: list[str]) -> tuple[list[int], int, int]:
    """Column indices of every cell in row-major order, and (rows, cols), from a sensor_raw.csv header."""
    found: dict[tuple[int, int], int] = {}
    for i, name in enumerate(head):
        m = CELL_COLUMN.match(name.strip())
        if m:
            found[(int(m.group(1)), int(m.group(2)))] = i
    if not found:
        raise ValueError("no r{R}c{C}_adc8 columns in the header")
    rows, cols = max(r for r, _ in found), max(c for _, c in found)
    missing = [(r, c) for r in range(1, rows + 1) for c in range(1, cols + 1) if (r, c) not in found]
    if missing:
        raise ValueError(f"grid is {rows}x{cols} but {len(missing)} cell columns are missing, "
                         f"e.g. r{missing[0][0]}c{missing[0][1]}_adc8")
    return [found[(r, c)] for r in range(1, rows + 1) for c in range(1, cols + 1)], rows, cols


def load_raw(session: Path):
    """(host_ns, frame_index, frames[N,R,C] float) from <session>/sensor_raw.csv; R x C from the header."""
    with (Path(session) / "sensor_raw.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        head = next(reader)
        cells, rows, cols = grid_columns(head)
        i_ns, i_idx = head.index("host_received_ns"), head.index("frame_index")
        ns, idx, frames = [], [], []
        for row in reader:
            if row:
                ns.append(int(row[i_ns]))
                idx.append(int(row[i_idx]))
                frames.append([float(row[i]) for i in cells])
    return (np.asarray(ns, dtype=np.int64), np.asarray(idx),
            np.asarray(frames, dtype=float).reshape(-1, rows, cols))


def session_meta(session: Path, rows: int | None = None, cols: int | None = None) -> dict:
    """Board facts of a session: its session.json (bench recorder or capture-script layout) over the defaults.

    Keys: board, protocol, rows, cols, adc_reference_v, sensor_drive_v,
    pressure_lowers_adc, pad_mm, pitch_mm, expected_fps (None: free-running,
    there is no rate to check against). Pass the grid read from the CSV so a
    session without session.json is recognised as the 16x16 board.
    """
    raw: dict = {}
    path = Path(session) / "session.json"
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            raw = {}
    correction = raw.get("sensor_correction") or {}
    protocol = raw.get("sensor_protocol") or raw.get("protocol") or ""
    board = raw.get("board")
    if board is None:
        board = "m16b" if (protocol.startswith("M16B") or (rows, cols) == (16, 16) or not raw) else "unknown"
    if board == "m16b":
        meta = dict(M16B_DEFAULTS)
    else:
        meta = {"board": board, "protocol": protocol, "rows": rows, "cols": cols}
        meta.update({k: None for k in META_KEYS if k not in meta})
    if protocol:
        meta["protocol"] = protocol
    for key in META_KEYS:
        if key in raw:
            meta[key] = raw[key]
        elif key in correction:
            meta[key] = correction[key]
    if rows is not None and cols is not None:
        meta["rows"], meta["cols"] = rows, cols
    # Only a session.json that states the polarity settles it; the defaults are a guess.
    meta["polarity_known"] = "pressure_lowers_adc" in raw
    return meta


def frame_rate(ns: np.ndarray) -> float:
    """Frames per second from the host timestamps (median interval); FPS when there are too few frames."""
    dt = np.diff(np.asarray(ns, dtype=np.int64)) / 1e9
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if dt.size >= 2 else FPS


def baseline(frames: np.ndarray, t: np.ndarray, seconds: float = 5.0) -> np.ndarray:
    """Per-cell no-load level: the median of the first `seconds` (keep the sensor unloaded)."""
    return np.median(frames[t <= seconds], axis=0)


def response_sign(frames: np.ndarray, base: np.ndarray, meta: dict | None = None) -> float:
    """+1 when pressing raises the ADC count (16x16 M16B), -1 when it lowers it (4x4 ESP32).

    A session.json that states pressure_lowers_adc (every --record session
    since 2026-10-30, see session_meta) decides. Otherwise the data does:
    whichever excursion of the summed baseline-subtracted frame is larger
    (99.5th percentile against the 0.5th), which needs a real press in the run.
    """
    if meta and meta.get("polarity_known"):
        return -1.0 if meta["pressure_lowers_adc"] else 1.0
    sums = (np.asarray(frames, dtype=float) - base).sum(axis=(1, 2))
    return 1.0 if np.percentile(sums, 99.5) >= -np.percentile(sums, 0.5) else -1.0


def clean(frames: np.ndarray, base: np.ndarray, deadband: float = DEADBAND,
          sign: float | None = None) -> np.ndarray:
    """Baseline-subtracted counts, pressure positive, with |change| <= deadband set to 0.

    `sign` is +1 / -1 (see response_sign); None detects it from a stack and
    assumes +1 for a single frame. Works on one frame or a stack.
    """
    frames = np.asarray(frames, dtype=float)
    if sign is None:
        sign = response_sign(frames, base) if frames.ndim == 3 and len(frames) > 1 else 1.0
    delta = sign * (frames - base)
    return np.where(np.abs(delta) > deadband, delta, 0.0)


def frame_features(delta: np.ndarray) -> dict[str, np.ndarray]:
    """Current-frame features of a cleaned stack [N,R,C] (pressure positive), around each frame's peak cell."""
    n = len(delta)
    cols = delta.shape[2]
    flat = delta.reshape(n, -1)
    k = flat.argmax(axis=1)
    pr, pc = np.divmod(k, cols)
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
    how far below that maximum the cell now is. `fps` is the session's real
    rate (frame_rate), so the lags are the same seconds on every board.
    """
    n = len(peak)
    out: dict[str, np.ndarray] = {}
    for s in LAGS_S:
        lag = max(1, int(round(s * fps)))
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
