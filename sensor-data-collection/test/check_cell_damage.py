#!/usr/bin/env python3
"""Were cells damaged by overload? Baseline history plus a suspect-vs-fresh probe.

    python check_cell_damage.py history C:/tactile_data/force_cal
    python check_cell_damage.py probe <session> [<session> ...] --suspect R5C5,R2C7

On the 16x16 board a 6x6 mm indenter puts the whole load on one 5x5 mm pad:
20 N is ~800 kPa, while Velostat has been reported to take damage above
~120 kPa (3 N there). On the 4x4 ESP32 board the pads are 8x8 mm, so the
indenter face (36 mm^2) sets the pressure: 120 kPa is ~4.3 N. An overloaded
pad shows up in three ways, and this checks all three:

1. history: the unloaded level of a pressed cell moves -- the gap has
   collapsed and the cell no longer returns to its old zero. Every capture
   under ROOT is scanned in time order; a cell whose no-load level moves by
   --shift counts from its first recording is flagged, and the cells that
   have been pressed hard (peak change >= --hard counts) are listed.
2. probe, sensitivity: the same low-force program (e.g. 1 / 2 / 4 N holds,
   never above 5 N) is run on suspect and on never-pressed cells, one
   capture per cell. Cells differ by up to 2x when new, so a suspect only
   counts as changed when it falls outside the fresh cells' range by
   --margin.
3. probe, recovery: after each release the cell should be back within 2
   counts of its pre-press level within a few seconds; a residual offset or
   a slow recovery means the film or spacer is not springing back.

With force_alignment/frames.csv in a probe session the plateaus are labelled
with their Mark-10 load and compared as counts per newton; without it they
are compared in order, which assumes the same program on every cell.

Works for any grid: the size, the frame rate and the press direction (the
16x16 board's ADC rises under load, the 4x4 board's falls) come from each
session's data, so a root can hold sessions of both boards.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python files"))
from tactile_preprocess import baseline, clean, frame_rate, load_raw, response_sign, session_meta  # noqa: E402


def cell_name(rc) -> str:
    return f"R{rc[0] + 1}C{rc[1] + 1}"


def parse_cell(text: str):
    r, c = text.upper().lstrip("R").split("C")
    return int(r) - 1, int(c) - 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("history", help="baseline shift of every cell across all captures")
    h.add_argument("root", type=Path)
    h.add_argument("--shift", type=float, default=2.0, help="counts of no-load change that count as a shift")
    h.add_argument("--hard", type=float, default=150.0, help="peak change that counts as a hard press")
    q = sub.add_parser("probe", help="compare suspect cells with fresh ones under the same program")
    q.add_argument("sessions", type=Path, nargs="+")
    q.add_argument("--suspect", required=True, help="comma-separated cells, e.g. R5C5,R5C7,R2C7")
    q.add_argument("--margin", type=float, default=0.25, help="fraction outside the fresh range to flag")
    q.add_argument("--min-hold-s", type=float, default=4.0)
    q.add_argument("--at", default="1,2,4", help="N; reference forces compared when force labels exist")
    return p.parse_args()


def load(session: Path):
    """(t, frames, frame index, fps) of a session; the grid is whatever the CSV holds."""
    ns, idx, frames = load_raw(session)
    t = (ns - ns[0]) / 1e9
    return t, frames, idx, frame_rate(ns)


# ------------------------------------------------------------------ history


def history(args) -> None:
    sessions = sorted(d for d in args.root.glob("capture_*") if (d / "sensor_raw.csv").exists())
    first: dict[tuple, float] = {}
    shifts, hard = [], {}
    print(f"{'capture':16s} {'s':>5s}  pressed  peak  no-load change of hard-pressed cells")
    for d in sessions:
        t, frames, _, _ = load(d)
        if t[-1] < 10:
            continue
        start, end = baseline(frames, t), np.median(frames[t >= t[-1] - 5], axis=0)
        sign = response_sign(frames, start, session_meta(d, *frames.shape[1:]))  # pressure positive on either board
        peak = (sign * (frames - start)).max(axis=0)
        rc = np.unravel_index(peak.argmax(), peak.shape)
        if sign * (start[rc] - np.median(start)) > 20:  # recording started with the cell already pressed
            print(f"{d.name[8:]:16s} {t[-1]:5.0f}  started under load on {cell_name(rc)}; skipped")
            continue
        for r, c in zip(*np.nonzero(peak >= args.hard)):
            hard.setdefault((int(r), int(c)), []).append(d.name[8:])
        for r, c in np.ndindex(start.shape):
            first.setdefault((r, c), start[r, c])
        # Cells of a different grid (the other board) have no history here; skip them.
        here = [k for k in hard if k[0] < start.shape[0] and k[1] < start.shape[1]]
        moved = [(k, start[k] - first[k]) for k in here if abs(start[k] - first[k]) >= args.shift]
        moved += [(k, end[k] - start[k]) for k in here if abs(end[k] - start[k]) >= args.shift]
        shifts += [(d.name[8:], cell_name(k), v) for k, v in moved]
        note = "  ".join(f"{cell_name(k)} {start[k] - first[k]:+.0f}" for k in here) or "-"
        print(f"{d.name[8:]:16s} {t[-1]:5.0f}  {cell_name(rc):7s} {peak.max():4.0f}  {note}")
    print("\nhard-pressed cells (peak change >= %.0f counts):" % args.hard)
    for k, names in hard.items():
        print(f"  {cell_name(k)}: {len(names)} captures, first {names[0]}")
    if shifts:
        print("\nNO-LOAD SHIFTS (possible collapsed gap):")
        for name, cell, v in shifts:
            print(f"  {name} {cell}: {v:+.0f} counts")
    else:
        print(f"\nno hard-pressed cell moved its no-load level by >= {args.shift:.0f} counts "
              "in any capture: no sign of a collapsed gap")


# ------------------------------------------------------------------ probe


def force_labels(session: Path, idx: np.ndarray):
    path = session / "force_alignment" / "frames.csv"
    if not path.exists():
        return None
    row_of = {f: i for i, f in enumerate(idx)}
    load_n = np.full(len(idx), np.nan)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["load_N"] not in ("", "nan"):
                load_n[row_of[int(row["frame_index"])]] = float(row["load_N"])
    return load_n


def plateaus(p: np.ndarray, min_len: int, fps: float):
    """Stable stretches of a cleaned cell series: (start, end) frame ranges."""
    win = max(3, int(fps * 1.5))
    std = np.array([p[max(0, i - win):i + 1].std() for i in range(len(p))])
    stable = (p > 0) & (std < 2.0)
    out, start = [], None
    for i, s in enumerate(np.append(stable, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def probe_session(session: Path, min_hold_s: float):
    t, frames, idx, fps = load(session)
    base = baseline(frames, t)
    sign = response_sign(frames, base, session_meta(session, *frames.shape[1:]))
    delta = clean(frames, base, sign=sign)
    rc = np.unravel_index(delta.max(axis=0).argmax(), delta.shape[1:])
    raw = sign * (frames[:, rc[0], rc[1]] - base[rc])  # pressure positive, no deadband
    p = delta[:, rc[0], rc[1]]
    load_n = force_labels(session, idx)
    holds = []
    for a, b in plateaus(p, int(min_hold_s * fps), fps):
        level = float(np.median(p[a:b]))
        force = float(np.nanmedian(load_n[a:b])) if load_n is not None and np.isfinite(load_n[a:b]).any() else None
        holds.append({"counts": level, "force_N": force})
    # A press is the indenter on the pad: load > 0.1 N when labelled, else the raw change > 15 counts
    # (a real contact is > 30). Judged on the raw change, so a residual inside the deadband still shows.
    pressed = load_n > 0.1 if load_n is not None else raw > 15
    edges = np.diff(pressed.astype(int))
    on, off = np.flatnonzero(edges == 1) + 1, np.flatnonzero(edges == -1) + 1
    recoveries = []
    for k in off:
        before = on[on < k]
        nxt = on[on > k]
        if before.size == 0:
            continue
        pre = raw[max(0, before[-1] - int(3 * fps)):before[-1]]
        stop = nxt[0] if nxt.size else len(raw)
        post = raw[k + int(3 * fps):min(stop, k + int(8 * fps))]
        if pre.size < 3 or post.size < 3:
            continue
        back = np.flatnonzero(np.abs(raw[k:stop] - np.median(pre)) <= 2)
        recoveries.append({"residual": float(np.median(post) - np.median(pre)),
                           "recover_s": float(back[0] / fps) if back.size else None})
    return {"session": session.name, "cell": rc, "holds": holds, "recoveries": recoveries}


def probe(args) -> None:
    suspect = {parse_cell(s) for s in args.suspect.split(",") if s.strip()}
    results = [probe_session(s, args.min_hold_s) for s in args.sessions]
    by_cell: dict[tuple, list] = {}
    for r in results:
        by_cell.setdefault(r["cell"], []).append(r)
    labelled = all(h["force_N"] is not None for r in results for h in r["holds"])

    refs = [float(x) for x in args.at.split(",")] if labelled else None

    def response(rs):
        """Counts at each reference force (labelled, interpolated, never extrapolated), else plateaus in order."""
        holds = [h for r in rs for h in r["holds"]]
        if labelled:
            f = np.array([h["force_N"] for h in holds])
            c = np.array([h["counts"] for h in holds])
            order = np.argsort(f)
            f, c = f[order], c[order]
            return [float(np.interp(x, f, c)) if f.size >= 2 and f[0] <= x <= f[-1] else None for x in refs]
        return [h["counts"] for h in holds]

    names = [f"@{x:g}N" for x in refs] if labelled else None
    print(f"{'cell':6s} {'kind':8s} {'n':>3s}  " + ("counts " + " ".join(f"{n:>6s}" for n in names) if labelled
                                                  else "plateau counts in order") + "   release residual/recovery")
    table = {}
    for cell, rs in sorted(by_cell.items()):
        kind = "suspect" if cell in suspect else "fresh"
        resp = response(rs)
        table[cell] = (kind, resp, rs)
        shown = " ".join(f"{v:6.0f}" if v is not None else f"{'-':>6s}" for v in resp)
        rec = ", ".join(f"{x['residual']:+.0f}/{x['recover_s']:.1f}s" if x["recover_s"] is not None
                        else f"{x['residual']:+.0f}/never" for r in rs for x in r["recoveries"])
        print(f"{cell_name(cell):6s} {kind:8s} {len(rs):3d}  {'       ' if labelled else ''}{shown}   {rec or '-'}")

    fresh = [v[1] for v in table.values() if v[0] == "fresh"]
    print()
    if len(fresh) < 3:
        print(f"only {len(fresh)} fresh cells: probe at least 3 never-pressed cells to have a reference range")
    width = len(refs) if labelled else min((len(v) for v in fresh), default=0)
    ranges = []
    for j in range(width):
        vals = [v[j] for v in fresh if j < len(v) and v[j] is not None]
        ranges.append((min(vals), max(vals)) if vals else None)
    label = names if labelled else [f"plateau {j + 1}" for j in range(width)]
    print("fresh cells: " + ", ".join(f"{label[j]} {r[0]:.0f}-{r[1]:.0f}" for j, r in enumerate(ranges) if r))
    for cell, (kind, resp, rs) in table.items():
        if kind != "suspect":
            continue
        flags = []
        for j, r in enumerate(ranges):
            v = resp[j] if j < len(resp) else None
            if r and v is not None and (v > r[1] * (1 + args.margin) or v < r[0] * (1 - args.margin)):
                flags.append(f"{label[j]} {v:.0f} outside fresh {r[0]:.0f}-{r[1]:.0f} by > {args.margin:.0%}")
        bad = [x for r in rs for x in r["recoveries"]
               if abs(x["residual"]) > 2 or x["recover_s"] is None or x["recover_s"] > 5]
        if bad:
            flags.append(f"{len(bad)} release(s) with a residual > 2 counts or recovery > 5 s")
        if not flags:
            flags.append("releases recover; sensitivity within the fresh cells' range" if any(ranges)
                         else "releases recover; no fresh reference, so sensitivity is not judged")
        print(f"  {cell_name(cell)}: " + "; ".join(flags))


def main() -> None:
    args = parse_args()
    history(args) if args.cmd == "history" else probe(args)


if __name__ == "__main__":
    main()
