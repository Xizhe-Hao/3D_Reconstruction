#!/usr/bin/env python3
"""Dead-weight calibration, pad by pad: guided recording plus per-frame force labels.

    python weight_cal.py --board esp32 --record C:/tactile_data/weight_cal --pads all
    python weight_cal.py --board esp32 --record C:/tactile_data/weight_cal --pads R2C2,R3C3 --reps 6
    python weight_cal.py --simulate --pads all                 # dry run without hardware
    python weight_cal.py --relabel <session> --last-s 5         # rebuild frames.csv from weights/events.csv
    python weight_cal.py --selftest                             # headless end-to-end check

One run walks through every pad you list. For each pad the window outlines
the target cell on the map, you press SPACE to start it (a new
capture_<stamp>_RxCy session folder opens and the first --baseline-s seconds
must stay unloaded), then it tells you which weight to put on ("PUT ON 100 g
on R2C2"), you centre it and press SPACE, the hold countdown runs, you take it
off and press SPACE again, and an unloaded rest follows. While a weight is on,
the status line shows where the response actually is; it turns red when the
peak is on a different cell than the target or when the target carries less
than 70 % of the response. When the pad's steps are done its session is
closed, frames.csv is written, a one-line summary (counts per weight) is
printed so a dead or weak pad shows at once, and the next pad is announced.

Every SPACE is logged on the host clock that also stamps the frames, so there
is nothing to align. Weights (10 / 20 / 50 / 100 / 200 g) sit directly on the
pad, one at a time, each once per repetition, in ascending, descending and
shuffled order across repetitions (--order mixed). --stack uses every
combination when weights can be piled; --tare-step adds a 0 g step.

Per pad session, next to sensor_raw.csv and session.json:
    weights/events.csv              one row per SPACE: host_received_ns, step, mass_g, action
    force_alignment/frames.csv      per-frame load_N / press / phase in the layout
                                    align_force_presses.py writes, so bench_quicklook.py,
                                    force_model_eval.py and check_cell_damage.py read it
Then: python weight_report.py <session>        and      python weight_summary.py <root>

Keys: SPACE start pad / placed / removed    U undo the last mark    N skip this step
      P skip the rest of this pad           B redo the display baseline    Q quit (labels are written)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Python files"))
from live_sensor_2d import BOARDS, Esp32Source, SerialSource, SessionRecorder  # noqa: E402
from tactile_preprocess import baseline, clean, load_raw, response_sign, session_meta  # noqa: E402

G = 9.80665
WEIGHT_SET_G = (10.0, 20.0, 50.0, 100.0, 200.0)
SPREAD_COUNTS = 3.0   # a cell changed by more than this (8-bit counts) is part of the weight's footprint
PLATEAU_LAST_S = 5.0  # the plateau is the last seconds of the hold (the response creeps for several seconds)


# ---------------------------------------------------------------- plan


def combos(weights, stack=False, tare_step=False):
    """The masses to apply: each piece on its own (default), or every distinct stack with --stack."""
    out = {}
    if tare_step:
        out[0.0] = (0.0,)
    for n in range(1, (len(weights) if stack else 1) + 1):
        for pick in itertools.combinations(weights, n):
            out.setdefault(round(sum(pick), 3), tuple(sorted(pick, reverse=True)))
    return sorted(out.items())


def plan_steps(masses, reps, order, seed):
    """[(step, mass_g, pieces, rep, order)]: one pass over the masses per repetition."""
    rng = random.Random(seed)
    steps = []
    for rep in range(reps):
        kind = order if order != "mixed" else ("ascending", "descending", "shuffled")[rep % 3]
        seq = list(masses)
        if kind == "descending":
            seq.reverse()
        elif kind == "shuffled":
            rng.shuffle(seq)
        for mass, pieces in seq:
            steps.append((len(steps) + 1, mass, pieces, rep + 1, kind))
    return steps


def pieces_text(pieces):
    if pieces == (0.0,):
        return "nothing (tare step)"
    return " + ".join(f"{p:g}" for p in pieces) if len(pieces) > 1 else f"{pieces[0]:g}"


def parse_pad(text):
    m = re.fullmatch(r"\s*[Rr](\d+)[Cc](\d+)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"pad '{text}' is not RxCy")
    return int(m.group(1)) - 1, int(m.group(2)) - 1


def pad_name(rc):
    return f"R{rc[0] + 1}C{rc[1] + 1}"


def pad_list(spec, shape, shuffle=False, seed=0):
    if spec.strip().lower() == "all":
        pads = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    else:
        pads = [parse_pad(p) for p in spec.split(",") if p.strip()]
    for r, c in pads:
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            raise SystemExit(f"pad {pad_name((r, c))} is outside the {shape[0]}x{shape[1]} array")
    if shuffle:
        random.Random(seed).shuffle(pads)
    return pads


# ---------------------------------------------------------------- events and labels


class EventLog:
    """weights/events.csv: rewritten after every change so undo stays simple."""

    FIELDS = ["host_received_ns", "t_s", "step", "mass_g", "pieces", "action", "rep", "order"]

    def __init__(self, session: Path, t0_ns: int):
        self.path = Path(session) / "weights" / "events.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0_ns = t0_ns
        self.rows: list[dict] = []
        self.write()

    def mark(self, step, action, host_ns):
        n, mass, pieces, rep, kind = step
        self.rows.append({
            "host_received_ns": host_ns, "t_s": f"{(host_ns - self.t0_ns) / 1e9:.3f}", "step": n,
            "mass_g": f"{mass:g}", "pieces": pieces_text(pieces), "action": action, "rep": rep, "order": kind,
        })
        self.write()

    def undo(self):
        if self.rows:
            self.rows.pop()
            self.write()

    def write(self):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)


def read_events(session: Path):
    path = Path(session) / "weights" / "events.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r["host_received_ns"] = int(r["host_received_ns"])
        r["step"] = int(r["step"])
        r["mass_g"] = float(r["mass_g"])
    return rows


def intervals(events):
    """[(step, mass_g, on_ns, off_ns)] from the on/off marks; an 'on' without 'off' ends at the recording end."""
    out, open_ = [], {}
    for r in events:
        if r["action"] == "on":
            open_[r["step"]] = r
        elif r["action"] == "off" and r["step"] in open_:
            on = open_.pop(r["step"])
            out.append((r["step"], r["mass_g"], on["host_received_ns"], r["host_received_ns"]))
    for step, on in open_.items():
        out.append((step, on["mass_g"], on["host_received_ns"], None))
    return sorted(out)


EDGE_SLACK_S = 4.0     # the SPACE may come this long before or after the weight actually moved
EDGE_MIN_COUNTS = 4.0  # below this peak the response is not visible and the marks are used as they are


def refine_edges(t, x, t_on, t_off, slack_s=EDGE_SLACK_S, min_counts=EDGE_MIN_COUNTS):
    """When the weight really was on: (a, b) from the signal, the marks only bound the search.

    People take the weight off and then press SPACE (or press first and then
    reach in), so the marks can be a few seconds off. Inside [t_on - slack,
    t_off + slack] the loaded stretch is where the response stays above half
    of its plateau; the longest such run is the step. Without a visible
    response (dead pad, 10 g) the marks are returned unchanged.
    """
    w = np.flatnonzero((t >= t_on - slack_s) & (t <= t_off + slack_s))
    if w.size < 5:
        return t_on, t_off
    x = np.asarray(x)[w]
    peak = float(np.percentile(x, 95))
    if peak < min_counts:
        return t_on, t_off
    on = x >= 0.5 * peak
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2])]
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return float(t[w[a]]), float(t[w[b - 1]])


def write_labels(session: Path, platen_g: float, settle_s: float, release_s: float, deadband: float,
                 last_s: float = PLATEAU_LAST_S):
    """force_alignment/frames.csv in the align_force_presses.py layout, from the weight marks.

    The marks bound each step; the on/off edges come from the loaded cell's
    own response (refine_edges), so a SPACE a few seconds early or late does
    not move the labels. load_N = (mass + platen) * g while the weight is on,
    0 otherwise; press = step number; phase = loading from the edge until
    last_s before the end (the response is still creeping), hold for the last
    last_s of the step (at least settle_s after the edge), unloading for
    release_s after the weight came off, idle elsewhere. distance and
    indentation stay blank; force_time_s equals t_s (one clock).
    """
    session = Path(session)
    ns, idx, frames = load_raw(session)
    t = (ns - ns[0]) / 1e9
    base = baseline(frames, t)
    meta = session_meta(session, *frames.shape[1:])
    sign = response_sign(frames, base, meta)
    delta = sign * (frames - base)
    total = clean(frames, base, deadband, sign=sign).sum(axis=(1, 2))
    raw_meta = json.loads((session / "session.json").read_text(encoding="utf-8")) if (session / "session.json").exists() else {}
    m = re.fullmatch(r"R(\d+)C(\d+)", raw_meta.get("pad_under_test") or "")
    if m and int(m.group(1)) - 1 < delta.shape[1] and int(m.group(2)) - 1 < delta.shape[2]:
        cell = delta[:, int(m.group(1)) - 1, int(m.group(2)) - 1]
    else:
        cell = delta.max(axis=(1, 2))
    load = np.zeros(len(t))
    press = np.zeros(len(t), dtype=int)
    phase = np.full(len(t), "idle", dtype=object)
    for step, mass, on_ns, off_ns in intervals(read_events(session)):
        t_on = (on_ns - ns[0]) / 1e9
        t_off = (off_ns - ns[0]) / 1e9 if off_ns is not None else t[-1] + 1
        t_on, t_off = refine_edges(t, cell, t_on, t_off)
        on = (t >= t_on) & (t <= t_off)
        load[on] = (mass + platen_g) * G / 1000.0
        press[on] = step
        hold_from = max(t_on + settle_s, t_off - last_s)
        phase[on & (t < hold_from)] = "loading"
        phase[on & (t >= hold_from)] = "hold"
        rel = (t > t_off) & (t < t_off + release_s)
        phase[rel] = "unloading"
        press[rel] = step
    out = session / "force_alignment"
    out.mkdir(exist_ok=True)
    with (out / "frames.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["host_received_ns", "frame_index", "t_s", "force_time_s", "load_N", "distance_mm",
                    "indentation_mm", "press", "phase", "tactile_response"])
        for k in range(len(t)):
            w.writerow([ns[k], idx[k], f"{t[k]:.4f}", f"{t[k]:.4f}", f"{load[k]:.4f}", "", "",
                        press[k], phase[k], f"{total[k]:.2f}"])
    (out / "alignment.json").write_text(json.dumps({
        "session": str(session), "source": "weight_cal.py (dead weights, host clock, no alignment needed)",
        "platen_g": platen_g, "settle_s": settle_s, "release_s": release_s, "last_s": last_s,
        "deadband": deadband, "steps_labelled": int(len(set(press[press > 0]))), "frames": int(len(t)),
        "frames_loaded": int((load > 0).sum()),
    }, indent=2))
    print(f"labels: {int((load > 0).sum())} loaded frames in {len(set(press[press > 0]))} steps -> "
          f"{out / 'frames.csv'}", flush=True)


# ---------------------------------------------------------------- simulated board


class WeightSimSource:
    """A 4x4 ESP32-like stream whose current pad responds to the mass the controller says is on it."""

    def __init__(self, shape=(4, 4), fps=10.0, bits=12, paced=True):
        self.shape, self.bits = tuple(shape), bits
        self.adc_max = float(2 ** bits - 1)
        self.board = "esp32"
        self.scale = 2.0 ** (bits - 8)
        self.period = 1.0 / fps
        self.paced = paced
        self.pad = (1, 1)
        self.mass_g = 0.0
        self.on_since = None
        self.rng = np.random.default_rng(1)
        self.base = (136.0 + self.rng.normal(0, 4, self.shape)) * self.scale
        self.gain = 1.0 + 0.3 * self.rng.standard_normal(self.shape)  # pads differ in sensitivity
        self.gain[self.shape[0] - 1, self.shape[1] - 1] = 0.05        # one nearly dead pad
        self.frame_index = 0
        self.t0 = time.perf_counter()
        self.last = 0.0
        self.clock = 0.0
        self.on_frame = None

    def now(self):
        return time.perf_counter() - self.t0 if self.paced else self.clock

    def now_ns(self):
        return time.time_ns() if self.paced else int(1.7e18 + self.now() * 1e9)

    def set_pad(self, pad):
        self.pad = tuple(pad)

    def set_mass(self, mass_g):
        self.mass_g = mass_g
        self.on_since = self.now() if mass_g > 0 else None

    def read_latest(self):
        now = self.now()
        if self.paced and now - self.last < self.period:
            return None
        if not self.paced:
            self.clock += self.period
            now = self.clock
        self.last = now
        f = self.mass_g * G / 1000.0
        resp = 0.0 if f <= 0 else min(95.0, 22.0 * (f / 1.0) ** 0.7) * self.gain[self.pad]
        if self.on_since is not None:
            resp *= 1.0 - 0.25 * np.exp(-(now - self.on_since) / 3.0)  # creeps up over a few seconds
        grid = self.base.copy()
        r, c = self.pad
        grid[r, c] -= resp * self.scale
        spill = 0.03 if self.mass_g <= 20 else (0.10 if self.mass_g <= 100 else 0.30)
        if r + 1 < self.shape[0]:
            grid[r + 1, c] -= spill * resp * self.scale
        grid += self.rng.normal(0, 0.3 * self.scale, self.shape)
        grid = np.clip(np.rint(grid), 0, self.adc_max).astype(np.dtype("<u2") if self.bits > 8 else np.uint8)
        self.frame_index += 1
        host_ns = self.now_ns()
        if self.on_frame is not None:
            self.on_frame(host_ns, self.frame_index, int(now * 1000), grid)
        return self.frame_index, int(now * 1000), grid

    def meta(self):
        return {**BOARDS["esp32"], "adc_bits": self.bits, "simulated": True}

    def start(self):
        pass

    def close(self):
        print("Simulated weights stopped.")


# ---------------------------------------------------------------- controller


class Controller:
    """The pad / step machine behind the keys; independent of the window so it runs headlessly too.

    states: pad_start -> baseline -> (place -> hold -> rest)* -> pad_done -> pad_start ... -> all_done
    """

    DEBOUNCE_S = 1.0  # a second SPACE within this is a key bounce, not a weight coming off again

    def __init__(self, pads, masses, args, source, clock_ns=None):
        self.pads = pads
        self.blocks = [plan_steps(masses, args.reps, args.order, args.seed + k) for k in range(len(pads))]
        self.args = args
        self.source = source
        self.now_ns = clock_ns or time.time_ns
        self.b, self.i = 0, 0
        self.state = "pad_start"
        self.since = time.time()
        self.recorder = None
        self.log = None
        self.levels = {}      # (pad index, step) -> plateau level of the target cell
        self.buffer = []      # (t, target-cell delta) while a weight is on
        self.done_steps = 0
        self.step_times = []  # seconds per completed step, for the ETA
        self.sessions = []
        self.started = time.time()

    # ---- what is where
    @property
    def pad(self):
        return self.pads[self.b] if self.b < len(self.pads) else None

    @property
    def steps(self):
        return self.blocks[self.b] if self.b < len(self.blocks) else []

    @property
    def step(self):
        return self.steps[self.i] if self.i < len(self.steps) else None

    @property
    def total_steps(self):
        return sum(len(b) for b in self.blocks)

    # ---- pad sessions
    def start_pad(self):
        pad = self.pad
        meta = {**(self.source.meta() if hasattr(self.source, "meta") else BOARDS[self.args.board]),
                "pad_under_test": pad_name(pad), "platen_g": self.args.platen_g,
                "weights_g": [float(w) for w in self.args.weights.split(",")], "reps": self.args.reps,
                "order": self.args.order, "hold_s": self.args.hold_s, "rest_s": self.args.rest_s,
                "plateau_last_s": self.args.last_s}
        self.recorder = SessionRecorder(self.args.record, self.source.shape, meta,
                                        tool=f"weight_cal.py pad {pad_name(pad)}", suffix=f"_{pad_name(pad)}")
        self.source.on_frame = self.recorder
        self.log = EventLog(self.recorder.session, self.now_ns())
        if hasattr(self.source, "set_pad"):
            self.source.set_pad(pad)
        self.i = 0
        self.state, self.since = "baseline", time.time()

    def finish_pad(self):
        if self.recorder is None:
            return
        self.source.on_frame = None
        time.sleep(0.05)
        self.recorder.close()
        session = self.recorder.session
        self.recorder = None
        if self.log.rows:
            write_labels(session, self.args.platen_g, self.args.settle_s, self.args.release_s,
                         self.args.deadband, self.args.last_s)
            self.sessions.append(session)
            print(self.pad_summary(), flush=True)
        else:
            print(f"pad {pad_name(self.pad)}: no marks logged, session left without labels", flush=True)

    def pad_summary(self):
        by_mass = {}
        for (b, step), level in self.levels.items():
            if b == self.b:
                mass = next(s[1] for s in self.blocks[b] if s[0] == step)
                by_mass.setdefault(mass, []).append(level)
        parts = []
        for mass in sorted(by_mass):
            v = np.array(by_mass[mass])
            parts.append(f"{mass:g} g {v.mean():5.1f}" + (f"+-{v.std(ddof=1):.1f}" if len(v) > 1 else ""))
        weak = all(np.mean(v) < 3 * 2 for v in by_mass.values()) if by_mass else False
        return (f"pad {pad_name(self.pad)} done: " + "   ".join(parts) + " counts"
                + ("   <- WEAK / DEAD PAD?" if weak else ""))

    # ---- keys
    def space(self, host_ns=None):
        if host_ns is None and self.state in ("hold", "rest") and time.time() - self.since < self.DEBOUNCE_S:
            return
        host_ns = self.now_ns() if host_ns is None else host_ns
        if self.state == "pad_start":
            self.start_pad()
        elif self.state == "baseline":
            pass  # keep the pad empty; the analysis takes its baseline from these seconds
        elif self.state == "place":
            self.log.mark(self.step, "on", host_ns)
            self.buffer = []
            if hasattr(self.source, "set_mass"):
                self.source.set_mass(self.step[1])
            self.state, self.since = "hold", time.time()
        elif self.state == "hold":
            self.log.mark(self.step, "off", host_ns)
            self.levels[(self.b, self.step[0])] = self.plateau_of(self.buffer)
            if hasattr(self.source, "set_mass"):
                self.source.set_mass(0.0)
            self.state, self.since = "rest", time.time()
        elif self.state == "rest":
            self._advance()
        elif self.state == "pad_done":
            self.b += 1
            self.state = "pad_start" if self.b < len(self.pads) else "all_done"
            if self.state == "pad_start":
                self.start_pad()

    def tick(self):
        if self.state == "baseline" and time.time() - self.since >= self.args.baseline_s:
            self.state, self.since = "place", time.time()
        elif self.state == "rest" and time.time() - self.since >= self.args.rest_s:
            self._advance()

    def note(self, value):
        """Called with the target cell's change on every frame; used for the hold plateau."""
        if self.state in ("hold", "rest"):
            self.buffer.append((time.time(), value))

    def plateau_of(self, buffer):
        """Median of the last --last-s seconds the weight was visibly on (not of the last seconds before SPACE)."""
        if not buffer:
            return float("nan")
        tt = np.array([b[0] for b in buffer])
        v = np.array([b[1] for b in buffer])
        peak = float(np.percentile(v, 95))
        if peak >= EDGE_MIN_COUNTS:
            on = np.flatnonzero(v >= 0.5 * peak)
            end = tt[on[-1]]
            v = v[(tt >= end - self.args.last_s) & (tt <= end)]
        else:
            v = v[tt >= tt[-1] - self.args.last_s]
        return float(np.median(v)) if v.size else float("nan")

    def _advance(self):
        self.done_steps += 1
        self.step_times.append(time.time() - self.since)
        self.i += 1
        self.state, self.since = "place", time.time()
        if self.step is None:
            self.finish_pad()
            self.state = "pad_done"
            if self.b + 1 >= len(self.pads):
                self.state = "all_done"

    def undo(self):
        if self.state == "hold":
            self.log.undo()
            if hasattr(self.source, "set_mass"):
                self.source.set_mass(0.0)
            self.state = "place"
        elif self.state == "rest":
            self.log.undo()
            self.levels.pop((self.b, self.step[0]), None)
            if hasattr(self.source, "set_mass"):
                self.source.set_mass(self.step[1])
            self.state, self.since = "hold", time.time()
        elif self.state == "place" and self.i > 0:
            self.i -= 1
            self.state, self.since = "rest", time.time()

    def skip_step(self):
        if self.state == "hold":
            self.space(self.now_ns())
        if self.state in ("place", "rest"):
            self._advance()

    def skip_pad(self):
        if self.state == "hold":
            self.space(self.now_ns())
        if self.state in ("place", "rest", "baseline"):
            self.finish_pad()
            self.state = "pad_done" if self.b + 1 < len(self.pads) else "all_done"

    def close(self):
        if self.recorder is not None:
            self.finish_pad()

    # ---- text
    def progress(self):
        done = self.done_steps
        total = self.total_steps
        per = np.mean(self.step_times[-10:]) if self.step_times else (self.args.hold_s + self.args.rest_s + 6)
        left = (total - done) * per + (len(self.pads) - self.b - 1) * (self.args.baseline_s + 20)
        elapsed = time.time() - self.started
        return (f"pad {min(self.b + 1, len(self.pads))}/{len(self.pads)}   step {done}/{total}   "
                f"elapsed {elapsed / 60:.0f} min   about {max(left, 0) / 60:.0f} min left")

    def prompt(self):
        pad = pad_name(self.pad) if self.pad else ""
        if self.state == "all_done":
            return "ALL PADS DONE - press Q", "#2a9d3a"
        if self.state == "pad_start":
            return f"NEXT PAD: {pad}\nmove the weights there, keep it EMPTY,\nthen press SPACE to start", "#1f5fbf"
        if self.state == "baseline":
            left = self.args.baseline_s - (time.time() - self.since)
            return f"{pad}: no-load baseline  {max(left, 0):3.1f} s\nkeep the pad empty", "#b8860b"
        if self.state == "pad_done":
            return f"{pad} finished\npress SPACE for the next pad", "#2a9d3a"
        n, mass, pieces, rep, kind = self.step
        head = f"{pad}  step {n}/{len(self.steps)}  (rep {rep}, {kind})"
        if self.state == "place":
            return f"{head}\nPUT ON  {mass:g} g  =  {pieces_text(pieces)}\ncentre it on {pad}, then SPACE", "#1f5fbf"
        if self.state == "hold":
            held = time.time() - self.since
            left = self.args.hold_s - held
            tail = f"hold {left:4.1f} s more" if left > 0 else "TAKE OFF the weight, then SPACE"
            return f"{head}\n{mass:g} g on for {held:4.1f} s\n{tail}", "#1f5fbf" if left > 0 else "#2a9d3a"
        left = self.args.rest_s - (time.time() - self.since)
        return f"{head}: done\nrest {max(left, 0):4.1f} s (SPACE to go on)", "#777777"


# ---------------------------------------------------------------- window


class WeightWindow:
    def __init__(self, args, source, controller: Controller):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        self.plt, self.args, self.source, self.ctl = plt, args, source, controller
        self.rows, self.cols = source.shape
        self.scale = source.adc_max / 255.0
        self.lowers = BOARDS[source.board]["pressure_lowers_adc"]
        self.base_samples, self.base = [], None
        self.base_t0 = None
        self.trend_t, self.trend_v = [], []
        self.t0 = time.perf_counter()
        self.closed = False
        self.shown_pad = None

        self.fig = plt.figure(figsize=(13, 6.6))
        try:
            self.fig.canvas.manager.set_window_title("Dead-weight calibration")
        except AttributeError:
            pass
        gs = self.fig.add_gridspec(2, 2, width_ratios=(1.0, 1.4), height_ratios=(1.1, 1.0),
                                   left=0.06, right=0.97, top=0.9, bottom=0.1, wspace=0.25, hspace=0.4)
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.img = self.ax_map.imshow(np.zeros(source.shape), cmap="magma", vmin=0, vmax=40,
                                      interpolation="nearest")
        self.ax_map.set_xticks(range(self.cols), [f"C{c + 1}" for c in range(self.cols)])
        self.ax_map.set_yticks(range(self.rows), [f"R{r + 1}" for r in range(self.rows)])
        self.fig.colorbar(self.img, ax=self.ax_map, shrink=0.8).set_label("change vs baseline (8-bit counts)")
        self.labels = [[self.ax_map.text(c, r, "", ha="center", va="center", fontsize=8.5, color="w",
                                         family="monospace")
                        for c in range(self.cols)] for r in range(self.rows)]
        self.target = Rectangle((-0.5, -0.5), 1, 1, fill=False, ec="#4de8ff", lw=3)
        self.ax_map.add_patch(self.target)
        self.map_title = self.ax_map.set_title("", fontsize=10)

        self.ax_prompt = self.fig.add_subplot(gs[0, 1])
        self.ax_prompt.axis("off")
        self.prompt = self.ax_prompt.text(0.5, 0.6, "", ha="center", va="center", fontsize=17,
                                          fontweight="bold", transform=self.ax_prompt.transAxes)
        self.sub = self.ax_prompt.text(0.5, 0.12, "", ha="center", va="bottom", fontsize=9.5, color="#555",
                                       transform=self.ax_prompt.transAxes)
        self.prog = self.fig.text(0.5, 0.945, "", ha="center", fontsize=10, color="#333")

        self.ax_trend = self.fig.add_subplot(gs[1, 1])
        self.line, = self.ax_trend.plot([], [], color="#c03028", lw=1.2)
        self.ax_trend.set(xlabel="time (s)", ylabel="target cell change (counts)", title="last 90 s")
        self.ax_trend.grid(alpha=0.3)
        self.fig.text(0.5, 0.02, "SPACE start pad / on / off   U undo   N skip step   P skip pad   "
                      "B display baseline   Q quit", ha="center", fontsize=9, color="#404040")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "closed", True))

    def on_key(self, event):
        k = event.key
        if k == " ":
            if self.ctl.state in ("place", "hold") and self.base is None:
                return
            was_start = self.ctl.state == "pad_start"
            self.ctl.space()
            if was_start:
                self.base, self.base_samples = None, []  # display baseline per pad, like the analysis one
        elif k == "u":
            self.ctl.undo()
        elif k == "n":
            self.ctl.skip_step()
        elif k == "p":
            self.ctl.skip_pad()
        elif k == "b":
            self.base, self.base_samples = None, []
        elif k in ("q", "escape"):
            self.plt.close(self.fig)

    def update(self, sample):
        _, _, grid = sample
        g = grid.astype(float) / self.scale
        now = time.perf_counter() - self.t0
        pad = self.ctl.pad
        if pad != self.shown_pad and pad is not None:
            self.shown_pad = pad
            self.target.set_xy((pad[1] - 0.5, pad[0] - 0.5))
            self.map_title.set_text(f"target pad {pad_name(pad)} (outlined)")
        self.prog.set_text(self.ctl.progress())
        if self.base is None:
            if self.base_t0 is None or not self.base_samples:
                self.base_t0 = now
            self.base_samples.append(g)
            if now - self.base_t0 >= min(self.args.baseline_s, 3.0) and len(self.base_samples) >= 5:
                self.base = np.median(np.stack(self.base_samples), axis=0)
            else:
                self.img.set_data(np.zeros_like(g))
                text, color = self.ctl.prompt()
                self.prompt.set_text(text if self.ctl.state != "place" else "collecting display baseline ...")
                self.prompt.set_color(color)
                return
        delta = (self.base - g) if self.lowers else (g - self.base)
        self.img.set_data(delta)
        for r in range(self.rows):
            for c in range(self.cols):
                self.labels[r][c].set_text(f"{delta[r, c]:.2f}")
        tv = float(delta[pad]) if pad is not None else float(delta.max())
        self.ctl.note(tv)
        self.trend_t.append(now)
        self.trend_v.append(tv)
        while self.trend_t and now - self.trend_t[0] > 90:
            self.trend_t.pop(0)
            self.trend_v.pop(0)
        self.line.set_data(self.trend_t, self.trend_v)
        self.ax_trend.set_xlim(max(0, now - 90), max(now, 10))
        lo, hi = min(self.trend_v), max(self.trend_v)
        self.ax_trend.set_ylim(min(lo, -2) - 2, max(hi, 10) * 1.1)

        text, color = self.ctl.prompt()
        self.prompt.set_text(text)
        self.prompt.set_color(color)
        r, c = np.unravel_index(int(delta.argmax()), delta.shape)
        foot = delta > SPREAD_COUNTS
        cells = int(foot.sum())
        warn = ""
        if cells:
            w = np.where(foot, delta, 0.0)
            rr, cc = np.indices(delta.shape)
            cy, cx = (w * rr).sum() / w.sum(), (w * cc).sum() / w.sum()
            where = f"peak R{r + 1}C{c + 1} {delta[r, c]:.2f}   centroid R{cy + 1:.2f} C{cx + 1:.2f}   {cells} cell(s)"
            if pad is not None and self.ctl.state == "hold":
                share = float(delta[pad] / w.sum())
                if (r, c) != tuple(pad):
                    warn = f"   PEAK ON R{r + 1}C{c + 1}, TARGET IS {pad_name(pad)}"
                elif share < 0.7:
                    warn = f"   target carries only {share:.0%}: RE-CENTRE"
        else:
            where = "no contact"
        self.sub.set_text(where + warn)
        self.sub.set_color("#b22222" if warn else "#555")

    def run(self):
        self.plt.ion()
        self.plt.show()
        try:
            while not self.closed and self.plt.fignum_exists(self.fig.number):
                self.ctl.tick()
                sample = self.source.read_latest()
                if sample is not None:
                    self.update(sample)
                    self.fig.canvas.draw_idle()
                self.plt.pause(0.02)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------- main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--board", choices=("esp32", "m16b"), default="esp32")
    p.add_argument("--port", default=None, help="COM port; default: found on USB (esp32)")
    p.add_argument("--record", type=Path, metavar="ROOT", help="capture root; one capture_<stamp>_RxCy per pad")
    p.add_argument("--simulate", action="store_true", help="no hardware: simulated pads that follow the marks")
    p.add_argument("--selftest", action="store_true", help="headless simulated run: records, labels, exits")
    p.add_argument("--relabel", type=Path, metavar="SESSION", help="only rebuild frames.csv from events.csv")
    p.add_argument("--pads", default=None, help="'all' (row by row) or a list like R2C2,R3C3,R1C4")
    p.add_argument("--pad", default=None, help="a single pad (same as --pads with one entry)")
    p.add_argument("--shuffle-pads", action="store_true", help="randomise the pad order (decouples drift)")
    p.add_argument("--platen-g", type=float, default=0.0,
                   help="tare that stays on the pad through every step (a platen); 0 when weights sit directly on it")
    p.add_argument("--weights", default=",".join(f"{w:g}" for w in WEIGHT_SET_G), help="pieces available, g")
    p.add_argument("--stack", action="store_true", help="weights can be stacked: use every combination")
    p.add_argument("--tare-step", action="store_true", help="also mark a 0 g step (nothing on the pad) each pass")
    p.add_argument("--masses", default=None, help="only these totals, g")
    p.add_argument("--reps", type=int, default=3, help="passes over the masses per pad")
    p.add_argument("--order", choices=("ascending", "descending", "shuffled", "mixed"), default="mixed",
                   help="mixed: rep 1 ascending, rep 2 descending, rep 3 shuffled, ...")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold-s", type=float, default=15.0, help="seconds a weight stays on (it creeps for ~5-10 s)")
    p.add_argument("--rest-s", type=float, default=8.0, help="unloaded seconds between steps")
    p.add_argument("--baseline-s", type=float, default=5.0, help="unloaded seconds at each pad's start")
    p.add_argument("--last-s", type=float, default=PLATEAU_LAST_S, help="plateau = the last seconds of the hold")
    p.add_argument("--settle-s", type=float, default=3.0, help="label: at least these seconds after 'on' are loading")
    p.add_argument("--release-s", type=float, default=2.0, help="label: seconds after 'off' counted as unloading")
    p.add_argument("--deadband", type=float, default=0.0, help="label: deadband for tactile_response (counts)")
    p.add_argument("--bits", type=int, choices=(8, 12), default=None)
    args = p.parse_args(argv)
    if args.selftest:
        args.simulate = True
    if args.pad and not args.pads:
        args.pads = args.pad
    if not (args.relabel or args.selftest):
        if not (args.simulate or args.record):
            p.error("--record ROOT is required for a real run (or --simulate / --relabel)")
        if not args.pads:
            p.error("--pads all  or  --pads R2C2,R3C3  (or --pad R2C2) is required")
    return args


def build_masses(args):
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    masses = combos(weights, stack=args.stack, tare_step=args.tare_step)
    if args.masses:
        wanted = {round(float(m), 3) for m in args.masses.split(",") if m.strip()}
        masses = [m for m in masses if m[0] in wanted]
        missing = wanted - {m[0] for m in masses}
        if missing:
            raise SystemExit(f"--masses {sorted(missing)} cannot be made from {weights}")
    return masses


def make_source(args):
    if args.simulate:
        return WeightSimSource()
    if args.board == "esp32":
        return Esp32Source(args.port, bits=args.bits)
    source = SerialSource(args.port, BOARDS["m16b"]["baud"], 10.0)
    source.shape, source.adc_max, source.board = (16, 16), 255.0, "m16b"
    return source


def run_selftest(args):
    """Simulated board + scripted marks on a fake clock, two pads; then labels; prints what the reports see."""
    import tempfile

    args.record = Path(tempfile.mkdtemp(prefix="weight_cal_selftest_"))
    args.reps = min(args.reps, 2)
    args.masses = args.masses or "20,100,200"
    source = WeightSimSource(paced=False)
    pads = pad_list("R2C2,R4C4,R1C3", source.shape)  # includes the simulated near-dead pad R4C4
    ctl = Controller(pads, build_masses(args), args, source, clock_ns=source.now_ns)
    fps = 10

    def feed(seconds):
        for _ in range(int(seconds * fps)):
            source.read_latest()

    for _ in pads:
        ctl.space(source.now_ns())  # pad_start -> start_pad
        feed(args.baseline_s + 1)
        ctl.state, ctl.since = "place", time.time()  # what tick() does after --baseline-s of wall time
        for step in ctl.steps:
            ctl.space(source.now_ns())           # on
            for _ in range(int(args.hold_s * fps)):
                source.read_latest()
                ctl.note(float(source.gain[source.pad] * 22.0 * (step[1] * G / 1000) ** 0.7))
            ctl.space(source.now_ns())           # off
            feed(args.rest_s)
            ctl.space(source.now_ns())           # rest -> advance
        if ctl.state == "pad_done":
            ctl.space(source.now_ns())           # next pad
    ctl.close()
    for session in ctl.sessions:
        rows = list(csv.DictReader((session / "force_alignment" / "frames.csv").open()))
        loads = sorted({float(r["load_N"]) for r in rows if float(r["load_N"]) > 0})
        print(f"selftest {session.name}: {len(rows)} frames, loads {[round(x, 3) for x in loads]} N, "
              f"phases {sorted({r['phase'] for r in rows})}")
    print(f"selftest root: {args.record}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.relabel:
        write_labels(args.relabel, args.platen_g, args.settle_s, args.release_s, args.deadband, args.last_s)
        return 0
    if args.selftest:
        import matplotlib
        matplotlib.use("Agg")
        return run_selftest(args)

    import matplotlib
    matplotlib.use("TkAgg")
    source = make_source(args)
    if args.simulate and not args.record:
        import tempfile
        args.record = Path(tempfile.mkdtemp(prefix="weight_cal_sim_"))
    pads = pad_list(args.pads, source.shape, args.shuffle_pads, args.seed)
    masses = build_masses(args)
    per_pad = len(masses) * args.reps
    print(f"{len(pads)} pads x {per_pad} steps ({len(masses)} masses x {args.reps} reps), about "
          f"{len(pads) * (per_pad * (args.hold_s + args.rest_s + 6) + 30) / 60:.0f} min; pad order: "
          + " ".join(pad_name(p) for p in pads), flush=True)
    source.start()
    ctl = Controller(pads, masses, args, source)
    try:
        WeightWindow(args, source, ctl).run()
    finally:
        ctl.close()
        source.close()
        if ctl.sessions:
            print(f"{len(ctl.sessions)} pad session(s) under {args.record}\n"
                  f"next: python weight_summary.py \"{args.record}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
