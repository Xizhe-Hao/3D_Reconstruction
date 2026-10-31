#!/usr/bin/env python3
"""Align an IntelliMESUR force export to a tactile capture, press by press.

    python align_force_presses.py <capture session> <force export .csv>
    python align_force_presses.py <session> <csv> --warp linear

Runs in seconds on long runs, where align_force_curve_only.py's brute-force
search takes many minutes. It does three things:

1. Coarse: both curves are put on the tactile frame grid and compared with a
   masked, normalised cross-correlation (FFT), for every offset at once and for
   a range of time scales. The export's "Time" column is not real time: it ran
   5 % slow on 2026-10-12 and 17 % slow on 2026-10-09, so the scale is searched,
   0.80-1.30 by default.
2. Fine: every press is found in both streams (force > --force-onset N,
   tactile response > --tactile-onset), paired through the coarse mapping, and
   the onsets -- the moment of contact -- become anchors of a piecewise-linear
   time warp. Release times are not used: the sensor recovers more slowly than
   the load drops, so they would bias the clock.
3. Labels: every tactile frame gets load, crosshead distance, indentation from
   that press's contact, press number and phase (loading / hold / unloading).

A capture that stalled (no frame for > 1 s, e.g. a click in the PowerShell
window) is cut at the stall: the frames flushed at Ctrl+C carry wrong times.

Works for any grid (16x16 M16B or 4x4 ESP32 sessions): the grid comes from
the CSV header, the frame times from host_received_ns, and the press
polarity from the data (see tactile_preprocess).

Outputs in <session>/force_alignment/:
    frames.csv      one row per tactile frame with the force labels
    alignment.json  scale, offset, warp anchors, residuals, warnings
    alignment.png   overlay, tactile vs force / indentation, residuals
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tactile_preprocess import load_raw, response_sign, session_meta

STALL_S = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("force_csv", type=Path)
    p.add_argument("--scale-range", type=float, nargs=2, default=[0.80, 1.30],
                   help="real seconds per export second")
    p.add_argument("--scale-step", type=float, default=0.0025)
    p.add_argument("--min-overlap", type=float, default=0.6,
                   help="coarse match must overlap this fraction of the shorter stream")
    p.add_argument("--force-onset", type=float, default=1.0, help="N; contact")
    p.add_argument("--tactile-onset", type=float, default=None,
                   help="response sum; default 10x the unloaded noise, at least 50")
    p.add_argument("--min-press-s", type=float, default=1.0)
    p.add_argument("--max-pair-s", type=float, default=3.0, help="pairing window after the coarse map")
    p.add_argument("--baseline-s", type=float, default=5.0, help="unloaded seconds at the start")
    p.add_argument("--warp", choices=["piecewise", "linear"], default="piecewise")
    p.add_argument("--clock", type=float, nargs=2, default=None, metavar=("SCALE", "OFFSET"),
                   help="skip the search: tactile_t = SCALE * force_t + OFFSET (e.g. from matched step edges)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


# ---------------------------------------------------------------- loading


def load_tactile(session: Path, baseline_s: float, warnings: list[str]):
    ns, idx, frames = load_raw(session)  # grid size from the header: 16x16 or 4x4
    gaps = np.flatnonzero(np.diff(ns) > STALL_S * 1e9)
    if gaps.size:
        k = gaps[0]
        warnings.append(f"capture stalled for {(ns[k + 1] - ns[k]) / 1e9:.0f} s after "
                        f"{(ns[k] - ns[0]) / 1e9:.1f} s; frames after the stall are dropped")
        ns, idx, frames = ns[:k + 1], idx[:k + 1], frames[:k + 1]
    t = (ns - ns[0]) / 1e9
    base_mask = t <= baseline_s
    base = np.median(frames[base_mask], axis=0)
    # The press direction depends on the board: the 16x16 board's ADC rises under load, the 4x4's falls.
    sign = response_sign(frames, base, session_meta(session, *frames.shape[1:]))
    total = np.clip(sign * (frames - base), 0, None).sum(axis=(1, 2))
    noise = float(np.std(total[base_mask]))
    return ns, idx, t, total, noise, frames.shape[1:]


def load_force(path: Path):
    meta, data, columns = {}, [], None
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            fields = [f.strip() for f in line.rstrip("\r\n").split("\t")]
            if columns is None:
                if fields[0] == "Reading":
                    columns = [f.split(" [")[0] for f in fields]
                elif ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip()
                continue
            try:
                data.append([float(f) for f in fields[:4]])
            except ValueError:
                continue
    if not data:
        raise SystemExit(f"no data rows in {path}")
    data = np.asarray(data)
    col = {name: i for i, name in enumerate(columns) if name}
    ft, load, dist = data[:, col["Time"]], data[:, col["Load"]], data[:, col["Distance"]]
    sign = 1.0 if load.max() >= -load.min() else -1.0
    return meta, ft - ft[0], sign * load, dist


# ---------------------------------------------------------------- coarse


def masked_ncc(x: np.ndarray, y: np.ndarray, mx: np.ndarray, my: np.ndarray):
    """Pearson correlation of x and y for every lag (y shifted by lag), over the overlap."""
    n = len(x) + len(y) - 1
    size = 1 << (n - 1).bit_length()

    def xc(a, b):
        return np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)

    x, y = x * mx, y * my
    cnt = xc(mx, my)
    sx, sy = xc(x, my), xc(mx, y)
    sxx, syy = xc(x * x, my), xc(mx, y * y)
    sxy = xc(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / cnt
        var = (sxx - sx * sx / cnt) * (syy - sy * sy / cnt)
        r = cov / np.sqrt(var)
    lags = np.r_[np.arange(0, len(x)), np.arange(-(len(y) - 1), 0)]
    keep = np.r_[np.arange(0, len(x)), np.arange(size - (len(y) - 1), size)]
    return lags, r[keep], np.round(cnt[keep]).astype(int)


def coarse_alignment(t, total, ft, load, args):
    dt = float(np.median(np.diff(t)))
    grid_t = np.arange(0, t[-1], dt)
    x = np.interp(grid_t, t, total)
    best = None
    for scale in np.arange(args.scale_range[0], args.scale_range[1] + 1e-9, args.scale_step):
        grid_f = np.arange(0, ft[-1] * scale, dt)
        y = np.interp(grid_f / scale, ft, load)
        lags, r, cnt = masked_ncc(x, y, np.ones_like(x), np.ones_like(y))
        need = args.min_overlap * min(len(x), len(y))
        r = np.where(cnt >= need, r, -np.inf)
        k = int(np.nanargmax(r))
        if best is None or r[k] > best[0]:
            best = (float(r[k]), float(scale), float(lags[k] * dt))
    return best  # corr, scale, offset: tactile_t = scale * force_t + offset


# ---------------------------------------------------------------- fine


def press_onsets(x, tt, threshold, min_len):
    active = x > threshold
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if a == 0 or tt[b - 1] - tt[a] < min_len:
            continue
        frac = (threshold - x[a - 1]) / (x[a] - x[a - 1])
        out.append((tt[a - 1] + frac * (tt[a] - tt[a - 1]), a, b))
    return out


def fit_warp(t, total, ft, load, coarse, tactile_onset, args, warnings):
    _, scale, offset = coarse
    tp = press_onsets(total, t, tactile_onset, args.min_press_s)
    fp = press_onsets(load, ft, args.force_onset, args.min_press_s)
    linear = np.array([[0.0, offset], [ft[-1], scale * ft[-1] + offset]])
    if args.clock:
        return scale, offset, linear, np.zeros((0, 2)), np.zeros(0), fp
    t_on = np.array([p[0] for p in tp])
    pairs, used = [], set()
    for f_on, a, b in fp:
        guess = scale * f_on + offset
        if t_on.size == 0:
            break
        k = int(np.argmin(np.abs(t_on - guess)))
        if abs(t_on[k] - guess) <= args.max_pair_s and k not in used:
            used.add(k)
            pairs.append((f_on, t_on[k]))
    pairs = np.array(pairs)
    covered = [p for p in fp if 0 <= scale * p[0] + offset <= t[-1]]
    if len(pairs) < len(covered):
        warnings.append(f"{len(covered) - len(pairs)} of {len(covered)} force presses inside the "
                        "tactile recording found no tactile partner")
    if len(pairs) < 2:
        warnings.append("fewer than 2 paired presses; keeping the coarse mapping")
        return scale, offset, linear, pairs, np.zeros(0), fp
    (lin_scale, lin_off), *_ = np.linalg.lstsq(np.c_[pairs[:, 0], np.ones(len(pairs))], pairs[:, 1], rcond=None)
    residual = pairs[:, 1] - (lin_scale * pairs[:, 0] + lin_off)
    if args.warp == "linear":
        anchors = np.array([[0.0, lin_off], [ft[-1], lin_scale * ft[-1] + lin_off]])
    else:  # piecewise through every onset, extended with the linear slope
        anchors = np.vstack([[pairs[0, 0] - 1e4, pairs[0, 1] - 1e4 * lin_scale], pairs,
                             [pairs[-1, 0] + 1e4, pairs[-1, 1] + 1e4 * lin_scale]])
    return lin_scale, lin_off, anchors, pairs, residual, fp


# ---------------------------------------------------------------- labels and outputs


def label_frames(t, ft, load, dist, anchors, fp, args):
    f_of_t = np.interp(t, anchors[:, 1], anchors[:, 0])  # export time for every tactile frame
    inside = (f_of_t >= 0) & (f_of_t <= ft[-1])
    lab_load = np.where(inside, np.interp(f_of_t, ft, load), np.nan)
    lab_dist = np.where(inside, np.interp(f_of_t, ft, dist), np.nan)
    indent = np.full_like(t, np.nan)
    press = np.zeros(len(t), dtype=int)
    into = np.zeros(len(t))  # +1 where the crosshead moves into the sample
    for n, (f_on, a, b) in enumerate(fp, 1):
        b = min(b, len(ft) - 1)
        contact = dist[a]
        direction = np.sign(dist[a:b + 1][np.argmax(np.abs(dist[a:b + 1] - contact))] - contact) or -1.0
        sel = inside & (f_of_t >= ft[a]) & (f_of_t <= ft[b])
        press[sel] = n
        indent[sel] = np.clip((lab_dist[sel] - contact) * direction, 0, None)
        into[sel] = direction
    velocity = np.gradient(np.interp(f_of_t, ft, dist), t) * into  # mm/s into the sample
    phase = np.where(press == 0, "free",
                     np.where(np.abs(velocity) < 0.05, "hold",
                              np.where(velocity > 0, "loading", "unloading")))
    return f_of_t, inside, lab_load, lab_dist, indent, press, phase


def plot(path, t, total, lab_load, indent, phase, inside, pairs, residual, scale, coarse_r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    a = ax[0, 0]
    a.plot(t, total / np.nanmax(total), lw=.7, label="tactile (norm.)")
    a.plot(t, lab_load / np.nanmax(lab_load), lw=.7, alpha=.8, label="force (norm.)")
    a.set(xlabel="capture time [s]", title=f"aligned (scale {scale:.4f}, coarse r {coarse_r:.2f})")
    a.legend()
    colors = {"loading": "tab:blue", "hold": "tab:green", "unloading": "tab:red"}
    for name, c in colors.items():
        s = inside & (phase == name) & (lab_load > 0.5)
        ax[0, 1].scatter(lab_load[s], total[s], s=4, c=c, label=name)
        ax[1, 0].scatter(indent[s], total[s], s=4, c=c, label=name)
    ax[0, 1].set(xlabel="force [N]", ylabel="tactile response", title="tactile vs force")
    ax[1, 0].set(xlabel="indentation from contact [mm]", ylabel="tactile response", title="tactile vs indentation")
    ax[0, 1].legend()
    ax[1, 0].legend()
    if len(pairs):
        ax[1, 1].plot(pairs[:, 1], residual * 1000, "o-")
    ax[1, 1].axhline(0, color="0.5", lw=.6)
    ax[1, 1].set(xlabel="capture time [s]", ylabel="onset residual to a straight clock fit [ms]",
                 title="clock drift (absorbed by the piecewise warp)")
    for a in ax.ravel():
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    ns, idx, t, total, noise, grid = load_tactile(args.session, args.baseline_s, warnings)
    meta, ft, load, dist = load_force(args.force_csv)
    tactile_onset = args.tactile_onset or max(50.0, 10 * noise)
    tare = float(np.median(load[:50]))
    if abs(tare) > 0.3:
        warnings.append(f"force not tared: {tare:+.1f} N at the start; subtracted before use")
        load = load - tare

    coarse = (float("nan"), *args.clock) if args.clock else coarse_alignment(t, total, ft, load, args)
    scale, offset, anchors, pairs, residual, fp = fit_warp(t, total, ft, load, coarse, tactile_onset, args, warnings)
    f_of_t, inside, lab_load, lab_dist, indent, press, phase = label_frames(t, ft, load, dist, anchors, fp, args)

    out = args.out or args.session / "force_alignment"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "frames.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["host_received_ns", "frame_index", "t_s", "force_time_s", "load_N", "distance_mm",
                    "indentation_mm", "press", "phase", "tactile_response"])
        for k in range(len(t)):
            if not inside[k]:
                continue
            w.writerow([ns[k], idx[k], f"{t[k]:.4f}", f"{f_of_t[k]:.4f}", f"{lab_load[k]:.3f}",
                        f"{lab_dist[k]:.3f}", "" if np.isnan(indent[k]) else f"{indent[k]:.3f}",
                        press[k], phase[k], f"{total[k]:.1f}"])
    covered_to = float(np.interp(t[-1], anchors[:, 1], anchors[:, 0]))
    loaded = inside & (lab_load > 0.5)
    summary = {
        "session": str(args.session), "force_csv": str(args.force_csv),
        "grid": [int(grid[0]), int(grid[1])], "frames": int(len(t)),
        "frame_rate_fps": float(1.0 / np.median(np.diff(t))) if len(t) > 2 else None,
        "coarse": {"corr": coarse[0], "scale": coarse[1], "offset_s": coarse[2]},
        "clock": {"scale": scale, "offset_s": offset, "warp": args.warp,
                  "onset_residual_ms": {"rms": float(np.sqrt(np.mean(residual ** 2)) * 1000) if residual.size else None,
                                        "max": float(np.abs(residual).max() * 1000) if residual.size else None}},
        "presses": {"force": len(fp), "paired": int(len(pairs))},
        "force_run_covered_s": [0.0, min(covered_to, float(ft[-1]))], "force_run_s": float(ft[-1]),
        "tactile_onset": tactile_onset, "frames_labelled": int(inside.sum()),
        "corr_tactile_force": float(np.corrcoef(total[loaded], lab_load[loaded])[0, 1]) if loaded.sum() > 10 else None,
        "anchors_force_s_to_tactile_s": anchors[1:-1].tolist() if args.warp == "piecewise" else anchors.tolist(),
        "force_sensor": meta.get("Force Sensor Model"), "calibration_due": meta.get("Calibration Due Date"),
        "warnings": warnings,
    }
    (out / "alignment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(out / "alignment.png", t, total, lab_load, indent, phase, inside, pairs, residual, scale, coarse[0])

    print(f"coarse: r {coarse[0]:.3f}, scale {coarse[1]:.4f}, offset {coarse[2]:.2f} s")
    rms = summary["clock"]["onset_residual_ms"]["rms"]
    print(f"presses: {len(pairs)} paired of {len(fp)} in the force run; clock scale {scale:.5f}, offset {offset:.2f} s, "
          f"onset residual to a straight fit {rms:.0f} ms rms" if rms is not None else "presses: none paired")
    print(f"force run covered: {summary['force_run_covered_s'][1]:.1f} of {ft[-1]:.1f} s; "
          f"{inside.sum()} frames labelled; tactile~force r = {summary['corr_tactile_force']}")
    for w_ in warnings:
        print("WARNING:", w_)
    print(f"wrote {out / 'frames.csv'}, alignment.json, alignment.png")


if __name__ == "__main__":
    main()
