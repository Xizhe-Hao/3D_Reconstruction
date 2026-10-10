#!/usr/bin/env python3
"""Indentation depth per frame: force-gauge ground truth vs reconstructed surface.

    python scripts/eval_indentation.py --result-dir outputs/20261009/seconds_0_end_target_368_da3

Ground truth (Mark-10): each press starts when the load rises above the
unloaded level and stays there (found in the ~85 Hz force samples). The
crosshead distance at that instant is the contact distance of that press, and
the indentation of every video frame inside the press is
``contact distance - distance now``. Each press has its own contact distance
because the sponge does not fully recover between presses.

Reconstructed indentation, per method, per frame: fused height map (median z
per --cell mm cell) in a disk of --radius mm around the dent centre, minus the
height map of the unloaded frames just before that press; the indentation is
the --percentile of that drop over the disk. The indenter hides the bottom of
the dent, so the deepest *visible* surface is its rim, which moves with the
punch; a high percentile reads that rim and ignores the indenter itself
(which stands above the surface and gives a negative drop).

Methods (the same measurement for all of them):
    rgb4        the four RGB cameras' reconstructed depth (depths_<backend>_m.npz)
    rgb4_d455   the same plus the D455 measured depth as a fifth view; each cell
                is the median of the per-view cell medians, so the D455 is one
                vote of five rather than outnumbering the RGB views by pixels
    d455        the D455 alone, as a reference sensor

Writes <result-dir>/indentation.csv (one row per frame, video frame index =
the Rerun ``frame_index``) and indentation_summary.json; export_rerun.py then
plots the curves on the same timeline as the camera views.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_accuracy import Rig  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[1]
METHODS = ("rgb4", "rgb4_d455", "d455")


# ------------------------------------------------------------------ ground truth
def detect_presses(session: Path, threshold_n: float, min_duration_s: float) -> tuple[list[dict], float]:
    """Sustained loaded segments in the high-rate force samples."""
    with (session / "force_samples_aligned.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    time = np.array([float(r["force_time_s"]) for r in rows])
    load = np.array([float(r["force_load"]) for r in rows])
    dist = np.array([float(r["force_distance"]) for r in rows])
    # The gauge is not zeroed (-7.4 N unloaded on 2026-10-09): the unloaded
    # level is the median of the lowest 30 % of readings, as in eval_accuracy.
    low = load[load <= np.percentile(load, 30)]
    baseline = float(np.median(low))
    rel = load - baseline
    if np.max(-rel) > np.max(rel):  # compression read as negative
        rel = -rel
    on = rel > threshold_n
    presses = []
    k = 0
    while k < len(on):
        if not on[k]:
            k += 1
            continue
        end = k
        while end < len(on) and on[end]:
            end += 1
        # Crosshead start/stop jerks give sub-second load spikes; a press holds.
        if time[end - 1] - time[k] >= min_duration_s:
            presses.append({"start_s": float(time[k]), "end_s": float(time[end - 1]),
                            "contact_distance": float(dist[k]),
                            "min_distance": float(dist[k:end].min()),
                            "max_distance": float(dist[k:end].max()),
                            "peak_load": float(rel[k:end].max())})
        k = end
    # Pressing lowers the crosshead on this stand; if it counts up, flip.
    for press in presses:
        down = press["contact_distance"] - press["min_distance"]
        up = press["max_distance"] - press["contact_distance"]
        press["sign"] = 1.0 if down >= up else -1.0
    return presses, baseline


# ------------------------------------------------------------------ geometry
def d455_world_points(session: Path, d455_world: dict, rel_path: str) -> np.ndarray | None:
    path = session / rel_path
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if rel_path else None
    if raw is None:
        return None
    intr = d455_world["depth_intrinsics"]
    R = np.asarray(d455_world["R_world_to_depth"], np.float64)
    t = np.asarray(d455_world["t_world_to_depth"], np.float64)
    corr = d455_world.get("depth_correction") or {"scale_error": 0.0, "offset_mm": 0.0}
    v, u = np.nonzero(raw > 0)
    z = raw[v, u].astype(np.float64) * float(d455_world["depth_scale_m"]) * 1000.0
    z = (z - corr["offset_mm"]) / (1.0 + corr["scale_error"])
    cam = np.stack([(u - intr["ppx"]) / intr["fx"] * z, (v - intr["ppy"]) / intr["fy"] * z, z], axis=1)
    return (cam - t) @ R


class HeightGrid:
    """Median z per cell of a disk around the dent centre."""

    def __init__(self, centre_xy, radius: float, cell: float, z_range):
        self.centre = np.asarray(centre_xy, np.float64)
        self.radius, self.cell, self.z_range = radius, cell, z_range
        self.n = int(np.ceil(2 * radius / cell))
        ij = (np.arange(self.n) + 0.5) * cell - radius
        gx, gy = np.meshgrid(ij, ij)
        self.inside = (np.hypot(gx, gy) <= radius).ravel()

    def of(self, points: np.ndarray) -> np.ndarray:
        grid = np.full(self.n * self.n, np.nan)
        if points is None or not len(points):
            return grid
        d = points[:, :2] - self.centre + self.radius
        keep = ((d >= 0) & (d < 2 * self.radius)).all(axis=1)
        keep &= (points[:, 2] >= self.z_range[0]) & (points[:, 2] <= self.z_range[1])
        idx = (d[keep] // self.cell).astype(np.int64)
        flat = idx[:, 1] * self.n + idx[:, 0]
        z = points[keep, 2]
        order = np.argsort(flat, kind="stable")
        flat, z = flat[order], z[order]
        cells, start = np.unique(flat, return_index=True)
        for cell_id, segment in zip(cells, np.split(z, start[1:])):
            if segment.size >= 3:
                grid[cell_id] = np.median(segment)
        grid[~self.inside] = np.nan
        return grid


def fuse(grids: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(grids)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(stack, axis=0)


def locate_centre(d455_points_ref, d455_points_peak, box, cell: float = 2.0, z_sponge=(25.0, 80.0)):
    """Centroid of the cells that dropped by >= half the deepest drop (D455 at peak load)."""
    x0, x1, y0, y1 = box
    nx, ny = int((x1 - x0) / cell), int((y1 - y0) / cell)

    def grid(points):
        g = np.full(nx * ny, np.nan)
        keep = (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
        keep &= (points[:, 2] >= z_sponge[0] - 40) & (points[:, 2] <= z_sponge[1] + 60)
        p = points[keep]
        flat = ((p[:, 1] - y0) // cell).astype(int) * nx + ((p[:, 0] - x0) // cell).astype(int)
        order = np.argsort(flat)
        cells, start = np.unique(flat[order], return_index=True)
        for c, seg in zip(cells, np.split(p[order, 2], start[1:])):
            if seg.size >= 3:
                g[c] = np.median(seg)
        return g.reshape(ny, nx)

    ref = fuse([grid(p) for p in d455_points_ref])
    peak = fuse([grid(p) for p in d455_points_peak])
    drop = ref - peak
    drop[~((ref >= z_sponge[0]) & (ref <= z_sponge[1]))] = np.nan  # sponge top only
    if not np.isfinite(drop).any():
        return None
    deep = np.nanpercentile(drop, 99.5)
    rows, cols = np.nonzero(np.nan_to_num(drop, nan=-np.inf) >= 0.5 * deep)
    return [float(x0 + (cols.mean() + 0.5) * cell), float(y0 + (rows.mean() + 0.5) * cell)]


# ------------------------------------------------------------------ report
def error_stats(estimate: np.ndarray, truth: np.ndarray) -> dict:
    ok = np.isfinite(estimate) & np.isfinite(truth)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    e = estimate[ok] - truth[ok]
    slope = float(np.sum(estimate[ok] * truth[ok]) / max(np.sum(truth[ok] ** 2), 1e-9))
    return {"n": int(ok.sum()), "bias_mm": float(np.mean(e)), "mae_mm": float(np.mean(np.abs(e))),
            "rmse_mm": float(np.sqrt(np.mean(e ** 2))), "p90_abs_mm": float(np.percentile(np.abs(e), 90)),
            "max_abs_mm": float(np.max(np.abs(e))), "slope": slope,
            "corr": float(np.corrcoef(estimate[ok], truth[ok])[0, 1]) if np.std(truth[ok]) > 0 else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, help="default: session_dir in run.json")
    parser.add_argument("--d455-world", type=Path, default=REPO_ROOT / "calibration" / "d455_world.json")
    parser.add_argument("--centre-xy", type=float, nargs=2, metavar=("X", "Y"),
                        help="dent centre in world mm; default: located from the D455 at peak load")
    parser.add_argument("--radius", type=float, default=30.0, help="disk radius (mm) read around the centre")
    parser.add_argument("--cell", type=float, default=2.0, help="height-map cell (mm)")
    parser.add_argument("--percentile", type=float, default=95.0, help="percentile of the surface drop used")
    parser.add_argument("--z-range", type=float, nargs=2, default=(0.0, 120.0), metavar=("Z0", "Z1"))
    parser.add_argument("--contact-threshold", type=float, default=1.0, help="N above the unloaded level")
    parser.add_argument("--min-press-s", type=float, default=0.5)
    parser.add_argument("--reference-frames", type=int, default=6,
                        help="unloaded frames before each press averaged as its rest surface")
    args = parser.parse_args()

    result = args.result_dir.expanduser().resolve()
    run = json.loads((result / "run.json").read_text(encoding="utf-8"))
    session = (args.session_dir or Path(run["session_dir"])).expanduser().resolve()
    with np.load(result / f"depths_{run['depth_backend']}_m.npz") as data:
        depths = data["depths_m"]
        frames = data["frame_indices"].astype(int)
    rig = Rig(json.loads((session / "calibration.json").read_text(encoding="utf-8")),
              (depths.shape[3], depths.shape[2]))
    d455_world = json.loads(args.d455_world.read_text(encoding="utf-8"))
    meta = json.loads((session / "d455" / "d455.json").read_text(encoding="utf-8"))
    if str(meta.get("serial")) != str(d455_world.get("serial")):
        raise SystemExit(f"{args.d455_world} is for D455 {d455_world.get('serial')}, session has {meta.get('serial')}")
    with (session / "multimodal_video_alignment.csv").open(newline="", encoding="utf-8-sig") as handle:
        alignment = {int(r["video_frame_index"]): r for r in csv.DictReader(handle)}

    def num(row, key):
        try:
            return float(row.get(key, ""))
        except ValueError:
            return np.nan

    rows = [alignment.get(int(f), {}) for f in frames]
    force_time = np.array([num(r, "force_time_s") for r in rows])
    distance = np.array([num(r, "force_distance") for r in rows])
    presses, baseline = detect_presses(session, args.contact_threshold, args.min_press_s)
    if not presses:
        raise SystemExit("no sustained press found in force_samples_aligned.csv")
    print(f"force: unloaded level {baseline:+.2f} N, {len(presses)} presses")

    # Ground truth and press membership per frame.
    T = len(frames)
    truth = np.where(np.isfinite(force_time), 0.0, np.nan)
    press_of = np.full(T, -1)
    for p, press in enumerate(presses):
        inside = (force_time >= press["start_s"]) & (force_time <= press["end_s"])
        press_of[inside] = p
        truth[inside] = np.maximum(press["sign"] * (press["contact_distance"] - distance[inside]), 0.0)
    loaded = press_of >= 0

    # Rest surface of each press: the last unloaded frames before it (by frame
    # order -- frames after the force log ends have no force time at all).
    starts = np.array([np.flatnonzero(press_of == p)[0] for p in range(len(presses))])
    reference_frames = []
    for p in range(len(presses)):
        before = np.flatnonzero(~loaded & (np.arange(T) < starts[p]))
        reference_frames.append(before[-args.reference_frames:].tolist())
    # Which press's rest surface each frame is compared with: the latest press
    # that has started, or the first press before any.
    ref_of = np.clip(np.searchsorted(starts, np.arange(T), side="right") - 1, 0, len(presses) - 1)

    d455_cache: dict[int, np.ndarray | None] = {}

    def d455_at(k: int):
        if k not in d455_cache:
            d455_cache[k] = d455_world_points(session, d455_world, rows[k].get("d455_depth_png", ""))
        return d455_cache[k]

    centre = args.centre_xy
    if centre is None:
        p_peak = int(np.argmax([p["peak_load"] for p in presses]))
        hold = np.flatnonzero(press_of == p_peak)
        deepest = hold[np.nanargmax(truth[hold])]
        centre = locate_centre([d455_at(k) for k in reference_frames[p_peak]],
                               [d455_at(k) for k in (deepest - 1, deepest, deepest + 1) if 0 <= k < T],
                               box=(-150, 150, -200, 100))
        if centre is None:
            raise SystemExit("could not locate the dent; pass --centre-xy X Y")
        print(f"dent centre located from the D455 at ({centre[0]:.1f}, {centre[1]:.1f}) mm "
              "-- pass --centre-xy to override")
    grid = HeightGrid(centre, args.radius, args.cell, args.z_range)

    def view_grids(k: int) -> list[np.ndarray]:
        out = [grid.of(rig.backproject(v, depths[v, k])[0]) for v in range(depths.shape[0])]
        out.append(grid.of(d455_at(k)))
        return out

    def method_grids(per_view: list[np.ndarray]) -> dict[str, np.ndarray]:
        return {"rgb4": fuse(per_view[:-1]), "rgb4_d455": fuse(per_view), "d455": per_view[-1]}

    references = []
    for frames_p in reference_frames:
        per_frame = [method_grids(view_grids(k)) for k in frames_p]
        references.append({m: fuse([g[m] for g in per_frame]) for m in METHODS})

    estimate = {m: np.full(T, np.nan) for m in METHODS}
    coverage = {m: np.zeros(T, int) for m in METHODS}
    for k in range(T):
        current = method_grids(view_grids(k))
        for m in METHODS:
            drop = references[ref_of[k]][m] - current[m]
            drop = drop[np.isfinite(drop)]
            coverage[m][k] = drop.size
            if drop.size >= 20:
                estimate[m][k] = float(np.percentile(drop, args.percentile))
        d455_cache.pop(k - 2, None)
        if k % 50 == 0:
            print(f"  frame {k}/{T}", flush=True)

    out_csv = result / "indentation.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "video_time_s", "press", "force_distance_mm", "ground_truth_mm",
                         *[f"{m}_mm" for m in METHODS], *[f"{m}_cells" for m in METHODS]])
        for k in range(T):
            writer.writerow([int(frames[k]), round(float(run["times_s"][k]), 4), int(press_of[k]),
                             "" if not np.isfinite(distance[k]) else round(float(distance[k]), 3),
                             "" if not np.isfinite(truth[k]) else round(float(truth[k]), 3),
                             *["" if not np.isfinite(estimate[m][k]) else round(float(estimate[m][k]), 3)
                               for m in METHODS],
                             *[int(coverage[m][k]) for m in METHODS]])

    with np.errstate(all="ignore"):
        surface_z = float(np.nanmedian(references[0]["rgb4_d455"]))
    summary = {"result_dir": str(result), "centre_xy_mm": centre, "radius_mm": args.radius,
               "surface_z_mm": surface_z,
               "cell_mm": args.cell, "percentile": args.percentile, "force_unloaded_level_n": baseline,
               "presses": [{**p, "frames": int((press_of == i).sum()),
                            "reference_frames": [int(frames[k]) for k in reference_frames[i]]}
                           for i, p in enumerate(presses)],
               "loaded_frames": {m: error_stats(estimate[m][loaded], truth[loaded]) for m in METHODS},
               "unloaded_frames_false_dent": {m: error_stats(estimate[m][~loaded & np.isfinite(truth)],
                                                             truth[~loaded & np.isfinite(truth)])
                                              for m in METHODS},
               "per_press_peak": []}
    for i in range(len(presses)):
        hold = np.flatnonzero(press_of == i)
        deepest = hold[np.argsort(truth[hold])[-max(3, len(hold) // 4):]]  # deepest quarter of the press
        summary["per_press_peak"].append({"press": i, "ground_truth_mm": float(np.median(truth[deepest])),
                                          **{m: float(np.nanmedian(estimate[m][deepest])) for m in METHODS}})
    (result / "indentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nindentation vs force-gauge ground truth, loaded frames (mm; + = reconstructed too deep)")
    print(f"{'method':10s} {'n':>4s} {'bias':>7s} {'MAE':>6s} {'RMSE':>6s} {'p90':>6s} {'slope':>6s} {'corr':>5s}")
    for m in METHODS:
        s = summary["loaded_frames"][m]
        if "mae_mm" in s:
            print(f"{m:10s} {s['n']:4d} {s['bias_mm']:+7.2f} {s['mae_mm']:6.2f} {s['rmse_mm']:6.2f} "
                  f"{s['p90_abs_mm']:6.2f} {s['slope']:6.3f} {s['corr'] or float('nan'):5.2f}")
    print("\nper press, deepest quarter (median mm):")
    for entry in summary["per_press_peak"]:
        print(f"  press {entry['press']}: truth {entry['ground_truth_mm']:6.2f}  " +
              "  ".join(f"{m} {entry[m]:6.2f}" for m in METHODS))
    print(f"\nwrote {out_csv}\n      {result / 'indentation_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
