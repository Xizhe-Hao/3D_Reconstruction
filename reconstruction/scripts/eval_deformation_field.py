#!/usr/bin/env python3
"""Whole-surface deformation: the RGB reconstruction's displacement field vs the depth camera's.

    python scripts/eval_deformation_field.py --result-dir outputs/20261110_d405/seconds_14_75_step_1_da3

eval_indentation.py reads one number per frame, the dent depth under the punch, which no
camera can see (the punch hides the dent bottom). This script scores the deformation of the
whole sponge top instead, against the depth camera (D405/D455) as the reference surface.

Per frame, each sensor's height map (median z per --cell mm cell over a square of
+-box-mm around the dent centre) is compared with that sensor's own rest surface (the
unloaded frames before the press, as in eval_indentation), giving a vertical displacement
field  d = rest - now  (+ = pushed down). Sources: the four RGB views fused (cell median
across views, "rgb4") and each view alone ("view0".."view3"). Scored cells:
    sponge   the depth camera's rest height is within --sponge-z (the foam top)
    minus    a disk of --punch-mm around the dent centre, any cell where the depth camera
             sees something --above-mm above its rest surface (the punch, its arm), and
             --erode-cells cells off every edge of what is left (an edge cell mixes the top
             with the sponge's side, the punch or the arm's shadow)

Needs indentation.csv / indentation_summary.json in the result dir (run eval_indentation.py
first: it supplies the presses, rest frames and dent centre). Writes to --out-dir
(default <result-dir>/deformation_field): field_summary.json, field_per_frame.csv,
displacement_fields.npz, field_maps.png and field_profiles.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_accuracy import Rig  # noqa: E402
from eval_indentation import d455_world_points  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[1]


def nanmedian(stack: list[np.ndarray]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(stack), axis=0)


class SquareGrid:
    """Median z per cell of a square around the dent centre (vectorised)."""

    def __init__(self, centre, half: float, cell: float):
        self.centre, self.half, self.cell = np.asarray(centre, np.float64), half, cell
        self.n = int(round(2 * half / cell))
        g = (np.arange(self.n) + 0.5) * cell - half
        gx, gy = np.meshgrid(g, g)
        self.radius = np.hypot(gx, gy).ravel()

    def of(self, points: np.ndarray | None) -> np.ndarray:
        out = np.full(self.n * self.n, np.nan)
        if points is None or not len(points):
            return out
        d = points[:, :2] - self.centre + self.half
        keep = ((d >= 0) & (d < 2 * self.half)).all(1) & (points[:, 2] > -20) & (points[:, 2] < 160)
        idx = (d[keep] // self.cell).astype(np.int64)
        flat, z = idx[:, 1] * self.n + idx[:, 0], points[keep, 2]
        order = np.lexsort((z, flat))
        flat, z = flat[order], z[order]
        cells, start, count = np.unique(flat, return_index=True, return_counts=True)
        ok = count >= 3
        out[cells[ok]] = 0.5 * (z[(start + (count - 1) // 2)[ok]] + z[(start + count // 2)[ok]])
        return out


def pooled(reference: np.ndarray, estimate: np.ndarray, cells: np.ndarray | None = None) -> dict:
    sel = np.isfinite(reference) & np.isfinite(estimate)
    if cells is not None:
        sel &= cells[None, :]
    x, y = reference[sel], estimate[sel]
    if x.size < 20:
        return {"cells": int(x.size)}
    e = y - x
    return {"cells": int(x.size), "reference_mean_disp_mm": float(np.mean(x)), "rgb_mean_disp_mm": float(np.mean(y)),
            "mae_mm": float(np.mean(np.abs(e))), "median_abs_mm": float(np.median(np.abs(e))),
            "rmse_mm": float(np.sqrt(np.mean(e ** 2))), "bias_mm": float(np.mean(e)),
            "p90_abs_mm": float(np.percentile(np.abs(e), 90)),
            "slope": float(np.sum(x * y) / max(np.sum(x * x), 1e-9)), "corr": float(np.corrcoef(x, y)[0, 1])}


def ring_series(reference: np.ndarray, estimate: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-frame mean displacement over a ring (cells both sources see), and how well they agree."""
    both = np.isfinite(reference) & np.isfinite(estimate) & cells[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.nanmean(np.where(both, reference, np.nan), axis=1)
        e = np.nanmean(np.where(both, estimate, np.nan), axis=1)
    ok = np.isfinite(r) & np.isfinite(e)
    stats = {"frames": int(ok.sum())}
    if ok.sum() > 3:
        stats.update(corr=float(np.corrcoef(r[ok], e[ok])[0, 1]), mae_mm=float(np.mean(np.abs(e[ok] - r[ok]))),
                     slope=float(np.sum(r[ok] * e[ok]) / max(np.sum(r[ok] ** 2), 1e-9)))
    return r, e, stats


def plot(out: Path, grid: SquareGrid, disp: dict, truth: np.ndarray, press_of: np.ndarray, loaded: np.ndarray,
         series: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, cx, cy, B = grid.n, *grid.centre, grid.half
    ext = [cx - B, cx + B, cy - B, cy + B]
    ref, rgb = disp["reference"], disp["rgb4"]
    presses = sorted({int(p) for p in press_of if p >= 0})

    def peak_frames(p):
        hold = np.flatnonzero(press_of == p)
        return hold[np.argsort(truth[hold])[-5:]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        show = sorted(presses, key=lambda p: -np.nanmax(truth[press_of == p]))[:2]
        fig, ax = plt.subplots(2, 3, figsize=(17, 10))
        for col, p in enumerate(show):
            k = peak_frames(p)
            for row, (name, f) in enumerate((("depth camera", ref), ("RGB4", rgb))):
                im = ax[row, col].imshow(np.nanmean(f[k], 0).reshape(n, n), origin="lower", extent=ext,
                                         cmap="RdBu_r", vmin=-8, vmax=8)
                ax[row, col].set_title(f"press {p} peak (punch {np.median(truth[k]):.1f} mm): {name}")
                plt.colorbar(im, ax=ax[row, col], label="downward displacement (mm)")
        for a, img, t in ((ax[0, 2], np.nanmean(np.abs(rgb[loaded] - ref[loaded]), 0), "|RGB4 - depth camera|, loaded frames"),
                          (ax[1, 2], np.nanmean(np.abs(ref[~loaded]), 0), "|depth camera| at rest (its own noise)")):
            im = a.imshow(img.reshape(n, n), origin="lower", extent=ext, cmap="magma", vmin=0, vmax=3)
            a.set_title(t)
            plt.colorbar(im, ax=a, label="mm")
        for a in ax.ravel():
            a.plot(cx, cy, "k+")
            a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)")
        fig.tight_layout()
        fig.savefig(out / "field_maps.png", dpi=80)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1 + len(series), figsize=(7 * (1 + len(series)), 5.5))
        edges = np.arange(10, 82, 4)
        mid = 0.5 * (edges[1:] + edges[:-1])
        colours = ["#1b6ca8", "#d1495b", "#2a9d8f", "#e9a03b", "#6a4c93", "#8a5a44"]
        both = np.isfinite(ref) & np.isfinite(rgb)
        R, E = np.where(both, ref, np.nan), np.where(both, rgb, np.nan)
        for p in presses:
            k = peak_frames(p)
            prof = [[np.nanmean(S[k][:, (grid.radius > a) & (grid.radius <= b)]) for a, b in zip(edges[:-1], edges[1:])]
                    for S in (R, E)]
            c = colours[p % len(colours)]
            ax[0].plot(mid, prof[0], "-", color=c, lw=2, label=f"press {p} depth camera (punch {np.median(truth[k]):.0f} mm)")
            ax[0].plot(mid, prof[1], "--", color=c, lw=2, label=f"press {p} RGB4")
        ax[0].set_xlabel("distance from dent centre (mm)"); ax[0].set_ylabel("mean downward displacement (mm)")
        ax[0].set_title("radial profile at each press peak"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        t = np.arange(len(truth))
        for a, (name, (r, e, s)) in zip(ax[1:], series.items()):
            a.plot(t, r, color="#1b6ca8", lw=1.8, label="depth camera")
            a.plot(t, e, color="#d1495b", lw=1.2, label="RGB4")
            tw = a.twinx(); tw.plot(t, truth, color="0.6", lw=1); tw.set_ylabel("punch travel (mm)", color="0.5")
            a.set_title(f"ring {name} mm mean: corr {s.get('corr', np.nan):.2f}, MAE {s.get('mae_mm', np.nan):.2f} mm, "
                        f"slope {s.get('slope', np.nan):.2f}")
            a.set_xlabel("frame"); a.set_ylabel("mm"); a.legend(loc="upper left"); a.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out / "field_profiles.png", dpi=80)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, help="default: <result-dir>/deformation_field")
    parser.add_argument("--session-dir", type=Path, help="default: session_dir in run.json")
    parser.add_argument("--d455-world", type=Path, default=REPO_ROOT / "calibration" / "d455_world.json")
    parser.add_argument("--box-mm", type=float, default=110.0, help="half-width of the square scored around the dent")
    parser.add_argument("--cell", type=float, default=2.0)
    parser.add_argument("--sponge-z", type=float, nargs=2, default=(30.0, 60.0), metavar=("Z0", "Z1"))
    parser.add_argument("--punch-mm", type=float, default=10.0, help="disk around the dent centre left out")
    parser.add_argument("--above-mm", type=float, default=3.0, help="depth-camera cell this far above rest = punch/arm")
    parser.add_argument("--erode-cells", type=int, default=3, help="cells trimmed off every edge of the scored area")
    parser.add_argument("--rings", default="10-30,30-60", help="rings (mm) for the per-frame mean series")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    result = args.result_dir.expanduser().resolve()
    out = (args.out_dir or result / "deformation_field").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    run = json.loads((result / "run.json").read_text(encoding="utf-8"))
    summary_in = json.loads((result / "indentation_summary.json").read_text(encoding="utf-8"))
    session = (args.session_dir or Path(run["session_dir"])).expanduser().resolve()
    with np.load(result / f"depths_{run['depth_backend']}_m.npz") as data:
        depths, frames = data["depths_m"], data["frame_indices"].astype(int)
    V, T, H, W = depths.shape
    rig = Rig(json.loads((session / "calibration.json").read_text(encoding="utf-8")), (W, H))
    d455_world = json.loads(args.d455_world.read_text(encoding="utf-8"))
    with (session / "multimodal_video_alignment.csv").open(newline="", encoding="utf-8-sig") as handle:
        alignment = {int(r["video_frame_index"]): r for r in csv.DictReader(handle)}
    with (result / "indentation.csv").open(newline="", encoding="utf-8") as handle:
        rows_in = list(csv.DictReader(handle))
    if [int(r["frame_index"]) for r in rows_in] != frames.tolist():
        raise SystemExit("indentation.csv does not match the depth frames; rerun eval_indentation.py")
    press_of = np.array([int(r["press"]) for r in rows_in])
    truth = np.array([float(r["ground_truth_mm"]) if r["ground_truth_mm"] else np.nan for r in rows_in])
    loaded = press_of >= 0
    grid = SquareGrid(summary_in["centre_xy_mm"], args.box_mm, args.cell)
    N = grid.n * grid.n

    sources = ["reference", "rgb4"] + [f"view{v}" for v in range(V)]
    rgb_sources = sources[1:]

    def grids(k: int) -> dict[str, np.ndarray]:
        per_view = [grid.of(rig.backproject(v, depths[v, k])[0]) for v in range(V)]
        cam = d455_world_points(session, d455_world, alignment.get(int(frames[k]), {}).get("d455_depth_png", ""))
        return {"reference": grid.of(cam), "rgb4": nanmedian(per_view), **{f"view{v}": g for v, g in enumerate(per_view)}}

    position = {f: k for k, f in enumerate(frames.tolist())}
    rests = []
    for press in summary_in["presses"]:
        g = [grids(position[f]) for f in press["reference_frames"]]
        rests.append({s: nanmedian([x[s] for x in g]) for s in sources})
    starts = np.array([np.flatnonzero(press_of == p)[0] for p in range(len(rests))])
    rest_of = np.clip(np.searchsorted(starts, np.arange(T), side="right") - 1, 0, len(rests) - 1)
    sponge = [np.isfinite(r["reference"]) & (r["reference"] >= args.sponge_z[0]) & (r["reference"] <= args.sponge_z[1])
              for r in rests]
    kernel = np.ones((2 * args.erode_cells + 1,) * 2, np.uint8) if args.erode_cells else None

    disp = {s: np.full((T, N), np.nan, np.float32) for s in sources}
    per_frame = []
    for k in range(T):
        now, rest = grids(k), rests[rest_of[k]]
        d = {s: rest[s] - now[s] for s in sources}
        valid = (sponge[rest_of[k]] & (grid.radius > args.punch_mm) & np.isfinite(d["reference"])
                 & ~(now["reference"] - rest["reference"] > args.above_mm))
        if kernel is not None:
            valid = cv2.erode(valid.reshape(grid.n, grid.n).astype(np.uint8), kernel, borderValue=0).astype(bool).ravel()
        for s in sources:
            disp[s][k] = np.where(valid, d[s], np.nan)
        row = {"frame_index": int(frames[k]), "press": int(press_of[k]),
               "ground_truth_mm": "" if not np.isfinite(truth[k]) else round(float(truth[k]), 3),
               "reference_cells": int(valid.sum()),
               "reference_mean_disp_mm": round(float(np.mean(d["reference"][valid])), 3) if valid.any() else ""}
        for s in rgb_sources:
            both = valid & np.isfinite(d[s])
            e = d[s][both] - d["reference"][both]
            row[f"{s}_cells"] = int(both.sum())
            row[f"{s}_mae_mm"] = round(float(np.mean(np.abs(e))), 3) if e.size else ""
            row[f"{s}_bias_mm"] = round(float(np.mean(e)), 3) if e.size else ""
        per_frame.append(row)
        if k % 50 == 0:
            print(f"  frame {k}/{T}", flush=True)

    deepest = np.zeros(T, bool)
    for p in range(len(rests)):
        hold = np.flatnonzero(press_of == p)
        deepest[hold[np.argsort(truth[hold])[-max(3, len(hold) // 4):]]] = True
    ring_bounds = [(args.punch_mm, 20), (20, 30), (30, 50), (50, 80), (80, 1e9)]
    series_rings = [tuple(float(x) for x in r.split("-")) for r in args.rings.split(",") if r]
    series = {f"{a:g}-{b:g}": ring_series(disp["reference"], disp["rgb4"], (grid.radius > a) & (grid.radius <= b))
              for a, b in series_rings}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        noise = disp["reference"][~loaded]
        summary = {"result_dir": str(result), "centre_xy_mm": grid.centre.tolist(), "box_mm": args.box_mm,
                   "cell_mm": args.cell, "sponge_z_mm": list(args.sponge_z), "punch_exclusion_mm": args.punch_mm,
                   "above_mm": args.above_mm, "erode_cells": args.erode_cells,
                   "reference_noise_at_rest": {"mae_mm": float(np.nanmean(np.abs(noise))),
                                               "rms_mm": float(np.sqrt(np.nanmean(noise ** 2)))},
                   "ring_mean_series": {name: s for name, (_, _, s) in series.items()},
                   "sets": {}}
    for name, mask in (("loaded", loaded), ("deepest_quarter", deepest), ("unloaded", ~loaded)):
        summary["sets"][name] = {
            "frames": int(mask.sum()),
            "cells": {s: pooled(disp["reference"][mask], disp[s][mask]) for s in rgb_sources},
            "rings_rgb4": {f"{a:g}-{b:g}mm" if b < 1e8 else f"{a:g}mm+":
                           pooled(disp["reference"][mask], disp["rgb4"][mask], (grid.radius > a) & (grid.radius <= b))
                           for a, b in ring_bounds}}
    (out / "field_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "field_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_frame[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame)
    np.savez_compressed(out / "displacement_fields.npz", frame_indices=frames, centre_xy_mm=grid.centre,
                        box_mm=args.box_mm, cell_mm=args.cell, **{f"disp_{s}": disp[s] for s in sources})
    if not args.no_plots:
        plot(out, grid, disp, truth, press_of, loaded, series)

    noise = summary["reference_noise_at_rest"]
    print(f"\ndepth camera at rest (its own noise): MAE {noise['mae_mm']:.2f} mm, RMS {noise['rms_mm']:.2f} mm")
    for name in ("loaded", "deepest_quarter", "unloaded"):
        S = summary["sets"][name]
        print(f"\n[{name}] {S['frames']} frames: RGB displacement vs depth-camera displacement, per cell")
        print(f"{'source':8s} {'cells':>8s} {'MAE':>6s} {'MedAE':>6s} {'RMSE':>6s} {'bias':>6s} {'slope':>6s} {'corr':>5s}")
        for s in rgb_sources:
            r = S["cells"][s]
            if "mae_mm" in r:
                print(f"{s:8s} {r['cells']:8d} {r['mae_mm']:6.2f} {r['median_abs_mm']:6.2f} {r['rmse_mm']:6.2f} "
                      f"{r['bias_mm']:+6.2f} {r['slope']:6.3f} {r['corr']:5.2f}")
        print("  rgb4 by distance from the dent centre:")
        for ring, r in S["rings_rgb4"].items():
            if "mae_mm" in r:
                print(f"    {ring:>9s} cells {r['cells']:7d}  ref {r['reference_mean_disp_mm']:6.2f}  "
                      f"RGB {r['rgb_mean_disp_mm']:6.2f}  MAE {r['mae_mm']:5.2f}  slope {r['slope']:5.2f}  corr {r['corr']:5.2f}")
    print("\nper-frame ring mean (rgb4 vs depth camera):")
    for name, s in summary["ring_mean_series"].items():
        if "corr" in s:
            print(f"  {name:>7s} mm: corr {s['corr']:.3f}, MAE {s['mae_mm']:.2f} mm, slope {s['slope']:.2f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
