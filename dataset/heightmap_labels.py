#!/usr/bin/env python3
"""Training pairs: the 16x16 tactile frame and the 16x16 surface depression above it.

    # from the overhead D455 (runs on the capture PC)
    python dataset/heightmap_labels.py --session <capture session> --source d455

    # from a multi-view reconstruction (run_test_session.py output, e.g. on the GPU server)
    python dataset/heightmap_labels.py --session <capture session> --source recon \
        --run-dir reconstruction/outputs/<name>/seconds_<a>_<b>_target_<N>_da3

Output: <session>/labels_<source>.npz (+ a .json summary and a preview PNG).

For every frame, the reconstructed surface is expressed in the SENSOR's own
frame -- along a row, along a column, and along the sensor normal -- using
calibration/sensor_grid.json (the 256 cell centres located from the corner
tapes). Each cell's height is the median height of the surface points inside
its 7.5 x 7.5 mm footprint; its DEPRESSION is how far that height dropped
below the unloaded reference (the median of the frames before the first
press, or --reference A:B seconds). Positive = pressed in, in millimetres.

Points that stand ABOVE the reference surface by more than --above-mm are
dropped first: a press only ever lowers the surface, so those are the
indenter, its rod, or a hand. Cells hidden under the indenter have no points
left; they are NaN in `depression`, and `valid` says which cells were
measured. They are not filled in -- the training code decides how to treat
them (the indenter face itself is known from the Mark-10 travel).

This is the Eulerian label -- the depression at the cell's rest position. Near
steep dent walls the sensor film also slides sideways a little; tracking the
cells (MVTracker queries at the cell centres) would give the Lagrangian label,
to be added if marker tests show the slide matters.

Stored arrays (T frames):
    tactile_adc      (T,16,16)  raw 8-bit ADC, row-major R1..R16 x C1..C16
    depression_mm    (T,16,16)  float, NaN where not measured
    valid            (T,16,16)  bool
    height_mm        (T,16,16)  surface height above the sensor plane
    reference_mm     (16,16)    unloaded height
    points           (T,16,16)  surface points used per cell
    force_n, travel_mm (T,)     zero-corrected load, crosshead position (NaN if absent)
    capture_sequence_index, video_time_s (T,)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reconstruction" / "scripts"))


# ------------------------------------------------------------------ grid
class SensorFrame:
    def __init__(self, grid: dict):
        self.pitch = float(grid["pitch_mm"])
        self.cells = int(grid["cells"])
        self.centre = np.asarray(grid["centre_world"], float)
        self.x = np.asarray(grid["x_axis_world_C1_to_C16"], float)
        self.y = np.asarray(grid["y_axis_world_R1_to_R16"], float)
        n = np.asarray(grid["normal_world"], float)
        self.n = n if n[2] >= 0 else -n            # pointing up, towards the cameras

    def local(self, world: np.ndarray) -> np.ndarray:
        d = world - self.centre
        return np.stack([d @ self.x, d @ self.y, d @ self.n], axis=1)

    def cell_heights(self, world: np.ndarray, ceiling=None, margin=0.0, band=25.0):
        """Median height per cell; `ceiling` (16,16) drops points above it.

        Only points within `band` mm of the sensor plane count: from an oblique
        view, things far above the sensor (the Mark-10 crosshead, a camera arm)
        project into a cell's footprint and would otherwise win the median.
        """
        local = self.local(world)
        local = local[np.abs(local[:, 2]) <= band]
        half = (self.cells - 1) / 2.0
        col = np.floor(local[:, 0] / self.pitch + half + 0.5).astype(int)
        row = np.floor(local[:, 1] / self.pitch + half + 0.5).astype(int)
        # keep points whose footprint is inside the cell (shrunk by `margin`)
        du = local[:, 0] / self.pitch + half - col
        dv = local[:, 1] / self.pitch + half - row
        inside = ((col >= 0) & (col < self.cells) & (row >= 0) & (row < self.cells)
                  & (np.abs(du) <= 0.5 - margin) & (np.abs(dv) <= 0.5 - margin))
        col, row, w = col[inside], row[inside], local[inside, 2]
        if ceiling is not None:
            keep = w <= ceiling[row, col]
            col, row, w = col[keep], row[keep], w[keep]
        heights = np.full((self.cells, self.cells), np.nan)
        counts = np.zeros((self.cells, self.cells), int)
        flat = row * self.cells + col
        order = np.argsort(flat)
        flat, w = flat[order], w[order]
        ids, starts = np.unique(flat, return_index=True)
        for cell, chunk in zip(ids, np.split(w, starts[1:])):
            counts.flat[cell] = chunk.size
            heights.flat[cell] = np.median(chunk)
        return heights, counts


# ------------------------------------------------------------------ sources
def d455_source(session: Path, d455_world: Path):
    """Yields (capture_sequence_index, world points mm) from the D455 depth PNGs."""
    import cv2

    w = json.loads(d455_world.read_text(encoding="utf-8"))
    R = np.asarray(w["R_world_to_depth"], float)
    t = np.asarray(w["t_world_to_depth"], float)
    intr = w["depth_intrinsics"]
    corr = w.get("depth_correction") or {"scale_error": 0.0, "offset_mm": 0.0}
    scale_mm = float(w["depth_scale_m"]) * 1000.0
    cache = {}

    def points(sequence: int):
        path = session / "d455" / "depth" / f"{sequence:09d}.png"
        if not path.exists():
            return None
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float64) * scale_mm
        h, wd = depth.shape
        if (h, wd) not in cache:
            v, u = np.mgrid[0:h, 0:wd].astype(np.float64)
            cache[(h, wd)] = ((u - intr["ppx"]) / intr["fx"], (v - intr["ppy"]) / intr["fy"])
        rx, ry = cache[(h, wd)]
        ok = depth > 0
        z = (depth[ok] - corr["offset_mm"]) / (1.0 + corr["scale_error"])
        cam = np.stack([rx[ok] * z, ry[ok] * z, z], axis=1)
        return (cam - t) @ R                        # depth frame -> world
    return points


def recon_source(run_dir: Path, session: Path):
    """Yields world points from a run_test_session depth cache (all views fused)."""
    from eval_accuracy import Rig

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    data = np.load(next(run_dir.glob(f"depths_{run['depth_backend']}_m.npz")))
    depths, frames = data["depths_m"], data["frame_indices"].astype(int)
    calibration = json.loads((session / "calibration.json").read_text(encoding="utf-8"))
    rig = Rig(calibration, (depths.shape[3], depths.shape[2]))
    by_video = {int(f): k for k, f in enumerate(frames)}

    def points(video_frame: int):
        k = by_video.get(video_frame)
        if k is None:
            return None
        return np.concatenate([rig.backproject(v, depths[v, k])[0] for v in range(depths.shape[0])])
    return points, sorted(by_video)


# ------------------------------------------------------------------ session
def read_alignment(session: Path):
    path = session / "multimodal_video_alignment.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run export_multimodal_mp4.py first")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_tactile(session: Path):
    with (session / "sensor_raw.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cols = [c for c in rows[0] if c.endswith("_adc8")]
    return {int(r["frame_index"]): np.array([float(r[c]) for c in cols]).reshape(16, 16)
            for r in rows}


def number(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return np.nan


# ------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--source", choices=["d455", "recon"], required=True)
    parser.add_argument("--run-dir", type=Path, help="run_test_session output (--source recon)")
    parser.add_argument("--grid", type=Path, default=ROOT / "calibration" / "sensor_grid.json")
    parser.add_argument("--d455-world", type=Path, default=ROOT / "calibration" / "d455_world.json")
    parser.add_argument("--reference", default=None, metavar="A:B",
                        help="unloaded seconds for the reference surface; default: frames "
                             "before the first press (load above 5%% of peak)")
    parser.add_argument("--above-mm", type=float, default=2.0,
                        help="drop points this far above the reference (indenter, hand)")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="shrink each cell footprint by this fraction per side")
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--band-mm", type=float, default=25.0,
                        help="only points this close to the sensor plane are surface "
                             "candidates (must exceed the deepest press)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    session = args.session.expanduser().resolve()
    grid = SensorFrame(json.loads(args.grid.read_text(encoding="utf-8")))
    alignment = read_alignment(session)
    tactile = read_tactile(session)

    if args.source == "d455":
        source = d455_source(session, args.d455_world)
        rows = alignment
        key = lambda row: int(row["capture_sequence_index"])
    else:
        if args.run_dir is None:
            parser.error("--source recon needs --run-dir")
        source, frames = recon_source(args.run_dir, session)
        wanted = set(frames)
        rows = [r for r in alignment if int(r["video_frame_index"]) in wanted]
        key = lambda row: int(row["video_frame_index"])

    times = np.array([number(r, "video_time_s") for r in rows])
    load = np.array([number(r, "force_load") for r in rows])
    travel = np.array([number(r, "force_distance") for r in rows])
    finite = load[np.isfinite(load)]
    if finite.size:
        zero = float(np.median(finite[finite <= np.percentile(finite, 30)]))
        load = load - zero
        if np.nanmax(-load) > np.nanmax(load):
            load = -load

    # reference frames: explicit window, or everything before the first press
    if args.reference:
        a, b = (float(v) for v in args.reference.split(":"))
        ref_mask = (times >= a) & (times <= b)
    elif finite.size and np.nanmax(load) > 0:
        pressed = np.flatnonzero(load > 0.05 * np.nanmax(load))
        first = pressed[0] if pressed.size else len(rows)
        ref_mask = np.arange(len(rows)) < max(first - 1, 1)
    else:
        ref_mask = np.arange(len(rows)) < min(10, len(rows))
    print(f"session {session.name}: {len(rows)} frames from {args.source}; reference "
          f"{int(ref_mask.sum())} frames")

    # pass 1: reference heights (no ceiling yet)
    ref_stack = []
    for row in np.asarray(rows, dtype=object)[ref_mask]:
        pts = source(key(row))
        if pts is not None:
            ref_stack.append(grid.cell_heights(pts, margin=args.margin, band=args.band_mm)[0])
    if not ref_stack:
        raise SystemExit("no reference frame had surface points over the sensor")
    reference = np.nanmedian(np.stack(ref_stack), axis=0)
    ceiling = reference + args.above_mm

    T = len(rows)
    height = np.full((T, 16, 16), np.nan)
    counts = np.zeros((T, 16, 16), int)
    tact = np.full((T, 16, 16), np.nan)
    for k, row in enumerate(rows):
        pts = source(key(row))
        if pts is not None:
            height[k], counts[k] = grid.cell_heights(pts, np.nan_to_num(ceiling, nan=np.inf),
                                                     args.margin, args.band_mm)
        sensor_index = number(row, "sensor_frame_index")
        if np.isfinite(sensor_index) and int(sensor_index) in tactile:
            tact[k] = tactile[int(sensor_index)]
    valid = (counts >= args.min_points) & np.isfinite(height) & np.isfinite(reference)[None]
    depression = np.where(valid, reference[None] - height, np.nan)

    out = args.out or session / f"labels_{args.source}.npz"
    np.savez_compressed(
        out, tactile_adc=tact, depression_mm=depression, valid=valid, height_mm=height,
        reference_mm=reference, points=counts, force_n=load, travel_mm=travel,
        capture_sequence_index=np.array([int(r["capture_sequence_index"]) for r in rows]),
        video_time_s=times)
    summary = {
        "session": str(session), "source": args.source,
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "frames": T, "reference_frames": int(ref_mask.sum()),
        "valid_fraction": float(valid.mean()),
        "reference_flatness_mm": float(np.nanstd(reference)),
        "max_depression_mm": float(np.nanmax(depression)) if np.isfinite(depression).any() else None,
        "grid": str(args.grid), "above_mm": args.above_mm, "margin": args.margin,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"valid cells {valid.mean():.0%}; reference surface flat to {np.nanstd(reference):.2f} mm "
          f"(std over cells); max depression {summary['max_depression_mm']}")
    print(f"wrote {out}")
    preview(out.with_suffix(".png"), depression, tact, load, times)
    return 0


def preview(path: Path, depression, tactile, load, times):
    """Peak-load frame: depression next to the tactile change, plus the curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if np.isfinite(load).any() and np.nanmax(load) > 0:
        k = int(np.nanargmax(load))
    else:
        k = int(np.nanargmax(np.nan_to_num(np.nanmax(depression.reshape(len(depression), -1), 1),
                                            nan=-1)))
    baseline = np.nanmedian(tactile[: max(3, len(tactile) // 10)], axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    im = axes[0].imshow(depression[k], cmap="viridis")
    axes[0].set_title(f"depression (mm), t={times[k]:.1f}s")
    fig.colorbar(im, ax=axes[0])
    im = axes[1].imshow(tactile[k] - baseline, cmap="magma")
    axes[1].set_title("tactile ADC change")
    fig.colorbar(im, ax=axes[1])
    for ax in axes[:2]:
        ax.set_xlabel("column")
        ax.set_ylabel("row")
    peak = np.nanmax(depression.reshape(len(depression), -1), axis=1)
    axes[2].plot(times, peak, label="max depression (mm)")
    if np.isfinite(load).any():
        axes[2].plot(times, load, label="force (N)")
    axes[2].axvline(times[k], color="k", lw=0.8)
    axes[2].legend()
    axes[2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"preview {path}")


if __name__ == "__main__":
    raise SystemExit(main())
