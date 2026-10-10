#!/usr/bin/env python3
"""Where each of the 16x16 tactile cells is in the world frame.

    python calibration/sensor_grid.py --session <capture session>          # click the tapes
    python calibration/sensor_grid.py --from-clicks <sensor_grid.json>     # recompute
    python calibration/sensor_grid.py --selftest

RUN WITH THE CALIBRATION ENVIRONMENT (.venv-calib).

Four black tapes mark the corners of the active area; the one at R1C1 is
L-shaped. In one unloaded frame of a capture session, click the INNER corner
of each tape (the corner pointing at the middle of the sensor) in every
Blackfly view that shows it, in this order:

    1  R1C1    the L-shaped tape
    2  R1C16   the other end of row 1
    3  R16C16  diagonally opposite the L
    4  R16C1   the other end of column 1

(Not sure which neighbour of the L is R1C16? Press near it with
live_sensor_2d.py and read the row/column under your finger.)

Each corner needs clicks in at least two views; it is triangulated with
calibration.json. The grid is then placed from the tapes' CENTRE and EDGE
DIRECTIONS only, with the known pitch -- so it does not depend on how far the
tapes sit from the corner cells, as long as all four sit about equally far.
That distance is only used as a check: the tape square should measure
(cells - 1) * pitch + 2 * offset.

Output (default calibration/sensor_grid.json): cell centres [row][col] in mm,
the grid axes, the clicked pixels (so it can be recomputed), and checks.
Overlays of the grid on every view are written next to the session so the
result can be judged by eye. Re-do it whenever the sponge or the sensor is
moved.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
CORNERS = ["R1C1", "R1C16", "R16C16", "R16C1"]
CORNER_HELP = {"R1C1": "the L-shaped tape", "R1C16": "other end of row 1",
               "R16C16": "diagonally opposite the L", "R16C1": "other end of column 1"}


# ------------------------------------------------------------------ geometry
def load_rig(path: Path):
    cams = sorted(json.loads(path.read_text(encoding="utf-8"))["cameras"],
                  key=lambda c: c["index"])
    return [{"serial": c["serial"], "K": np.asarray(c["K"], float),
             "dist": np.asarray(c["dist"], float).reshape(-1),
             "R": np.asarray(c["R_world_to_cam"], float),
             "t": np.asarray(c["t_world_to_cam"], float).reshape(3)} for c in cams]


def project(rig, view, points):
    cam = rig[view]
    rvec, _ = cv.Rodrigues(cam["R"])
    uv, _ = cv.projectPoints(np.asarray(points, float).reshape(-1, 1, 3), rvec, cam["t"],
                             cam["K"], cam["dist"])
    return uv.reshape(-1, 2)


def triangulate(rig, observations: dict[int, np.ndarray]):
    rows = []
    for view, pixel in observations.items():
        cam = rig[view]
        n = cv.undistortPoints(np.asarray(pixel, float).reshape(1, 1, 2),
                               cam["K"], cam["dist"]).reshape(2)
        P = np.hstack([cam["R"], cam["t"].reshape(3, 1)])
        rows += [n[0] * P[2] - P[0], n[1] * P[2] - P[1]]
    _, _, vt = np.linalg.svd(np.asarray(rows))
    point = vt[-1][:3] / vt[-1][3]
    errors = [np.linalg.norm(project(rig, v, point)[0] - np.asarray(p, float))
              for v, p in observations.items()]
    return point, float(np.sqrt(np.mean(np.square(errors))))


def fit_grid(corners: dict[str, np.ndarray], pitch: float, cells: int, offset: float):
    """Grid from the tape square's centre and edge directions."""
    c1, c2, c3, c4 = (corners[name] for name in CORNERS)
    x = (c2 - c1) + (c3 - c4)                    # along a row: C1 -> C16
    y = (c4 - c1) + (c3 - c2)                    # along a column: R1 -> R16
    skew = float(np.degrees(np.arccos(abs(x @ y) / np.linalg.norm(x) / np.linalg.norm(y))))
    x = x / np.linalg.norm(x)
    y = y - (y @ x) * x
    y = y / np.linalg.norm(y)
    normal = np.cross(x, y)
    centre = (c1 + c2 + c3 + c4) / 4.0
    half = (cells - 1) / 2.0
    rows, cols = np.mgrid[0:cells, 0:cells]
    centres = (centre + (cols[..., None] - half) * pitch * x
               + (rows[..., None] - half) * pitch * y)
    sides = {"row1 (R1C1-R1C16)": float(np.linalg.norm(c2 - c1)),
             "row16 (R16C1-R16C16)": float(np.linalg.norm(c3 - c4)),
             "col1 (R1C1-R16C1)": float(np.linalg.norm(c4 - c1)),
             "col16 (R1C16-R16C16)": float(np.linalg.norm(c3 - c2))}
    planarity = float(abs((c1 - centre) @ normal) + abs((c2 - centre) @ normal)
                      + abs((c3 - centre) @ normal) + abs((c4 - centre) @ normal)) / 4
    expected = (cells - 1) * pitch + 2 * offset
    checks = {"sides_mm": sides, "expected_side_mm": expected,
              "implied_tape_offset_mm": float((np.mean(list(sides.values()))
                                                - (cells - 1) * pitch) / 2),
              "skew_from_square_deg": abs(90.0 - skew) if skew > 45 else skew,
              "corner_out_of_plane_mm": planarity,
              "normal_tilt_from_world_z_deg": float(np.degrees(np.arccos(abs(normal[2]))))}
    return {"centre": centre, "x_axis": x, "y_axis": y, "normal": normal,
            "cell_centres": centres, "checks": checks}


# ------------------------------------------------------------------ session io
def load_frame(session: Path, view: int, frame: int | None):
    """Grey full-resolution image of one camera; BMPs first, MP4 otherwise."""
    folders = sorted(p for p in session.glob(f"camera_{view}_*") if p.is_dir())
    if folders:
        with (folders[0] / "frames.csv").open(newline="", encoding="utf-8") as h:
            rows = [r for r in csv.DictReader(h) if r.get("complete", "1") == "1"]
        candidates = rows if frame is None else [r for r in rows
                                                 if int(r["capture_sequence_index"]) == frame]
        for row in candidates[5:] if frame is None else candidates:
            path = session / row["image_path"].replace("\\", "/")
            if not path.exists():
                continue
            raw = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            gray = cv.cvtColor(raw, cv.COLOR_BayerBG2GRAY) if raw.ndim == 2 else \
                cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
            if gray.mean() > 15:                 # the first frames can be black
                return gray, int(row["capture_sequence_index"])
    export = session / "multimodal_video_export.json"
    if export.exists():
        video = session / json.loads(export.read_text(encoding="utf-8"))["videos"][view]
        capture = cv.VideoCapture(str(video))
        capture.set(cv.CAP_PROP_POS_FRAMES, 5 if frame is None else frame)
        ok, image = capture.read()
        capture.release()
        if ok:
            return cv.cvtColor(image, cv.COLOR_BGR2GRAY), (5 if frame is None else frame)
    raise SystemExit(f"no usable image for camera {view} in {session}")


# ------------------------------------------------------------------ clicking
ZOOM_HALF = 110        # px around the coarse click shown in the fine step


def click_view(image, view, serial, predicted=None):
    """Collect the four inner corners in one view; None where not visible.

    Two clicks per corner: a coarse one anywhere on the tape, which zooms in,
    then the exact inner corner. Once two views are done, the remaining ones
    show where each corner should be (cyan); 'a' accepts that spot as the
    coarse click.
    """
    import matplotlib.pyplot as plt

    predicted = predicted or {}
    clicks: list = []
    state = {"zoom": None}
    fig, ax = plt.subplots(figsize=(14, 10.5))
    ax.imshow(image, cmap="gray")
    ax.set_axis_off()
    full = (ax.get_xlim(), ax.get_ylim())
    artists = []
    for name, uv in predicted.items():
        if uv is None:
            continue
        ax.plot(*uv, "o", mfc="none", mec="cyan", ms=26, mew=2)
        ax.text(uv[0] + 18, uv[1] + 26, f"{name}?", color="cyan", fontsize=11)

    def title():
        if len(clicks) >= 4:
            ax.set_title(f"camera {view}: done -- close the window or press q", fontsize=12)
        else:
            name = CORNERS[len(clicks)]
            step = ("ZOOMED: click the exact INNER corner (pointing at the sensor middle)"
                    if state["zoom"] else
                    "click anywhere on the tape" + (", or press a to use the cyan guess"
                                                    if predicted.get(name) else ""))
            ax.set_title(f"camera {view} ({serial}) -- {name} ({CORNER_HELP[name]}): {step}" + chr(10) +
                         "n = not visible here   u = undo   Esc = leave zoom", fontsize=11)
        fig.canvas.draw_idle()

    def redraw():
        for a in artists:
            a.remove()
        artists.clear()
        for name, point in zip(CORNERS, clicks):
            if point is not None:
                artists.append(ax.plot(*point, "+", color="red" if name == "R1C1" else "yellow",
                                       ms=24, mew=2)[0])
                artists.append(ax.text(point[0] + 10, point[1] - 10, name, color="yellow",
                                       fontsize=12, weight="bold"))
        title()

    def zoom_to(x, y):
        state["zoom"] = (x, y)
        ax.set_xlim(x - ZOOM_HALF, x + ZOOM_HALF)
        ax.set_ylim(y + ZOOM_HALF, y - ZOOM_HALF)
        title()

    def unzoom():
        state["zoom"] = None
        ax.set_xlim(*full[0])
        ax.set_ylim(*full[1])

    def on_click(event):
        if event.inaxes != ax or fig.canvas.toolbar.mode or event.button != 1:
            return
        if len(clicks) >= 4:
            return
        if state["zoom"] is None:
            zoom_to(float(event.xdata), float(event.ydata))
        else:
            clicks.append((float(event.xdata), float(event.ydata)))
            unzoom()
            redraw()

    def on_key(event):
        name = CORNERS[len(clicks)] if len(clicks) < 4 else None
        if event.key == "n" and name:
            clicks.append(None)
            unzoom()
            redraw()
        elif event.key == "a" and name and predicted.get(name) and state["zoom"] is None:
            zoom_to(*predicted[name])
        elif event.key == "escape" and state["zoom"] is not None:
            unzoom()
            title()
        elif event.key == "u" and clicks:
            clicks.pop()
            unzoom()
            redraw()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    title()
    plt.tight_layout()
    plt.show()
    clicks += [None] * (4 - len(clicks))
    return {name: (list(p) if p is not None else None) for name, p in zip(CORNERS, clicks)}


def predict(rig, clicks: dict, view: int) -> dict:
    """Where each corner should appear in `view`, from the views clicked so far."""
    out = {}
    for name in CORNERS:
        seen = {int(v): np.asarray(c[name], float) for v, c in clicks.items()
                if c.get(name) is not None}
        if len(seen) >= 2:
            point, residual = triangulate(rig, seen)
            uv = project(rig, view, point)[0]
            h, w = 1536, 2048
            out[name] = uv.tolist() if (residual < 6 and 0 <= uv[0] < w and 0 <= uv[1] < h) else None
    return out


def consistency(rig, clicks: dict):
    """Per corner and view: how far the click is from the other views' answer."""
    lines = []
    for name in CORNERS:
        seen = {int(v): np.asarray(c[name], float) for v, c in clicks.items()
                if c.get(name) is not None}
        if len(seen) < 2:
            continue
        point, _ = triangulate(rig, seen)
        errs = {v: float(np.linalg.norm(project(rig, v, point)[0] - p)) for v, p in seen.items()}
        flag = "  <- re-click" if max(errs.values()) > 6 else ""
        lines.append(f"   {name:7s} " + "  ".join(f"cam{v} {e:5.1f}px" for v, e in errs.items()) + flag)
    return lines


# ------------------------------------------------------------------ solve + report
def solve(rig, clicks: dict[str, dict], pitch, cells, offset):
    corners, residuals = {}, {}
    for name in CORNERS:
        seen = {int(v): np.asarray(c[name], float) for v, c in clicks.items()
                if c.get(name) is not None}
        if len(seen) < 2:
            raise SystemExit(f"{name} was clicked in {len(seen)} view(s); it needs two")
        corners[name], residuals[name] = triangulate(rig, seen)
    grid = fit_grid(corners, pitch, cells, offset)
    grid["corners"] = corners
    grid["click_residual_px"] = residuals
    return grid


def report(grid, offset):
    print("tape inner corners (world mm) and click consistency:")
    for name in CORNERS:
        p = grid["corners"][name]
        print(f"   {name:7s} ({p[0]:7.1f}, {p[1]:7.1f}, {p[2]:6.1f})   views disagree by "
              f"{grid['click_residual_px'][name]:.1f} px")
    checks = grid["checks"]
    print("tape square sides: " + ", ".join(f"{k} {v:.1f}" for k, v in checks["sides_mm"].items())
          + f" mm  (expected {checks['expected_side_mm']:.1f} for a {offset:g} mm offset)")
    print(f"implied tape offset {checks['implied_tape_offset_mm']:.1f} mm; skew "
          f"{checks['skew_from_square_deg']:.1f} deg; corners off-plane "
          f"{checks['corner_out_of_plane_mm']:.1f} mm; sensor tilted "
          f"{checks['normal_tilt_from_world_z_deg']:.1f} deg from the platform")
    sides = list(checks["sides_mm"].values())
    warnings = []
    if max(sides) - min(sides) > 4:
        warnings.append("the four sides differ by more than 4 mm -- a corner was "
                        "mis-clicked, or the tapes are not placed alike")
    if checks["skew_from_square_deg"] > 3:
        warnings.append("the tape quadrilateral is not square -- check the click order")
    if max(grid["click_residual_px"].values()) > 4:
        warnings.append("a corner's clicks disagree by > 4 px between views -- re-click it")
    for w in warnings:
        print("WARNING: " + w)
    return warnings


def overlays(rig, grid, images, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    centres = grid["cell_centres"].reshape(-1, 3)
    for view, image in images.items():
        canvas = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        uv = project(rig, view, centres)
        for k, (u, v) in enumerate(uv):
            r, c = divmod(k, 16)
            colour = (0, 0, 255) if (r, c) == (0, 0) else \
                (0, 200, 255) if r == 0 else (0, 255, 0)
            cv.circle(canvas, (int(round(u)), int(round(v))), 6 if (r, c) == (0, 0) else 3,
                      colour, -1)
        for name in CORNERS:
            u, v = project(rig, view, grid["corners"][name])[0]
            cv.drawMarker(canvas, (int(u), int(v)), (255, 0, 255), cv.MARKER_CROSS, 30, 2)
            cv.putText(canvas, name, (int(u) + 10, int(v) - 10), cv.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 0, 255), 2)
        path = out_dir / f"grid_camera_{view}.jpg"
        cv.imwrite(str(path), canvas)
    print(f"overlays: {out_dir}  (red = R1C1, orange = row 1, green = the rest)")


def save(path: Path, grid, clicks, args, session, frame):
    payload = {
        "schema": "sensor-grid/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "units": "millimetre",
        "frame": "calibration world frame (calibration.json)",
        "calibration": str(args.calibration.resolve()),
        "session": str(session) if session else None,
        "capture_sequence_index": frame,
        "cells": args.cells, "pitch_mm": args.pitch, "tape_offset_mm_assumed": args.tape_offset,
        "corners_world": {k: v.tolist() for k, v in grid["corners"].items()},
        "centre_world": grid["centre"].tolist(),
        "x_axis_world_C1_to_C16": grid["x_axis"].tolist(),
        "y_axis_world_R1_to_R16": grid["y_axis"].tolist(),
        "normal_world": grid["normal"].tolist(),
        "cell_centres_world": grid["cell_centres"].round(4).tolist(),
        "checks": grid["checks"],
        "click_residual_px": grid["click_residual_px"],
        "clicks": clicks,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def selftest(args) -> int:
    rig = load_rig(args.calibration)
    rng = np.random.default_rng(4)
    centre = np.array([65.0, -75.0, 48.0])
    angle = np.radians(17)
    x = np.array([np.cos(angle), np.sin(angle), 0.0])
    y = np.array([-np.sin(angle), np.cos(angle), 0.0])
    half = (args.cells - 1) / 2 * args.pitch + args.tape_offset
    truth = {"R1C1": centre - half * x - half * y, "R1C16": centre + half * x - half * y,
             "R16C16": centre + half * x + half * y, "R16C1": centre - half * x + half * y}
    clicks = {}
    for view in range(len(rig)):
        clicks[str(view)] = {}
        for name, p in truth.items():
            uv = project(rig, view, p)[0]
            in_view = 0 <= uv[0] < 2048 and 0 <= uv[1] < 1536   # off-image = not clicked
            clicks[str(view)][name] = ((uv + rng.normal(0, 2.0, 2)).tolist()
                                       if in_view else None)
    grid = solve(rig, clicks, args.pitch, args.cells, args.tape_offset)
    rows, cols = np.mgrid[0:args.cells, 0:args.cells]
    h = (args.cells - 1) / 2
    expected = centre + (cols[..., None] - h) * args.pitch * x + (rows[..., None] - h) * args.pitch * y
    error = np.linalg.norm(grid["cell_centres"] - expected, axis=-1)
    report(grid, args.tape_offset)
    ok = error.max() < 0.6 and abs(grid["checks"]["implied_tape_offset_mm"] - args.tape_offset) < 0.5
    print(f"cell centre error with 2 px click noise: median {np.median(error):.2f} mm, "
          f"max {error.max():.2f} mm")
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--frame", type=int, default=None,
                        help="capture_sequence_index to click on (unloaded); default: first "
                             "bright frame after the first five")
    parser.add_argument("--views", default="0,1,2,3")
    parser.add_argument("--from-clicks", type=Path, help="recompute from a saved sensor_grid.json")
    parser.add_argument("--calibration", type=Path, default=HERE / "calibration.json")
    parser.add_argument("--pitch", type=float, default=7.5)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--tape-offset", type=float, default=7.0,
                        help="tape inner corner to the nearest cell centre (only a check)")
    parser.add_argument("--out", type=Path, default=HERE / "sensor_grid.json")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest(args)

    rig = load_rig(args.calibration)
    if args.from_clicks:
        saved = json.loads(args.from_clicks.read_text(encoding="utf-8"))
        clicks = saved["clicks"]
        session = Path(saved["session"]) if saved.get("session") else None
        frame = saved.get("capture_sequence_index")
    else:
        if args.session is None:
            parser.error("--session (or --from-clicks) is required")
        session, frame, clicks = args.session.expanduser().resolve(), args.frame, {}
        for view in (int(v) for v in args.views.split(",")):
            image, frame_used = load_frame(session, view, frame)
            frame = frame if frame is not None else frame_used
            while True:
                clicks[str(view)] = click_view(image, view, rig[view]["serial"],
                                               predict(rig, clicks, view))
                lines = consistency(rig, clicks)
                if lines:
                    print(f"after camera {view} -- distance of each click from the other "
                          "views' answer:")
                    print(chr(10).join(lines))
                if any("re-click" in line for line in lines):
                    answer = input(f"re-click camera {view}? [Y/n] ").strip().lower()
                    if answer in ("", "y", "yes"):
                        continue
                break
    grid = solve(rig, clicks, args.pitch, args.cells, args.tape_offset)
    report(grid, args.tape_offset)
    save(args.out, grid, clicks, args, session, frame)
    if session is not None and session.exists():
        images = {int(v): load_frame(session, int(v), frame)[0] for v in clicks}
        overlays(rig, grid, images, session / "sensor_grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
