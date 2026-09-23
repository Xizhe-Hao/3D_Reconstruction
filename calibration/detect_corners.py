#!/usr/bin/env python3
"""Detect ChArUco corners in every frame of a capture session.

    python detect_corners.py <session_dir> --board board/board.json --out detections.npz

Detection is by far the slowest step, so it is a separate stage: run it once,
then iterate on `calibrate.py` as much as you like.

Two things happen here that a naive script gets wrong:

* the saved BMPs are **raw Bayer**, not grayscale (`demosaic_on_save` is off by
  default in the capture script). Reading them as if they were grayscale leaves
  the mosaic pattern in the image and destroys sub-pixel corner accuracy, so we
  demosaic with the pattern recorded in `session.json`;
* frames are keyed by `capture_sequence_index` from `frames.csv`, i.e. by the
  Arduino trigger pulse, not by file order.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
import numpy as np

from charuco_board import BoardSpec, make_detector
from session_io import load_session, read_bgr, read_gray

SUBPIX_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect ChArUco corners across a synchronised capture session",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("session", type=Path, help="capture_YYYYmmdd_HHMMSS directory")
    parser.add_argument("--board", type=Path, required=True, help="board.json")
    parser.add_argument("--out", type=Path, default=Path("detections.npz"))
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth trigger; at 24 fps neighbouring frames "
                             "are near-duplicates and add cost, not information")
    parser.add_argument("--frames", default=None, metavar="A:B",
                        help="restrict to capture_sequence_index in [A, B)")
    parser.add_argument("--min-corners", type=int, default=6,
                        help="below this a view is recorded as 'not detected'")
    parser.add_argument("--bayer", default=None,
                        help="override the Bayer pattern (BG/RG/GB/GR/none); "
                             "default is read from session.json")
    parser.add_argument("--subpix-win", default="auto",
                        help="cornerSubPix half-window in px, or 'auto', or 0 to skip")
    parser.add_argument("--workers", type=int, default=0,
                        help="detection threads; 0 = one per CPU")
    parser.add_argument("--debug-dir", type=Path, default=None,
                        help="write annotated overlays for a few frames per camera")
    parser.add_argument("--debug-count", type=int, default=3)
    return parser.parse_args()


def refine(gray: np.ndarray, corners: np.ndarray, mode) -> np.ndarray:
    """Optional extra cornerSubPix pass on top of the ChArUco refinement."""
    if not isinstance(mode, str) and int(mode) <= 0:
        return corners
    if len(corners) < 2:
        return corners

    if mode == "auto":
        # Nearest-neighbour spacing tells us how large a window can be before
        # it swallows the neighbouring corner and biases the fit.
        diff = corners[:, None, :] - corners[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        spacing = float(np.median(dist.min(axis=1)))
        half = int(np.clip(round(spacing / 4.0), 3, 11))
    else:
        half = int(mode)

    refined = corners.reshape(-1, 1, 2).astype(np.float32).copy()
    cv.cornerSubPix(gray, refined, (half, half), (-1, -1), SUBPIX_CRITERIA)
    return refined.reshape(-1, 2)


def foreshortening(object_points: np.ndarray, image_points: np.ndarray):
    """How square-on the board is, without needing camera intrinsics.

    Fits the board plane to the image with a homography and returns the ratio of
    the smaller to the larger singular value of its linear part: 1.0 means the
    board faces the camera, values near 0 mean it is seen edge-on. A camera that
    only ever sees small values cannot constrain its own focal length.
    """
    if len(object_points) < 8:
        return None
    homography, _ = cv.findHomography(
        np.ascontiguousarray(object_points[:, :2], np.float64),
        np.ascontiguousarray(image_points, np.float64), 0)
    if homography is None:
        return None
    singular = np.linalg.svd(homography[:2, :2], compute_uv=False)
    if singular[0] < 1e-9:
        return None
    return float(singular[1] / singular[0])


def main() -> int:
    args = parse_args()
    spec = BoardSpec.load(args.board)
    board = spec.build()
    num_pts = spec.num_corners

    object_points = spec.object_points()
    session = load_session(args.session, args.bayer)
    print(f"session   : {session.root}")
    print(f"cameras   : {', '.join(f'{c.index}:{c.serial}' for c in session.cameras)}")
    print(f"bayer     : {session.bayer or 'none (already grayscale/BGR on disk)'}")
    print(f"board     : {spec.describe()}")

    frames = session.frame_indices
    if args.frames:
        lo, hi = (int(v) for v in args.frames.split(":"))
        frames = [f for f in frames if lo <= f < hi]
    frames = frames[:: max(1, args.stride)]
    if not frames:
        raise SystemExit("no frames selected")
    num_cams, num_frames = len(session.cameras), len(frames)
    print(f"frames    : {num_frames} (stride {args.stride})")

    subpix = args.subpix_win if args.subpix_win == "auto" else int(args.subpix_win)

    corners = np.full((num_cams, num_frames, num_pts, 2), -1.0, np.float32)
    counts = np.zeros((num_cams, num_frames), np.int32)
    image_sizes = np.zeros((num_cams, 2), np.int32)
    paths = [["" for _ in frames] for _ in session.cameras]

    # CharucoDetector is not documented as thread-safe: give every worker
    # thread its own instance, keyed by camera.
    detectors = [make_detector(board) for _ in session.cameras]

    jobs = []
    for ci, cam in enumerate(session.cameras):
        for fi, frame in enumerate(frames):
            path = cam.images.get(frame)
            if path is not None:
                jobs.append((ci, fi, path))

    brightness_samples = [[] for _ in session.cameras]
    facing = [[] for _ in session.cameras]
    done = 0
    started = time.time()

    def work(job):
        ci, fi, path = job
        gray = read_gray(path, session.bayer)
        if len(brightness_samples[ci]) < 40:
            brightness_samples[ci].append(float(np.percentile(gray, 99)))
        charuco_corners, charuco_ids, _, _ = detectors[ci].detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < args.min_corners:
            return ci, fi, path, gray.shape, None, None, None
        ids = np.asarray(charuco_ids).reshape(-1)
        pts = refine(gray, np.asarray(charuco_corners).reshape(-1, 2), subpix)
        return ci, fi, path, gray.shape, ids, pts, foreshortening(object_points[ids], pts)

    with ThreadPoolExecutor(max_workers=args.workers or None) as pool:
        for ci, fi, path, shape, ids, pts, squareness in pool.map(work, jobs):
            image_sizes[ci] = (shape[1], shape[0])
            paths[ci][fi] = str(path.relative_to(session.root)).replace("\\", "/")
            if ids is not None:
                corners[ci, fi, ids] = pts
                counts[ci, fi] = len(ids)
                if squareness is not None:
                    facing[ci].append(squareness)
            done += 1
            if done % 200 == 0 or done == len(jobs):
                rate = done / max(1e-6, time.time() - started)
                print(f"  detected {done}/{len(jobs)} views ({rate:.0f}/s)", end="\r")
    print()

    # ---------------------------------------------------------------- report
    full = counts == num_pts
    usable = counts >= args.min_corners
    brightness = [float(np.median(v)) if v else float("nan") for v in brightness_samples]
    print()
    print("per camera:")
    print("  cam serial          views  detected  full board  median pts")
    for ci, cam in enumerate(session.cameras):
        seen = counts[ci][usable[ci]]
        median = int(np.median(seen)) if seen.size else 0
        print(f"  {ci:<3} {cam.serial:<12} {len(frames):>6} {usable[ci].sum():>9}"
              f" {full[ci].sum():>11} {median:>11}")

    print()
    print("co-visibility (frames where BOTH cameras see the FULL board -- this is")
    print("what the strict multi-view solve uses):")
    print("      " + "".join(f"{c.index:>7}" for c in session.cameras))
    for i in range(num_cams):
        cells = []
        for j in range(num_cams):
            cells.append("      -" if i == j else f"{int((full[i] & full[j]).sum()):>7}")
        print(f"  {i:<4}" + "".join(cells))

    print()
    print("pose diversity (how square-on the board was to each camera).")
    print("Intrinsics need views where the board FACES the camera; a view that is")
    print("always edge-on cannot pin down focal length or principal point, and it")
    print("magnifies any warp in the target.")
    print("  cam serial        median  best   frac facing (>0.55)")
    thin = []
    for ci, cam in enumerate(session.cameras):
        values = np.asarray(facing[ci])
        if not values.size:
            print(f"  {ci:<3} {cam.serial:<12}      -     -      -")
            continue
        fraction = float((values > 0.55).mean())
        flag = "" if fraction >= 0.15 else "   <- TOO EDGE-ON"
        if fraction < 0.15:
            thin.append(cam.serial)
        print(f"  {ci:<3} {cam.serial:<12} {np.median(values):6.2f} {values.max():5.2f}"
              f" {100 * fraction:9.0f}%{flag}")
    if thin:
        print()
        print(f"  {', '.join(thin)} never saw the board close to square-on.")
        print("  Re-shoot tilting the board so it points AT each camera in turn --")
        print("  for a camera mounted low, that means holding the board nearly")
        print("  upright, not flat like the platform.")

    if brightness:
        print()
        print("exposure check (99th percentile of the raw pixels):")
        for ci, cam in enumerate(session.cameras):
            level = brightness[ci]
            note = ("TOO DARK -- corner accuracy will suffer" if level < 120 else
                    "clipping" if level > 245 else "ok")
            print(f"  cam {ci} ({cam.serial}): p99 {level:5.0f}   {note}")
        if min(brightness) < 120:
            print()
            print("  These frames are underexposed. Detection may still succeed while")
            print("  sub-pixel accuracy is quietly destroyed, so fix the exposure and")
            print("  re-shoot rather than calibrating on this:")
            print("    python calibration\tune_exposure.py --auto")

    weak = [(i, j) for i in range(num_cams) for j in range(i + 1, num_cams)
            if (full[i] & full[j]).sum() < 20]
    if weak:
        print()
        print("  WARNING: these pairs share < 20 full-board frames: "
              + ", ".join(f"{i}-{j}" for i, j in weak))
        print("  The solver only needs a connected graph, but thin links mean a weak")
        print("  extrinsic. Re-shoot with the board held between those two views.")

    # ---------------------------------------------------------------- write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        corners=corners,
        counts=counts,
        frame_indices=np.asarray(frames, np.int64),
        image_sizes=image_sizes,
        object_points=spec.object_points(),
    )
    meta = {
        "schema": "charuco-detections/1",
        "session": str(session.root),
        "serials": session.serials,
        "camera_indices": [c.index for c in session.cameras],
        "bayer": session.bayer,
        "board": json.loads(args.board.read_text(encoding="utf-8")),
        "board_path": str(args.board),
        "num_corners": num_pts,
        "min_corners": args.min_corners,
        "stride": args.stride,
        "subpix_win": args.subpix_win,
        "image_paths": paths,
    }
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print()
    print(f"wrote {args.out} and {meta_path}")

    # ---------------------------------------------------------------- debug
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(0)
        for ci, cam in enumerate(session.cameras):
            options = np.flatnonzero(usable[ci])
            if not options.size:
                continue
            for fi in rng.choice(options, min(args.debug_count, options.size), replace=False):
                img = read_bgr(session.root / paths[ci][fi], session.bayer)
                ids = np.flatnonzero(corners[ci, fi, :, 0] >= 0)
                cv.aruco.drawDetectedCornersCharuco(
                    img, corners[ci, fi, ids].reshape(-1, 1, 2), ids.astype(np.int32))
                out = args.debug_dir / f"cam{ci}_{cam.serial}_frame{frames[fi]}.png"
                cv.imwrite(str(out), img)
        print(f"debug overlays -> {args.debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
