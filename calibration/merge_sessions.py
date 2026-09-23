#!/usr/bin/env python3
"""Combine several capture sessions into one set of detections.

    python merge_sessions.py <session A> <session B> [...] --out merged.npz
    python calibrate.py merged.npz --out calibration.json --world-frames 0:12 --world-flip-z

RUN WITH THE **CALIBRATION** ENVIRONMENT (.venv-calib).

Why this exists
---------------
A session that fails on one row rarely needs redoing from scratch. Intrinsics
are a property of each camera alone -- they are fitted from that camera's own
views and never touch the extrinsics -- so a short run that adds nothing but
square-on views of the board fixes an intrinsics failure without disturbing an
extrinsic solve that already passes. The expensive part of a session is the
pair and group steps; the part that fixes intrinsics is the cheap part.

The catch is the one thing that makes merging unsound: **extrinsics are only
shared if the cameras did not move between the sessions.** That is not a matter
of opinion and it is measurable, so this refuses to merge until it has checked.
The test is the same one the session report runs, applied across the seam: the
static scene at the end of one session must line up with the static scene at
the start of the next.

    < 1 px    the rig is the same rig; merging is sound
    1-3 px    marginal -- the extrinsics will be an average of two poses
    > 3 px    refused; the sessions describe different rigs (--force overrides)

Frames are concatenated in the order the sessions are given, so `--world-frames`
still refers to the first session's world segment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2 as cv  # noqa: E402

SEAM_SOLID, SEAM_BAD = 1.0, 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", type=Path, nargs="+",
                        help="capture directories, each already holding detections.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="merge even if the rig moved between sessions")
    parser.add_argument("--bayer", default="RG")
    return parser.parse_args()


def _mask_board(gray, corners_row):
    """Blank the printed target, which moves on purpose and would dominate."""
    mask = np.full(gray.shape, 255, np.uint8)
    seen = corners_row[:, 0] >= 0
    if seen.sum() > 4:
        hull = cv.convexHull(corners_row[seen].astype(np.float32).reshape(-1, 1, 2))
        cv.fillConvexPoly(mask, hull.astype(np.int32), 0)
        mask = cv.erode(mask, np.ones((161, 161), np.uint8))
    return mask


def seam_drift(prev_session, prev_det, next_session, next_det, bayer):
    """Per-camera displacement of the static scene across the join.

    Late frames of one session against early frames of the next, medianed, so a
    single frame in which the board escaped the mask cannot decide the verdict.
    """
    orb = cv.ORB_create(6000)
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    code = getattr(cv, f"COLOR_Bayer{bayer}2GRAY")
    out = {}

    for ci, serial in enumerate(prev_det["meta"]["serials"]):
        def load(session, det, frame):
            path = Path(session) / det["meta"]["image_paths"][ci][frame]
            raw = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            if raw is None:
                return None
            gray = (cv.cvtColor(raw, code) if raw.ndim == 2
                    else cv.cvtColor(raw, cv.COLOR_BGR2GRAY))
            return gray, _mask_board(gray, det["corners"][ci, frame])

        def usable(det):
            return [f for f in range(det["corners"].shape[1])
                    if (det["corners"][ci, f, :, 0] >= 0).sum() >= 30]

        before, after = usable(prev_det), usable(next_det)
        if not before or not after:
            out[serial] = None
            continue
        values = []
        for tail in before[-4:]:
            first = load(prev_session, prev_det, tail)
            if first is None:
                continue
            for head in after[:4]:
                second = load(next_session, next_det, head)
                if second is None:
                    continue
                value = _displacement(first, second, orb, matcher)
                if value is not None:
                    values.append(value)
        out[serial] = float(np.median(values)) if values else None
    return out


def _displacement(first, second, orb, matcher):
    ka, da = orb.detectAndCompute(first[0], first[1])
    kb, db = orb.detectAndCompute(second[0], second[1])
    if da is None or db is None or len(da) < 40 or len(db) < 40:
        return None
    matches = sorted(matcher.match(da, db), key=lambda m: m.distance)[:800]
    if len(matches) < 40:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, inliers = cv.findHomography(src, dst, cv.RANSAC, 3.0)
    if inliers is None or inliers.sum() < 25:
        return None
    keep = inliers.ravel().astype(bool)
    return float(np.median(np.linalg.norm(
        dst[keep].reshape(-1, 2) - src[keep].reshape(-1, 2), axis=1)))


def load_session(path: Path):
    npz = path / "detections.npz"
    meta_path = path / "detections.json"
    if not npz.exists() or not meta_path.exists():
        raise SystemExit(f"{path} has no detections.npz -- run detect_corners.py on it first")
    data = np.load(npz, allow_pickle=True)
    return {"corners": data["corners"], "counts": data["counts"],
            "frame_indices": data["frame_indices"], "image_sizes": data["image_sizes"],
            "object_points": data["object_points"],
            "meta": json.loads(meta_path.read_text(encoding="utf-8"))}


def main() -> int:
    args = parse_args()
    loaded = [(path, load_session(path)) for path in args.sessions]

    reference = loaded[0][1]
    for path, det in loaded[1:]:
        if det["meta"]["serials"] != reference["meta"]["serials"]:
            raise SystemExit(
                f"{path} has cameras {det['meta']['serials']}, not "
                f"{reference['meta']['serials']}. Merging would mix up the cameras.")
        if det["object_points"].shape != reference["object_points"].shape or \
                not np.allclose(det["object_points"], reference["object_points"]):
            raise SystemExit(f"{path} used a different board. Merging is meaningless.")

    for path, det in loaded:
        print(f"{path.name}: {det['corners'].shape[1]} frames, "
              f"{int((det['corners'][:, :, :, 0] >= 0).sum())} corners")

    moved = []
    for (prev_path, prev_det), (next_path, next_det) in zip(loaded, loaded[1:]):
        print(f"\nseam {prev_path.name} -> {next_path.name}: "
              f"did the rig stay put between them?")
        drift = seam_drift(prev_path, prev_det, next_path, next_det, args.bayer)
        for serial, value in drift.items():
            if value is None:
                print(f"  {serial}      -- (could not measure)")
                continue
            tag = ("same rig" if value < SEAM_SOLID else
                   "marginal" if value < SEAM_BAD else "MOVED")
            print(f"  {serial} {value:7.2f} px   {tag}")
            if value >= SEAM_BAD:
                moved.append(f"{serial} ({value:.1f} px)")

    if moved and not args.force:
        print(f"\nrefusing to merge: {', '.join(moved)} moved between sessions.")
        print("The extrinsics of the two sessions describe different rigs, and a")
        print("merged solve would silently average them. Re-shoot instead, or pass")
        print("--force if you only want the merge for INTRINSICS and will discard")
        print("the extrinsics.")
        return 1

    corners = np.concatenate([det["corners"] for _, det in loaded], axis=1)
    counts = np.concatenate([det["counts"] for _, det in loaded], axis=1)
    # Frame indices have to stay unique across the join: `--world-frames` and the
    # hold-out split both address frames by this number.
    indices, offset = [], 0
    for _, det in loaded:
        indices.append(np.asarray(det["frame_indices"]) + offset)
        offset += int(np.max(det["frame_indices"])) + 1
    frame_indices = np.concatenate(indices)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, corners=corners, counts=counts, frame_indices=frame_indices,
        image_sizes=reference["image_sizes"], object_points=reference["object_points"])

    merged_meta = dict(reference["meta"])
    merged_meta["session"] = [str(path.resolve()) for path, _ in loaded]
    # Absolute here, because the paths in each session are relative to their own
    # root and there is no single root any more.
    merged_meta["image_paths"] = [
        [str((path / relative).resolve())
         for path, det in loaded
         for relative in det["meta"]["image_paths"][ci]]
        for ci in range(len(reference["meta"]["serials"]))]
    args.out.with_suffix(".json").write_text(json.dumps(merged_meta, indent=2),
                                             encoding="utf-8")

    total = int((corners[:, :, :, 0] >= 0).sum())
    print(f"\nmerged: {corners.shape[1]} frames, {total} corners -> {args.out}")
    world = loaded[0][1]["meta"].get("world_frames")
    print("world frames still come from the first session; pass the same "
          f"--world-frames you would have used for {loaded[0][0].name}"
          + (f" ({world[0]}:{world[1]})" if world else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
