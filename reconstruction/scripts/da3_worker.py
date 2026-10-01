"""Calibrated, timestamp-wise DA3 inference; no cross-time rigid-scene assumption."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    output = Path(job["output"])
    with np.load(output / "input.npz") as data:
        rgbs, intrinsics, extrinsics, frames = [data[k].copy() for k in ("rgbs", "intrinsics", "extrinsics", "frame_indices")]
    V, T, _, H, W = rgbs.shape
    model = None
    depths = np.empty((V, T, H, W), np.float32)
    stats = []
    for t, frame in enumerate(frames):
        path = output / f"frame_{int(frame):05d}.npz"
        cached = False
        if path.is_file() and not job.get("recompute", False):
            with np.load(path) as previous:
                if str(previous["fingerprint"]) == job["fingerprint"]:
                    depths[:, t] = previous["depths_m"]
                    cached = True
        if not cached:
            if model is None:
                model = DepthAnything3.from_pretrained(job["model"]).to(job["device"]).eval()
            start = time.perf_counter()
            exts = np.repeat(np.eye(4, dtype=np.float32)[None], V, axis=0)
            exts[:, :3] = extrinsics[:, t]
            prediction = model.inference([rgb.transpose(1, 2, 0) for rgb in rgbs[:, t]],
                extrinsics=exts, intrinsics=intrinsics[:, t].copy(),
                align_to_input_ext_scale=True, process_res=job["process_res"],
                process_res_method="upper_bound_resize")
            raw = np.asarray(prediction.depth, np.float32)
            if raw.shape[0] != V or not np.isfinite(raw).all() or not (raw > 0).any():
                raise ValueError("DA3 produced invalid depths")
            expected_k = intrinsics[:, t].copy()
            expected_k[:, 0] *= raw.shape[2] / W
            expected_k[:, 1] *= raw.shape[1] / H
            if not np.allclose(prediction.intrinsics, expected_k, rtol=1e-4, atol=1e-3):
                raise ValueError("DA3 preprocessing changed the calibrated field of view")
            if not np.allclose(prediction.extrinsics, extrinsics[:, t], atol=1e-5):
                raise ValueError("DA3 did not preserve input camera extrinsics")
            filtered = raw.copy()
            conf = np.asarray(prediction.conf, np.float32)
            for v in range(V):
                valid = np.isfinite(conf[v]) & (raw[v] > 0)
                if not valid.any():
                    raise ValueError(f"DA3 has no confident pixels for view {v}")
                threshold = np.percentile(conf[v][valid], job["confidence_percentile"])
                filtered[v][~valid | (conf[v] < threshold)] = 0
                depths[v, t] = cv2.resize(filtered[v], (W, H), interpolation=cv2.INTER_NEAREST)
            # Resize-only preprocessing keeps the original field of view and calibration.
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, depths_m=depths[:, t], raw_depths_m=raw,
                confidence=conf, processed_intrinsics=prediction.intrinsics,
                fingerprint=job["fingerprint"])
            os.replace(temporary, path)
            print(f"DA3 frame {int(frame)} ({t+1}/{T}): {time.perf_counter()-start:.2f}s, median={np.median(raw):.4f} m", flush=True)
        stats.append({"frame": int(frame), "valid_fraction": float((depths[:, t] > 0).mean()),
                      "median_depth_m": float(np.median(depths[:, t][depths[:, t] > 0]))})
    temporary = output / "depths.tmp.npz"
    np.savez_compressed(temporary, depths_m=depths, frame_indices=frames)
    os.replace(temporary, output / "depths.npz")
    job["stats"] = stats
    job["peak_cuda_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3 if job["device"].startswith("cuda") else 0
    (output / "manifest.json").write_text(json.dumps(job, indent=2))


if __name__ == "__main__":
    main()
