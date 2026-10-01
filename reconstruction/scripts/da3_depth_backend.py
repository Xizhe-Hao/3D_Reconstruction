"""Run calibrated DA3 in an isolated Python environment."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np


def estimate_depths_with_da3(args, clip, run_dir, report):
    root = Path(__file__).resolve().parents[1]
    output = run_dir / "da3"
    output.mkdir(parents=True, exist_ok=True)
    model = args.da3_model.expanduser().resolve()
    job = {"version": 1, "model": str(model), "process_res": args.da3_process_res,
           "confidence_percentile": args.da3_confidence_percentile, "device": args.device,
           "source_commit": subprocess.check_output(["git", "-C", str(root / "submodule/depth-anything-3"), "rev-parse", "HEAD"], text=True).strip()}
    digest = hashlib.sha256(json.dumps(job, sort_keys=True).encode())
    for array in (clip.rgbs, clip.intrinsics, clip.extrinsics_w2c_m, clip.frame_indices):
        digest.update(np.ascontiguousarray(array).tobytes())
    for name in ("config.json", "model.safetensors"):
        stat = (model / name).stat()
        digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    job["fingerprint"] = digest.hexdigest()
    manifest = output / "manifest.json"
    cache = output / "depths.npz"
    if (args.recompute_depth or not cache.is_file() or not manifest.is_file()
            or json.loads(manifest.read_text()).get("fingerprint") != job["fingerprint"]):
        np.savez(output / "input.npz", rgbs=clip.rgbs, intrinsics=clip.intrinsics,
                 extrinsics=clip.extrinsics_w2c_m, frame_indices=clip.frame_indices)
        job["recompute"] = bool(args.recompute_depth)
        job["output"] = str(output)
        (output / "job.json").write_text(json.dumps(job, indent=2))
        report("3/8 depth", "DA3 giant: calibrated four-view inference per timestamp", not args.no_progress)
        subprocess.run([str(args.da3_python.expanduser().absolute()), str(root / "scripts/da3_worker.py"),
                        "--job", str(output / "job.json")], check=True, cwd=root)
    with np.load(cache) as data:
        depths = data["depths_m"].copy()
    if depths.shape != clip.rgbs.shape[:2] + clip.rgbs.shape[-2:]:
        raise ValueError(f"DA3 cache shape mismatch: {depths.shape}")
    return depths
