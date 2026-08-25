from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _read_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _select_records(rows: List[Dict[str, Any]], n: int, mode: str, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    if mode == "random":
        rng = random.Random(int(seed))
        return [rows[i] for i in sorted(rng.sample(range(len(rows)), int(n)))]
    return list(rows[: int(n)])


def _load_replay_base_config(sample_dir: Path, action: str, phys_dir: Path) -> tuple[Dict[str, Any], str]:
    cfg_path = sample_dir / f"auto_config_{action}.json"
    if cfg_path.is_file():
        return _load_json(cfg_path), str(cfg_path)

    template = phys_dir / "config" / f"{action}_cube_jelly.json"
    cfg = _load_json(template) if template.is_file() else {}
    run_params_path = sample_dir / "meta" / "run_parameters.json"
    if run_params_path.is_file():
        run_params = _load_json(run_params_path)
        if isinstance(run_params.get("material_params"), dict):
            cfg.update(run_params["material_params"])
        if isinstance(run_params.get("time_params"), dict):
            cfg.update(run_params["time_params"])

    bc_path = sample_dir / "meta" / "boundary_conditions.json"
    if bc_path.is_file():
        bc_meta = _load_json(bc_path)
        if isinstance(bc_meta.get("boundary_conditions"), list):
            raw_bcs = [x.get("raw") for x in bc_meta["boundary_conditions"] if isinstance(x, dict) and x.get("raw")]
            if raw_bcs:
                cfg["boundary_conditions"] = raw_bcs
    return cfg, f"fallback:{template}"


def _load_rgb_pack(sample_dir: Path) -> np.ndarray:
    pack = sample_dir / "sample_pack.npz"
    if not pack.is_file():
        raise FileNotFoundError(f"missing sample_pack: {pack}")
    data = np.load(str(pack))
    return np.asarray(data["rgb"], dtype=np.float32) / 255.0


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def _ssim_global(pred: np.ndarray, gt: np.ndarray) -> float:
    x = pred.astype(np.float64)
    y = gt.astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    mux = float(x.mean())
    muy = float(y.mean())
    vx = float(((x - mux) ** 2).mean())
    vy = float(((y - muy) ** 2).mean())
    cov = float(((x - mux) * (y - muy)).mean())
    den = (mux * mux + muy * muy + c1) * (vx + vy + c2)
    if den <= 1e-12:
        return 1.0 if float(np.mean(np.abs(x - y))) <= 1e-12 else 0.0
    return float(((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / den)


def _compare_sample(pred_dir: Path, gt_dir: Path) -> Dict[str, Any]:
    pred = _load_rgb_pack(pred_dir)
    gt = _load_rgb_pack(gt_dir)
    shape = tuple(min(a, b) for a, b in zip(pred.shape, gt.shape))
    slices = tuple(slice(0, n) for n in shape)
    pred_c = pred[slices]
    gt_c = gt[slices]
    return {
        "ssim": _ssim_global(pred_c, gt_c),
        "psnr": _psnr(pred_c, gt_c),
        "compared_shape": list(shape),
    }


def _build_pred_config(
    sample_dir: Path,
    record: Dict[str, Any],
    sample_out: Path,
    phys_dir: Path,
) -> tuple[Dict[str, Any], str, Path]:
    gt = _load_json(sample_dir / "gt.json")
    action = str(gt["action"])
    cfg, _ = _load_replay_base_config(sample_dir, action, phys_dir)

    material = str((gt.get("params") or {}).get("material") or cfg.get("material") or "jelly")
    cfg["material"] = material
    cfg["E"] = max(_as_float(record.get("E_pred_raw"), _as_float(cfg.get("E"), 1.0)), 1e-6)
    cfg["nu"] = min(max(_as_float(record.get("nu_pred_raw"), _as_float(cfg.get("nu"), 0.3)), 1e-6), 0.499)
    cfg["density"] = max(
        _as_float(record.get("density_pred_raw"), _as_float(cfg.get("density"), 1.0)),
        1e-6,
    )

    yield_pred = _as_float(record.get("yield_stress_pred_raw"), 0.0)
    if material == "metal" or "yield_stress" in cfg:
        cfg["yield_stress"] = max(yield_pred, 1e-6)

    pred_cfg = sample_out / "pred_config.json"
    _write_json(pred_cfg, cfg)
    return gt, action, pred_cfg


def _run_one(args: argparse.Namespace, idx: int, record: Dict[str, Any]) -> Dict[str, Any]:
    phys_dir = Path(args.phys_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    samples_out = Path(args.out_dir).resolve() / "replay_samples"

    sid = str(record["sample_id"])
    gt_dir = dataset_root / sid
    sample_out = samples_out / sid
    metrics_path = sample_out / "replay_metrics.json"
    if metrics_path.is_file() and not bool(args.force):
        return _load_json(metrics_path)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"missing gt sample dir: {gt_dir}")

    gt, action, pred_cfg = _build_pred_config(gt_dir, record, sample_out, phys_dir)
    ply_path = Path(str(gt.get("ply_path", "")).strip())
    if not ply_path.is_absolute():
        ply_path = phys_dir / ply_path
    if not ply_path.is_file():
        raise FileNotFoundError(f"missing ply: {ply_path}")

    run_params = _load_json(gt_dir / "meta" / "run_parameters.json")
    cmd = [
        sys.executable,
        str(phys_dir / "modified_simulation.py"),
        "--ply_path",
        str(ply_path),
        "--config",
        str(pred_cfg),
        "--output_path",
        str(sample_out),
        "--ply_flat_output",
        "--sim_type",
        action,
        "--num_views",
        str(int(run_params.get("num_views_cli", 3))),
        "--num_render_views",
        str(int(run_params.get("num_render_views_cli", -1))),
        "--num_render_timesteps",
        str(int(run_params.get("num_render_timesteps_cli", 0))),
        "--render_outputs_per_sim_second",
        str(float(run_params.get("render_outputs_per_sim_second", 16.0) or 16.0)),
        "--field_output_interval",
        str(int(run_params.get("field_output_interval", 1))),
        "--render_img",
        "--output_view_stress_gaussian",
        "--output_view_flow_gaussian",
        "--output_view_force_mask",
        "--output_view_object_mask",
        "--force_mask_single_channel",
        "--pack_sample_pack",
        "--pack_sample_pack_include_object_mask",
        "--arch4_lmdb_resize",
        "224",
        "--render_export_max_side",
        str(int(run_params.get("render_export_max_side", 512) or 512)),
        "--render_export_scale",
        str(float(run_params.get("render_export_scale", 0.5) or 0.5)),
        "--camera_distance_scale",
        str(float(run_params.get("camera_distance_scale", 1.3) or 1.3)),
        "--quiet",
    ]

    sample_out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if int(args.num_gpus) > 0:
        env["CUDA_VISIBLE_DEVICES"] = str(idx % int(args.num_gpus))
    log_path = sample_out / "simulation.log"
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("CMD: " + " ".join(cmd) + "\n")
        log_f.flush()
        subprocess.run(cmd, cwd=str(phys_dir), env=env, stdout=log_f, stderr=subprocess.STDOUT, check=True)

    metrics = _compare_sample(sample_out, gt_dir)
    out = {
        "sample_id": sid,
        "action": action,
        "material": str(record.get("material", "")),
        "object_name": str(record.get("object_name", "")),
        "gt_dir": str(gt_dir),
        "pred_replay_dir": str(sample_out),
        "pred_params": {
            "E": _as_float(record.get("E_pred_raw")),
            "nu": _as_float(record.get("nu_pred_raw")),
            "density": _as_float(record.get("density_pred_raw")),
            "yield_stress": _as_float(record.get("yield_stress_pred_raw")),
        },
        **metrics,
    }
    _write_json(metrics_path, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Replay predicted parameters in simulation and report SSIM/PSNR")
    ap.add_argument("--phys_dir", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--sample_mode", choices=("first", "random"), default="first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_records = _read_records(Path(args.records).resolve())
    records = _select_records(all_records, int(args.num_samples), str(args.sample_mode), int(args.seed))
    _write_json(
        out_root / "replay_selection.json",
        {
            "records": str(Path(args.records).resolve()),
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "num_records_available": len(all_records),
            "num_samples_requested": int(args.num_samples),
            "num_samples_used": len(records),
            "sample_mode": str(args.sample_mode),
            "seed": int(args.seed),
            "num_gpus": int(args.num_gpus),
        },
    )

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    workers = max(1, int(args.num_gpus))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, args, i, r): r for i, r in enumerate(records)}
        for fut in as_completed(futs):
            rec = futs[fut]
            sid = str(rec.get("sample_id", "unknown"))
            try:
                row = fut.result()
                rows.append(row)
                print(f"[replay OK] {sid} ssim={row['ssim']:.4f} psnr={row['psnr']:.2f}")
            except Exception as exc:
                failures.append({"sample_id": sid, "error": repr(exc)})
                print(f"[replay FAIL] {sid}: {exc}", file=sys.stderr)

    rows.sort(key=lambda r: str(r["sample_id"]))
    with (out_root / "replay_metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = ["sample_id", "action", "material", "object_name", "ssim", "psnr", "pred_replay_dir", "gt_dir"]
    with (out_root / "replay_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    ssims = [float(r["ssim"]) for r in rows if math.isfinite(float(r["ssim"]))]
    psnrs = [float(r["psnr"]) for r in rows if math.isfinite(float(r["psnr"]))]
    summary = {
        "num_success": len(rows),
        "num_failed": len(failures),
        "ssim_mean": float(np.mean(ssims)) if ssims else float("nan"),
        "ssim_median": float(np.median(ssims)) if ssims else float("nan"),
        "psnr_mean": float(np.mean(psnrs)) if psnrs else float("nan"),
        "psnr_median": float(np.median(psnrs)) if psnrs else float("nan"),
        "failures": failures,
    }
    _write_json(out_root / "replay_summary.json", summary)
    if failures:
        raise SystemExit(f"parameter replay finished with {len(failures)} failures; see replay_summary.json")


if __name__ == "__main__":
    main()
