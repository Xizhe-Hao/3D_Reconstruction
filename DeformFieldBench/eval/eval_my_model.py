from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

from eval_abalation import paths
from eval_abalation.dataset_meta import load_metadata_map
from eval_abalation.field_export import build_field_export_payload, export_field_tensors_pt, export_field_videos
from eval_abalation.metrics_field import aggregate_field_records, align_field_to_prediction, build_field_sample_record
from eval_abalation.metrics_param import (
    aggregate_param_records,
    build_param_sample_record,
    enrich_param_records_with_composite,
)
from eval_abalation.report_utils import write_csv, write_json, write_jsonl
from eval_abalation.split_utils import load_split_info, pick_eval_ids
from my_model.arch4_model import Arch4VideoMAEPhysModel, build_arch4_model
from my_model.dataset import DatasetArch4, resolve_flat_dataset_root


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate one my_model checkpoint with grouped metrics")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--eval_split", type=str, choices=("train", "test"), default="test")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--num_samples", type=int, default=0)
    ap.add_argument("--sample_mode", type=str, choices=("random", "first"), default="first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--run_visuals", action="store_true")
    ap.add_argument("--num_vis", type=int, default=8)
    ap.add_argument("--vis_view", type=int, default=0)
    ap.add_argument("--export_field_tensors", action="store_true")
    ap.add_argument("--field_export_dir", type=str, default="")
    ap.add_argument("--field_export_format", type=str, choices=("pt",), default="pt")
    ap.add_argument("--export_field_videos", action="store_true")
    ap.add_argument("--field_video_dir", type=str, default="")
    ap.add_argument("--field_video_fps", type=int, default=8)
    ap.add_argument("--field_video_view", type=int, default=0)
    ap.add_argument("--field_video_color_mode", type=str, choices=("auto", "rgb", "jet"), default="auto")
    return ap.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config 顶层必须是 object: {path}")
    return raw


def _resolve_device(device_s: str) -> torch.device:
    dev = str(device_s).strip().lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev in ("cuda", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _select_sample_ids(sample_ids: Sequence[str], n_req: int, sample_mode: str, seed: int) -> List[str]:
    ids = list(sample_ids)
    if n_req <= 0 or n_req >= len(ids):
        return ids
    if sample_mode == "first":
        return ids[:n_req]
    rng = random.Random(int(seed))
    return [ids[i] for i in sorted(rng.sample(range(len(ids)), n_req))]


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if all(isinstance(k, str) for k in ckpt.keys()) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore[return-value]
    raise ValueError("无法从 checkpoint 中提取 state_dict")


def _build_model(cfg_model: Dict[str, Any]) -> Arch4VideoMAEPhysModel:
    sig = signature(Arch4VideoMAEPhysModel.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    model_kwargs = {k: v for k, v in cfg_model.items() if k in accepted}
    return build_arch4_model(**model_kwargs)


def _run_visual_export(args: argparse.Namespace, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "my_model.eval",
        "--config",
        str(args.config),
        "--weights",
        str(args.weights),
        "--sample_ids_json",
        str(args._split_json_path),
        "--sample_ids_key",
        str("test_ids" if args.eval_split == "test" else "train_ids"),
        "--sample_limit",
        str(args.num_samples),
        "--out_dir",
        str(out_dir / "visuals"),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--num_vis",
        str(args.num_vis),
        "--vis_view",
        str(args.vis_view),
    ]
    subprocess.run(cmd, cwd=str(paths.PHYS_ROOT), check=True)


def _aggregate_and_write(rows: List[Dict[str, object]], out_dir: Path, *, kind: str) -> Dict[str, Any]:
    target = out_dir / f"{kind}_metrics"
    if kind == "param":
        rows, weights, composite_rows, composite_summary = enrich_param_records_with_composite(rows)
        write_jsonl(target / "sample_records.jsonl", rows)
        write_jsonl(target / "sample_composite_errors.jsonl", composite_rows)
        write_csv(target / "sample_composite_errors.csv", composite_rows)
        write_json(target / "param_error_weights.json", weights)
        write_csv(target / "param_error_weights.csv", weights)
        overall = aggregate_param_records(rows, group_fields=())
        by_action = aggregate_param_records(rows, group_fields=("action",))
        by_material = aggregate_param_records(rows, group_fields=("material",))
        by_action_material = aggregate_param_records(rows, group_fields=("action", "material"))
    else:
        write_jsonl(target / "sample_records.jsonl", rows)
        overall = aggregate_field_records(rows, group_fields=())
        by_action = aggregate_field_records(rows, group_fields=("action",))
        by_material = aggregate_field_records(rows, group_fields=("material",))
        by_action_material = aggregate_field_records(rows, group_fields=("action", "material"))
    for name, data in (
        ("overall", overall),
        ("by_action", by_action),
        ("by_material", by_material),
        ("by_action_material", by_action_material),
    ):
        write_json(target / f"{name}.json", data)
        write_csv(target / f"{name}.csv", data)
    result = {
        "overall": overall,
        "by_action": by_action,
        "by_material": by_material,
        "by_action_material": by_action_material,
    }
    if kind == "param":
        result.update(
            {
                "param_error_weights": weights,
                "sample_composite_errors": composite_rows,
                "composite_summary": composite_summary,
            }
        )
    return result


def _mean_field_metric(rows: Sequence[Dict[str, object]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_json(cfg_path)
    cfg_data = cfg.get("data") or {}
    cfg_model = cfg.get("model") or {}

    split_json_rel = cfg.get("train", {}).get("train_ids_json") or cfg_data.get("train_ids_json") or "train_test_split.json"
    split_json_path = Path(str(split_json_rel))
    if not split_json_path.is_absolute():
        split_json_path = (cfg_path.parents[1] / split_json_path).resolve()
    args._split_json_path = split_json_path  # type: ignore[attr-defined]
    split_info = load_split_info(split_json_path)
    eval_ids = pick_eval_ids(split_info, args.eval_split)
    chosen_ids = _select_sample_ids(eval_ids, int(args.num_samples), str(args.sample_mode), int(args.seed))
    split_root_cfg = str(cfg_data.get("split_root") or "")
    auto_output = str(cfg_data.get("auto_output", "auto_output"))
    dataset_root = resolve_flat_dataset_root(split_root_cfg, paths.PHYS_ROOT / auto_output)

    ds = DatasetArch4(
        dataset_root,
        img_size=int(cfg_model.get("img_size", 224)),
        max_views=int(cfg_model.get("num_views", 3)),
        num_frames=int(cfg_model.get("num_frames", 16)),
        source="auto",
        preflight=False,
        verbose=False,
        sample_ids=chosen_ids,
        input_mode="images",
        return_sample_id=True,
        return_object_mask=True,
    )
    actual_ids = [str(p.name) for p in ds.samples]
    meta_map = load_metadata_map(split_info.split_root, actual_ids)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    weights_path = Path(args.weights).expanduser().resolve()
    try:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(weights_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"invalid checkpoint: {weights_path}")
    model = _build_model(cfg_model)
    model.load_state_dict(_extract_state_dict(ckpt), strict=False)
    device = _resolve_device(args.device)
    model.to(device)
    model.eval()

    param_rows: List[Dict[str, object]] = []
    field_rows: List[Dict[str, object]] = []
    field_export_dir = (
        Path(args.field_export_dir).expanduser().resolve()
        if str(args.field_export_dir).strip()
        else (Path(args.out_dir).expanduser().resolve() / "field_exports")
    )
    field_video_dir = (
        Path(args.field_video_dir).expanduser().resolve()
        if str(args.field_video_dir).strip()
        else (Path(args.out_dir).expanduser().resolve() / "field_videos")
    )

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 7:
                x, stress_gt, flow_gt, force_gt, object_gt, params_gt, sample_ids = batch
            elif len(batch) == 6:
                x, stress_gt, flow_gt, force_gt, object_gt, params_gt = batch
                sample_ids = ["unknown"] * int(x.shape[0])
            else:
                x, stress_gt, flow_gt, force_gt, params_gt = batch
                object_gt = torch.ones_like(force_gt)
                sample_ids = ["unknown"] * int(x.shape[0])

            x = x.to(device)
            stress_gt = stress_gt.to(device)
            flow_gt = flow_gt.to(device)
            force_gt = force_gt.to(device)
            object_gt = object_gt.to(device)
            params_gt = params_gt.to(device)

            out = model(x)
            pred_raw = out["param_pred_raw"].detach().cpu().numpy()
            gt_raw = params_gt.detach().cpu().numpy()

            stress_pred = out["stress_field_pred"]
            flow_pred = out["flow_field_pred"]
            force_pred = out["force_pred"]
            stress_tgt = align_field_to_prediction(stress_gt, stress_pred)
            flow_tgt = align_field_to_prediction(flow_gt, flow_pred)
            force_tgt = align_field_to_prediction(force_gt, force_pred)

            for i, sid in enumerate([str(s) for s in sample_ids]):
                meta = meta_map.get(sid)
                action = meta.action if meta is not None else "unknown"
                material = meta.material_group if meta is not None else "unknown"
                object_name = meta.object_name if meta is not None else "unknown"
                stress_pred_cpu = stress_pred[i : i + 1].detach().cpu()
                flow_pred_cpu = flow_pred[i : i + 1].detach().cpu()
                force_pred_cpu = force_pred[i : i + 1].detach().cpu()
                stress_tgt_cpu = stress_tgt[i : i + 1].detach().cpu()
                flow_tgt_cpu = flow_tgt[i : i + 1].detach().cpu()
                force_tgt_cpu = force_tgt[i : i + 1].detach().cpu()
                object_gt_cpu = object_gt[i : i + 1].detach().cpu()
                param_rows.append(
                    build_param_sample_record(
                        sample_id=sid,
                        action=action,
                        material=material,
                        object_name=object_name,
                        gt_raw=gt_raw[i].tolist(),
                        pred_raw=pred_raw[i].tolist(),
                    )
                )
                field_rows.append(
                    build_field_sample_record(
                        sample_id=sid,
                        action=action,
                        material=material,
                        object_name=object_name,
                        field_name="stress",
                        pred_bvcthw=stress_pred_cpu,
                        gt_bvcthw=stress_tgt_cpu,
                        object_mask_bvcthw=object_gt_cpu,
                    )
                )
                field_rows.append(
                    build_field_sample_record(
                        sample_id=sid,
                        action=action,
                        material=material,
                        object_name=object_name,
                        field_name="flow",
                        pred_bvcthw=flow_pred_cpu,
                        gt_bvcthw=flow_tgt_cpu,
                        object_mask_bvcthw=object_gt_cpu,
                    )
                )
                field_rows.append(
                    build_field_sample_record(
                        sample_id=sid,
                        action=action,
                        material=material,
                        object_name=object_name,
                        field_name="force_mask",
                        pred_bvcthw=force_pred_cpu,
                        gt_bvcthw=force_tgt_cpu,
                        object_mask_bvcthw=object_gt_cpu,
                    )
                )
                if args.export_field_tensors:
                    payload = build_field_export_payload(
                        sample_id=sid,
                        stress_pred=stress_pred_cpu,
                        stress_gt=stress_tgt_cpu,
                        flow_pred=flow_pred_cpu,
                        flow_gt=flow_tgt_cpu,
                        force_pred=force_pred_cpu,
                        force_gt=force_tgt_cpu,
                        object_mask=object_gt_cpu,
                        meta={
                            "checkpoint": str(weights_path),
                            "model_name": "arch4",
                            "eval_split": str(args.eval_split),
                            "action": action,
                            "material": material,
                            "object_name": object_name,
                        },
                    )
                    export_field_tensors_pt(
                        root_dir=field_export_dir,
                        sample_id=sid,
                        payload=payload,
                    )
                if args.export_field_videos:
                    export_field_videos(
                        root_dir=field_video_dir,
                        sample_id=sid,
                        stress_pred=stress_pred_cpu,
                        stress_gt=stress_tgt_cpu,
                        flow_pred=flow_pred_cpu,
                        flow_gt=flow_tgt_cpu,
                        force_pred=force_pred_cpu,
                        force_gt=force_tgt_cpu,
                        view_idx=int(args.field_video_view),
                        fps=int(args.field_video_fps),
                        color_mode=str(args.field_video_color_mode),
                    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    param_metrics = _aggregate_and_write(param_rows, out_dir, kind="param")
    field_metrics = _aggregate_and_write(field_rows, out_dir, kind="field")

    param_maes = [float(r["mae"]) for r in param_metrics["overall"]]
    param_gt_trim_maes = [float(r["gt_trim_mae"]) for r in param_metrics["overall"]]
    param_err_trim_maes = [float(r["err_trim_mae"]) for r in param_metrics["overall"]]
    field_mses = [float(r["mse"]) for r in field_metrics["overall"]]
    field_ssims = [float(r["ssim"]) for r in field_metrics["overall"]]
    composite_summary = param_metrics.get("composite_summary", {})
    score = {
        "param_mean_mae": float(sum(param_maes) / max(len(param_maes), 1)),
        "param_mean_gt_trim_mae": float(sum(param_gt_trim_maes) / max(len(param_gt_trim_maes), 1)),
        "param_mean_err_trim_mae": float(sum(param_err_trim_maes) / max(len(param_err_trim_maes), 1)),
        "param_mean_composite_error": composite_summary.get("mean"),
        "param_median_composite_error": composite_summary.get("median"),
        "param_p90_composite_error": composite_summary.get("p90"),
        "field_mean_mse": float(sum(field_mses) / max(len(field_mses), 1)),
        "field_mean_ssim": float(sum(field_ssims) / max(len(field_ssims), 1)),
        "field_mean_region_iou": _mean_field_metric(field_metrics["overall"], "region_iou"),
        "field_mean_soft_region_iou": _mean_field_metric(field_metrics["overall"], "soft_region_iou"),
        "field_mean_region_recall": _mean_field_metric(field_metrics["overall"], "region_recall"),
        "flow_mean_coverage_iou": _mean_field_metric(field_metrics["overall"], "flow_coverage_iou"),
        "flow_mean_coverage_recall": _mean_field_metric(field_metrics["overall"], "flow_coverage_recall"),
        "flow_mean_velocity_weighted_overlap": _mean_field_metric(field_metrics["overall"], "flow_velocity_weighted_overlap"),
        "flow_mean_velocity_similarity_overlap": _mean_field_metric(
            field_metrics["overall"], "flow_velocity_similarity_overlap"
        ),
        "flow_mean_velocity_weighted_epe": _mean_field_metric(field_metrics["overall"], "flow_velocity_weighted_epe"),
        "flow_mean_motion_quality": _mean_field_metric(field_metrics["overall"], "flow_motion_quality"),
        "force_mean_main_recall": _mean_field_metric(field_metrics["overall"], "force_main_recall"),
        "force_mean_centroid_dist": _mean_field_metric(field_metrics["overall"], "force_centroid_dist"),
        "stress_mean_hotspot_recall": _mean_field_metric(field_metrics["overall"], "stress_hotspot_recall"),
        "stress_mean_weighted_mae": _mean_field_metric(field_metrics["overall"], "stress_weighted_mae"),
    }
    meta = {
        "config": str(cfg_path),
        "weights": str(weights_path),
        "eval_split": str(args.eval_split),
        "num_samples_requested": int(args.num_samples),
        "num_samples_used": len(actual_ids),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "run_visuals": bool(args.run_visuals),
        "export_field_tensors": bool(args.export_field_tensors),
        "field_export_dir": str(field_export_dir) if args.export_field_tensors else "",
        "field_export_format": str(args.field_export_format),
        "export_field_videos": bool(args.export_field_videos),
        "field_video_dir": str(field_video_dir) if args.export_field_videos else "",
        "field_video_fps": int(args.field_video_fps),
        "field_video_view": int(args.field_video_view),
        "field_video_color_mode": str(args.field_video_color_mode),
        "field_metric_version": "rgb_semantic_v3_flow_uv_quality",
        "field_metrics_masked_by_object": True,
        "field_export_includes_object_mask": bool(args.export_field_tensors),
        "chosen_sample_ids": actual_ids,
    }
    write_json(out_dir / "meta.json", meta)
    write_json(out_dir / "score.json", score)
    write_json(
        out_dir / "summary.json",
        {
            "meta": meta,
            "score": score,
            "param_overall": param_metrics["overall"],
            "param_composite_summary": param_metrics["composite_summary"],
            "param_error_weights": param_metrics["param_error_weights"],
            "field_overall": field_metrics["overall"],
        },
    )
    if args.run_visuals:
        _run_visual_export(args, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "score": score}, ensure_ascii=False))


if __name__ == "__main__":
    main()
