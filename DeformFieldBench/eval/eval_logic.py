from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from eval_abalation.dataset_meta import SampleMeta, load_metadata_map
from eval_abalation.field_export import build_field_export_payload, export_field_tensors_pt, export_field_videos
from eval_abalation.metrics_field import aggregate_field_records, build_field_sample_record
from eval_abalation.metrics_param import (
    aggregate_param_records,
    build_param_sample_record,
    enrich_param_records_with_composite,
)
from eval_abalation.report_utils import write_csv, write_json, write_jsonl
from eval_abalation.split_utils import load_split_info, pick_eval_ids
from logic_model.dataset import LmdbGtDataset, collate_lmdb_gt_batch
from logic_model.eval_visual import _load_config, _pick, _resample_time_bvcthw, _resolve_path_relative_to_config
from logic_model.model import LogicPhysModel
from logic_model.model2 import LogicPhysModel2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate one logic_model checkpoint and export grouped metrics")
    ap.add_argument("--config", type=str, required=True, help="logic_model 训练配置 json")
    ap.add_argument("--checkpoint", type=str, required=True, help="待评估 checkpoint")
    ap.add_argument("--eval_split", type=str, choices=("train", "test"), default="test")
    ap.add_argument("--out_dir", type=str, required=True, help="评估输出目录")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--num_samples", type=int, default=0, help="0=全量")
    ap.add_argument("--sample_mode", type=str, choices=("random", "first"), default="first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--run_visuals", action="store_true", help="额外调用 logic_model.eval_visual 生成散点图/视频")
    ap.add_argument("--save_field_videos", action="store_true", help="与 --run_visuals 联用")
    ap.add_argument("--max_field_video_samples", type=int, default=8)
    ap.add_argument("--field_video_color_mode", type=str, choices=("auto", "rgb", "jet"), default="auto")
    ap.add_argument("--export_field_tensors", action="store_true")
    ap.add_argument("--field_export_dir", type=str, default="")
    ap.add_argument("--field_export_format", type=str, choices=("pt",), default="pt")
    ap.add_argument("--export_field_videos", action="store_true")
    ap.add_argument("--field_video_dir", type=str, default="")
    ap.add_argument("--field_video_fps", type=int, default=8)
    ap.add_argument("--field_video_view", type=int, default=0)
    return ap.parse_args()


def _select_indices(n_total: int, n_req: int, mode: str, seed: int) -> List[int]:
    if n_req <= 0 or n_req >= n_total:
        return list(range(n_total))
    if mode == "first":
        return list(range(n_req))
    rng = random.Random(int(seed))
    return sorted(rng.sample(range(n_total), n_req))


def _resolve_device(device_s: str) -> torch.device:
    dev = str(device_s).strip().lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev in ("cuda", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    if not dist.is_available():
        raise RuntimeError("torch.distributed 不可用")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return True, rank, local_rank, world_size


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def _build_eval_dataset(
    *,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[LmdbGtDataset, LmdbGtDataset | Subset, List[str], Dict[str, SampleMeta]]:
    cfg_data = cfg.get("data") or {}
    cfg_model = cfg.get("model") or {}
    split_json_rel = cfg_data.get("train_ids_json")
    if not split_json_rel:
        raise ValueError("config.data.train_ids_json 不能为空")
    split_json_path = _resolve_path_relative_to_config(str(split_json_rel), args.config)
    split_info = load_split_info(split_json_path)
    eval_ids = pick_eval_ids(split_info, args.eval_split)

    indices = _select_indices(len(eval_ids), int(args.num_samples), str(args.sample_mode), int(args.seed))
    chosen_ids = [eval_ids[i] for i in indices]
    meta_map = load_metadata_map(split_info.split_root, chosen_ids)

    split_root = str(_pick(cfg_data, "split_root", split_info.split_root))
    lmdb_env_subdir = str(_pick(cfg_data, "lmdb_env_subdir", "arch4_data.lmdb") or "arch4_data.lmdb").strip()
    sample_storage_backend = str(_pick(cfg_data, "sample_storage_backend", "auto") or "auto").strip()
    sample_pack_name = str(_pick(cfg_data, "sample_pack_name", "sample_pack.npz") or "sample_pack.npz").strip()
    max_views = int(_pick(cfg_model, "num_views", 3))
    num_frames = int(_pick(cfg_model, "num_frames", 0))
    img_size = int(_pick(cfg_model, "img_size", 0))
    purify_force_mask_on_read = bool(_pick(cfg_data, "purify_force_mask_on_read", False))
    force_mask_purify_mode = str(_pick(cfg_data, "force_mask_purify_mode", "red_minus_others") or "red_minus_others")
    force_mask_keep_three_channels = bool(_pick(cfg_data, "force_mask_keep_three_channels", True))
    force_mask_single_channel = bool(_pick(cfg_data, "force_mask_single_channel", False))
    force_mask_binary_threshold = float(_pick(cfg_data, "force_mask_binary_threshold", 0.5))

    ds = LmdbGtDataset(
        split_root=str(split_root),
        lmdb_env_subdir=lmdb_env_subdir,
        sample_pack_name=sample_pack_name,
        sample_storage_backend=sample_storage_backend,
        sample_pack_deep_validate=False,
        max_views=max_views,
        num_frames=(None if num_frames <= 0 else num_frames),
        img_size=(None if img_size <= 0 else img_size),
        return_action_name=True,
        sample_ids=chosen_ids,
        purify_force_mask_on_read=purify_force_mask_on_read,
        force_mask_purify_mode=force_mask_purify_mode,
        force_mask_keep_three_channels=force_mask_keep_three_channels,
        force_mask_single_channel=force_mask_single_channel,
        force_mask_binary_threshold=force_mask_binary_threshold,
    )
    return ds, ds, chosen_ids, meta_map


def _build_model_from_checkpoint(
    *,
    cfg: Dict[str, Any],
    ckpt: Dict[str, Any],
    ds: LmdbGtDataset,
) -> torch.nn.Module:
    cfg_model = cfg.get("model") or {}
    mh = ckpt.get("model_hparams") if isinstance(ckpt.get("model_hparams"), dict) else {}
    max_views = int(_pick(cfg_model, "num_views", 3))
    img_size = int(_pick(cfg_model, "img_size", 224))
    dec_h = int(_pick(cfg_model, "dec_h", 112))
    dec_w = int(_pick(cfg_model, "dec_w", 112))

    model_arch = str(mh.get("arch", _pick(cfg_model, "arch", "logic_v1"))).strip().lower()
    num_frames_model = int(mh.get("num_frames", _pick(cfg_model, "num_frames", 0)) or 0)
    if num_frames_model <= 0:
        sample0 = ds[0]
        num_frames_model = int(sample0["rgb"].shape[2])

    if model_arch in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
        model = LogicPhysModel2(
            num_views=int(mh.get("num_views", max_views)),
            in_channels=int(mh.get("in_channels", _pick(cfg_model, "in_channels", 3))),
            num_frames=int(mh.get("num_frames", num_frames_model)),
            img_size=int(_pick(cfg_model, "img_size", img_size if img_size > 0 else 224)),
            num_targets=4,
            num_actions=ds.num_actions,
            dec_h=int(mh.get("dec_h", dec_h)),
            dec_w=int(mh.get("dec_w", dec_w)),
            fusion_dim=int(_pick(cfg_model, "fusion_dim", 512)),
            fusion_heads=int(_pick(cfg_model, "fusion_heads", 8)),
            head_dropout=float(_pick(cfg_model, "head_dropout", 0.1)),
            use_uncertainty=bool(_pick(cfg_model, "use_uncertainty", False)),
            bottleneck_dim=int(_pick(cfg_model, "bottleneck_dim", 128)),
            dino_backbone_name=str(mh.get("dino_backbone_name", _pick(cfg_model, "dino_backbone_name", "dinov2_vits14"))),
            dino_backbone_pretrained=bool(mh.get("dino_backbone_pretrained", _pick(cfg_model, "dino_backbone_pretrained", True))),
            dino_backbone_source=str(mh.get("dino_backbone_source", _pick(cfg_model, "dino_backbone_source", "torchhub"))),
            dino_out_dim=int(mh.get("dino_out_dim", _pick(cfg_model, "dino_out_dim", 384))),
            temporal_adapter_type=str(mh.get("temporal_adapter_type", _pick(cfg_model, "temporal_adapter_type", "transformer"))),
            temporal_adapter_layers=int(mh.get("temporal_adapter_layers", _pick(cfg_model, "temporal_adapter_layers", 2))),
            temporal_adapter_heads=int(mh.get("temporal_adapter_heads", _pick(cfg_model, "temporal_adapter_heads", 6))),
            temporal_adapter_dropout=float(mh.get("temporal_adapter_dropout", _pick(cfg_model, "temporal_adapter_dropout", 0.1))),
            frame_pool=str(mh.get("frame_pool", _pick(cfg_model, "frame_pool", "mean"))),
            freeze_backbone=bool(mh.get("freeze_backbone", _pick(cfg_model, "freeze_backbone", True))),
            torchhub_dir=str(mh.get("torchhub_dir", _pick(cfg_model, "torchhub_dir", "")) or ""),
            dino_torchhub_repo=str(mh.get("dino_torchhub_repo", _pick(cfg_model, "dino_torchhub_repo", "facebookresearch/dinov2:main"))),
            dino_force_reload=bool(mh.get("dino_force_reload", _pick(cfg_model, "dino_force_reload", False))),
            dino_trust_repo=bool(mh.get("dino_trust_repo", _pick(cfg_model, "dino_trust_repo", True))),
            dino_skip_validation=bool(mh.get("dino_skip_validation", _pick(cfg_model, "dino_skip_validation", True))),
            dino_hub_verbose=bool(mh.get("dino_hub_verbose", _pick(cfg_model, "dino_hub_verbose", False))),
            dino_log_torchhub_dir=bool(mh.get("dino_log_torchhub_dir", _pick(cfg_model, "dino_log_torchhub_dir", False))),
            field_head_mode=str(mh.get("field_head_mode", _pick(cfg_model, "field_head_mode", "independent"))),
            field_token_dim=int(mh.get("field_token_dim", _pick(cfg_model, "field_token_dim", 512))),
            field_base_channels=int(mh.get("field_base_channels", _pick(cfg_model, "field_base_channels", 128))),
            field_shared_channels=int(mh.get("field_shared_channels", _pick(cfg_model, "field_shared_channels", 64))),
            field_temporal_layers=int(mh.get("field_temporal_layers", _pick(cfg_model, "field_temporal_layers", 0))),
            field_spatial_channels=int(mh.get("field_spatial_channels", _pick(cfg_model, "field_spatial_channels", 0))),
            field_use_multiscale_spatial=bool(mh.get("field_use_multiscale_spatial", _pick(cfg_model, "field_use_multiscale_spatial", False))),
            field_use_shared_task_phys=bool(mh.get("field_use_shared_task_phys", _pick(cfg_model, "field_use_shared_task_phys", False))),
            field_use_geometry_residual=bool(mh.get("field_use_geometry_residual", _pick(cfg_model, "field_use_geometry_residual", False))),
            field_sequential_stress=bool(mh.get("field_sequential_stress", _pick(cfg_model, "field_sequential_stress", False))),
            use_stress_spatial_enhancer=bool(mh.get("use_stress_spatial_enhancer", _pick(cfg_model, "use_stress_spatial_enhancer", False))),
            use_flow_spatial_enhancer=mh.get("use_flow_spatial_enhancer", cfg_model.get("use_flow_spatial_enhancer")),
            flow_output_scale_init=float(mh.get("flow_output_scale_init", _pick(cfg_model, "flow_output_scale_init", 0.5))),
            flow_output_bias_init=float(mh.get("flow_output_bias_init", _pick(cfg_model, "flow_output_bias_init", -0.5))),
            force_out_channels=int(mh.get("force_out_channels", _pick(cfg_model, "force_out_channels", 3))),
            field_use_patch_tokens=bool(mh.get("field_use_patch_tokens", _pick(cfg_model, "field_use_patch_tokens", False))),
            field_patch_dim=int(mh.get("field_patch_dim", _pick(cfg_model, "field_patch_dim", 256))),
            param_chain_mode=str(mh.get("param_chain_mode", _pick(cfg_model, "param_chain_mode", "baseline"))),
            param_token_dim=int(mh.get("param_token_dim", _pick(cfg_model, "param_token_dim", 256))),
            param_mixer_layers=int(mh.get("param_mixer_layers", _pick(cfg_model, "param_mixer_layers", 2))),
            param_mixer_heads=int(mh.get("param_mixer_heads", _pick(cfg_model, "param_mixer_heads", 8))),
            param_mixer_dropout=float(
                mh.get(
                    "param_mixer_dropout",
                    _pick(cfg_model, "param_mixer_dropout", _pick(cfg_model, "head_dropout", 0.1)),
                )
            ),
            param_use_rgb_static=bool(mh.get("param_use_rgb_static", _pick(cfg_model, "param_use_rgb_static", True))),
            param_use_rgb_residual=bool(
                mh.get("param_use_rgb_residual", _pick(cfg_model, "param_use_rgb_residual", True))
            ),
            param_use_masked_field_tokens=bool(
                mh.get(
                    "param_use_masked_field_tokens",
                    _pick(cfg_model, "param_use_masked_field_tokens", True),
                )
            ),
        )
    else:
        model = LogicPhysModel(
            num_views=int(mh.get("num_views", max_views)),
            in_channels=int(mh.get("in_channels", _pick(cfg_model, "in_channels", 3))),
            num_frames=int(mh.get("num_frames", num_frames_model)),
            img_size=int(_pick(cfg_model, "img_size", img_size if img_size > 0 else 224)),
            num_targets=4,
            num_actions=ds.num_actions,
            dec_h=int(mh.get("dec_h", dec_h)),
            dec_w=int(mh.get("dec_w", dec_w)),
            encoder_embed_dim=int(_pick(cfg_model, "encoder_embed_dim", 384)),
            encoder_depth=int(_pick(cfg_model, "encoder_depth", 6)),
            encoder_num_heads=int(_pick(cfg_model, "encoder_num_heads", 6)),
            tubelet_size=int(_pick(cfg_model, "tubelet_size", 1)),
            patch_size=int(_pick(cfg_model, "patch_size", 32)),
            fusion_dim=int(_pick(cfg_model, "fusion_dim", 512)),
            fusion_heads=int(_pick(cfg_model, "fusion_heads", 8)),
            head_dropout=float(_pick(cfg_model, "head_dropout", 0.1)),
            use_uncertainty=bool(_pick(cfg_model, "use_uncertainty", False)),
            bottleneck_dim=int(_pick(cfg_model, "bottleneck_dim", 128)),
        )
    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        raise KeyError("checkpoint 缺少 model_state")
    model.load_state_dict(model_state, strict=True)
    return model


def _run_visual_export(args: argparse.Namespace, out_dir: Path) -> None:
    visual_dir = out_dir / "visuals"
    cmd: List[str] = [
        sys.executable,
        "-m",
        "logic_model.eval_visual",
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--eval_split",
        str(args.eval_split),
        "--batch_size",
        str(args.batch_size),
        "--num_samples",
        str(args.num_samples),
        "--sample_mode",
        str(args.sample_mode),
        "--seed",
        str(args.seed),
        "--out_dir",
        str(visual_dir),
    ]
    if args.save_field_videos:
        cmd += [
            "--save_field_videos",
            "--max_field_video_samples",
            str(args.max_field_video_samples),
            "--field_video_color_mode",
            str(args.field_video_color_mode),
        ]
    run_env = os.environ.copy()
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        run_env.pop(k, None)
    subprocess.run(cmd, check=True, env=run_env)


def _aggregate_and_write_param(sample_rows: List[Dict[str, object]], out_dir: Path) -> Dict[str, Any]:
    param_dir = out_dir / "param_metrics"
    sample_rows, weights, composite_rows, composite_summary = enrich_param_records_with_composite(sample_rows)
    write_jsonl(param_dir / "sample_records.jsonl", sample_rows)
    write_jsonl(param_dir / "sample_composite_errors.jsonl", composite_rows)
    write_csv(param_dir / "sample_composite_errors.csv", composite_rows)
    write_json(param_dir / "param_error_weights.json", weights)
    write_csv(param_dir / "param_error_weights.csv", weights)
    overall = aggregate_param_records(sample_rows, group_fields=())
    by_action = aggregate_param_records(sample_rows, group_fields=("action",))
    by_material = aggregate_param_records(sample_rows, group_fields=("material",))
    by_action_material = aggregate_param_records(sample_rows, group_fields=("action", "material"))
    for name, rows in (
        ("overall", overall),
        ("by_action", by_action),
        ("by_material", by_material),
        ("by_action_material", by_action_material),
    ):
        write_json(param_dir / f"{name}.json", rows)
        write_csv(param_dir / f"{name}.csv", rows)
    return {
        "overall": overall,
        "by_action": by_action,
        "by_material": by_material,
        "by_action_material": by_action_material,
        "param_error_weights": weights,
        "sample_composite_errors": composite_rows,
        "composite_summary": composite_summary,
    }


def _aggregate_and_write_field(sample_rows: List[Dict[str, object]], out_dir: Path) -> Dict[str, Any]:
    field_dir = out_dir / "field_metrics"
    write_jsonl(field_dir / "sample_records.jsonl", sample_rows)
    overall = aggregate_field_records(sample_rows, group_fields=())
    by_action = aggregate_field_records(sample_rows, group_fields=("action",))
    by_material = aggregate_field_records(sample_rows, group_fields=("material",))
    by_action_material = aggregate_field_records(sample_rows, group_fields=("action", "material"))
    for name, rows in (
        ("overall", overall),
        ("by_action", by_action),
        ("by_material", by_material),
        ("by_action_material", by_action_material),
    ):
        write_json(field_dir / f"{name}.json", rows)
        write_csv(field_dir / f"{name}.csv", rows)
    return {
        "overall": overall,
        "by_action": by_action,
        "by_material": by_material,
        "by_action_material": by_action_material,
    }


def _mean_field_metric(rows: Sequence[Dict[str, object]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _summary_scores(param_metrics: Dict[str, Any], field_overall: Sequence[Dict[str, object]]) -> Dict[str, float]:
    param_overall = param_metrics["overall"]
    param_maes = [float(r["mae"]) for r in param_overall]
    param_gt_trim_maes = [float(r["gt_trim_mae"]) for r in param_overall]
    param_err_trim_maes = [float(r["err_trim_mae"]) for r in param_overall]
    field_mses = [float(r["mse"]) for r in field_overall]
    field_ssim = [float(r["ssim"]) for r in field_overall]
    composite_summary = param_metrics.get("composite_summary", {})
    return {
        "param_mean_mae": float(sum(param_maes) / max(len(param_maes), 1)),
        "param_mean_gt_trim_mae": float(sum(param_gt_trim_maes) / max(len(param_gt_trim_maes), 1)),
        "param_mean_err_trim_mae": float(sum(param_err_trim_maes) / max(len(param_err_trim_maes), 1)),
        "param_mean_composite_error": composite_summary.get("mean"),
        "param_median_composite_error": composite_summary.get("median"),
        "param_p90_composite_error": composite_summary.get("p90"),
        "field_mean_mse": float(sum(field_mses) / max(len(field_mses), 1)),
        "field_mean_ssim": float(sum(field_ssim) / max(len(field_ssim), 1)),
        "field_mean_region_iou": _mean_field_metric(field_overall, "region_iou"),
        "field_mean_soft_region_iou": _mean_field_metric(field_overall, "soft_region_iou"),
        "field_mean_region_recall": _mean_field_metric(field_overall, "region_recall"),
        "flow_mean_coverage_iou": _mean_field_metric(field_overall, "flow_coverage_iou"),
        "flow_mean_coverage_recall": _mean_field_metric(field_overall, "flow_coverage_recall"),
        "flow_mean_velocity_weighted_overlap": _mean_field_metric(field_overall, "flow_velocity_weighted_overlap"),
        "flow_mean_velocity_similarity_overlap": _mean_field_metric(field_overall, "flow_velocity_similarity_overlap"),
        "flow_mean_velocity_weighted_epe": _mean_field_metric(field_overall, "flow_velocity_weighted_epe"),
        "flow_mean_motion_quality": _mean_field_metric(field_overall, "flow_motion_quality"),
        "force_mean_main_recall": _mean_field_metric(field_overall, "force_main_recall"),
        "force_mean_centroid_dist": _mean_field_metric(field_overall, "force_centroid_dist"),
        "stress_mean_hotspot_recall": _mean_field_metric(field_overall, "stress_hotspot_recall"),
        "stress_mean_weighted_mae": _mean_field_metric(field_overall, "stress_weighted_mae"),
    }


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = _init_distributed()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(args.config)
    ds, eval_ds, chosen_ids, meta_map = _build_eval_dataset(cfg=cfg, args=args)
    eval_sampler: Optional[DistributedSampler] = None
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("多卡 eval_logic 需要 CUDA")
        eval_sampler = DistributedSampler(
            eval_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    loader = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=eval_sampler,
        num_workers=int(args.num_workers),
        collate_fn=collate_lmdb_gt_batch,
        drop_last=False,
    )

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"invalid checkpoint: {ckpt_path}")

    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = _resolve_device(args.device)
    model = _build_model_from_checkpoint(cfg=cfg, ckpt=ckpt, ds=ds)
    model.to(device)
    model.eval()
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    param_sample_rows: List[Dict[str, object]] = []
    field_sample_rows: List[Dict[str, object]] = []
    field_export_dir = (
        Path(args.field_export_dir).expanduser().resolve()
        if str(args.field_export_dir).strip()
        else (out_dir / "field_exports")
    )
    field_video_dir = (
        Path(args.field_video_dir).expanduser().resolve()
        if str(args.field_video_dir).strip()
        else (out_dir / "field_videos")
    )

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            stress_gt = batch["stress"].to(device)
            flow_gt = batch["flow"].to(device)
            force_gt = batch["force_mask"].to(device)
            object_gt = batch["object_mask"].to(device)
            params_gt = batch["params"].to(device)

            mcore = _unwrap_model(model)
            in_channels = int(getattr(mcore, "in_channels", 3))
            x = rgb[:, :, :in_channels, :, :, :] if in_channels > 1 else rgb[:, :, :1, :, :, :]
            num_frames_model = int(getattr(mcore, "num_frames", int(x.shape[3])))
            if int(x.shape[3]) != num_frames_model:
                x = _resample_time_bvcthw(x, num_frames_model)
                stress_gt = _resample_time_bvcthw(stress_gt, num_frames_model)
                flow_gt = _resample_time_bvcthw(flow_gt, num_frames_model)
                force_gt = _resample_time_bvcthw(force_gt, num_frames_model)
                object_gt = _resample_time_bvcthw(object_gt, num_frames_model)

            try:
                out = model(x, object_mask=object_gt)
            except TypeError:
                out = model(x)
            pred_raw = out["param_pred_raw"].detach().cpu().numpy()
            gt_raw = params_gt.detach().cpu().numpy()
            sample_ids = [str(s) for s in batch["sample_id"]]

            for i, sid in enumerate(sample_ids):
                meta = meta_map.get(sid)
                action = meta.action if meta is not None else str(batch.get("action_name", ["unknown"] * len(sample_ids))[i])
                material = meta.material_group if meta is not None else "unknown"
                object_name = meta.object_name if meta is not None else "unknown"
                param_sample_rows.append(
                    build_param_sample_record(
                        sample_id=sid,
                        action=action,
                        material=material,
                        object_name=object_name,
                        gt_raw=gt_raw[i].tolist(),
                        pred_raw=pred_raw[i].tolist(),
                    )
                )

                s_pred = out["stress_field_pred"][i : i + 1].detach().cpu()
                f_pred = out["flow_field_pred"][i : i + 1].detach().cpu()
                m_pred = out["force_pred"][i : i + 1].detach().cpu()
                s_gt = stress_gt[i : i + 1].detach().cpu()
                f_gt = flow_gt[i : i + 1].detach().cpu()
                m_gt = force_gt[i : i + 1].detach().cpu()
                o_gt = object_gt[i : i + 1].detach().cpu()

                field_sample_rows.append(
                    build_field_sample_record(
                        sample_id=sid,
                        action=action,
                        material=material,
                        object_name=object_name,
                        field_name="stress",
                        pred_bvcthw=s_pred,
                        gt_bvcthw=s_gt,
                        object_mask_bvcthw=o_gt,
                    )
                )
                field_sample_rows.append(
                    build_field_sample_record(
                        sample_id=sid,
                        action=action,
                        material=material,
                        object_name=object_name,
                        field_name="flow",
                        pred_bvcthw=f_pred,
                        gt_bvcthw=f_gt,
                        object_mask_bvcthw=o_gt,
                    )
                )
                field_sample_rows.append(
                    build_field_sample_record(
                        sample_id=sid,
                        action=action,
                        material=material,
                        object_name=object_name,
                        field_name="force_mask",
                        pred_bvcthw=m_pred,
                        gt_bvcthw=m_gt,
                        object_mask_bvcthw=o_gt,
                    )
                )
                if args.export_field_tensors:
                    payload = build_field_export_payload(
                        sample_id=sid,
                        stress_pred=s_pred,
                        stress_gt=s_gt,
                        flow_pred=f_pred,
                        flow_gt=f_gt,
                        force_pred=m_pred,
                        force_gt=m_gt,
                        object_mask=o_gt,
                        meta={
                            "checkpoint": str(ckpt_path),
                            "model_name": "logic_model2" if isinstance(mcore, LogicPhysModel2) else "logic_model",
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
                        stress_pred=s_pred,
                        stress_gt=s_gt,
                        flow_pred=f_pred,
                        flow_gt=f_gt,
                        force_pred=m_pred,
                        force_gt=m_gt,
                        view_idx=int(args.field_video_view),
                        fps=int(args.field_video_fps),
                        color_mode=str(args.field_video_color_mode),
                    )

    if distributed:
        dist.barrier()
        if rank == 0:
            param_rows_gather: List[Optional[List[Dict[str, object]]]] = [None] * world_size
            field_rows_gather: List[Optional[List[Dict[str, object]]]] = [None] * world_size
        else:
            param_rows_gather = None
            field_rows_gather = None
        dist.gather_object(param_sample_rows, object_gather_list=param_rows_gather, dst=0)
        dist.gather_object(field_sample_rows, object_gather_list=field_rows_gather, dst=0)
        if rank == 0:
            merged_param_rows: List[Dict[str, object]] = []
            merged_field_rows: List[Dict[str, object]] = []
            for sub in param_rows_gather or []:
                if sub:
                    merged_param_rows.extend(sub)
            for sub in field_rows_gather or []:
                if sub:
                    merged_field_rows.extend(sub)
            param_sample_rows = merged_param_rows
            field_sample_rows = merged_field_rows

    if rank == 0:
        param_metrics = _aggregate_and_write_param(param_sample_rows, out_dir)
        field_metrics = _aggregate_and_write_field(field_sample_rows, out_dir)
        score = _summary_scores(param_metrics, field_metrics["overall"])

        meta = {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(ckpt_path),
            "eval_split": str(args.eval_split),
            "num_samples_requested": int(args.num_samples),
            "num_samples_used": len(chosen_ids),
            "sample_mode": str(args.sample_mode),
            "seed": int(args.seed),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "distributed": bool(distributed),
            "world_size": int(world_size),
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
            "chosen_sample_ids": chosen_ids,
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

        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "checkpoint": str(ckpt_path),
                    "param_mean_mae": score["param_mean_mae"],
                    "field_mean_mse": score["field_mean_mse"],
                    "field_mean_ssim": score["field_mean_ssim"],
                    "field_mean_region_iou": score["field_mean_region_iou"],
                    "flow_mean_coverage_iou": score["flow_mean_coverage_iou"],
                    "flow_mean_velocity_weighted_overlap": score["flow_mean_velocity_weighted_overlap"],
                    "flow_mean_velocity_similarity_overlap": score["flow_mean_velocity_similarity_overlap"],
                    "flow_mean_velocity_weighted_epe": score["flow_mean_velocity_weighted_epe"],
                    "flow_mean_motion_quality": score["flow_mean_motion_quality"],
                    "force_mean_main_recall": score["force_mean_main_recall"],
                    "stress_mean_hotspot_recall": score["stress_mean_hotspot_recall"],
                    "num_samples": len(chosen_ids),
                    "distributed": bool(distributed),
                    "world_size": int(world_size),
                },
                ensure_ascii=False,
            )
        )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
