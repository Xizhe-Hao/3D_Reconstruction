# -*- coding: utf-8 -*-
"""
Arch4 多卡并行推理/评估 + 自动可视化分析。

支持两种运行方式：
1) 单卡：直接 python -m my_model.eval ...
2) 多卡：torchrun --nproc_per_node=N -m my_model.eval ...

评估输出：
- `eval_metrics.json`（rank0 写出）
- `visual_report.md` + 若干样本目录（rank0 写出）
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .arch4_model import Arch4VideoMAEPhysModel, build_arch4_model
from .dataset import DatasetArch4, resolve_flat_dataset_root


def _to_target_params(p: torch.Tensor) -> torch.Tensor:
    # p(raw): [B,4] => [log(1+clamp(E)), nu, log(1+clamp(density)), log(1+clamp(yield_stress))]
    e = torch.log1p(torch.clamp(p[:, 0], min=0))
    nu = p[:, 1]
    density = torch.log1p(torch.clamp(p[:, 2], min=0))
    yield_stress = torch.log1p(torch.clamp(p[:, 3], min=0))
    return torch.stack([e, nu, density, yield_stress], dim=1)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint(weights_path: Path) -> Dict[str, Any]:
    obj = torch.load(str(weights_path), map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj
        if "model" in obj and isinstance(obj["model"], dict):
            return obj
        return obj
    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    # 兼容直接保存 state_dict 的情况
    if all(isinstance(k, str) for k in ckpt.keys()) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore[return-value]
    raise ValueError("无法从 checkpoint 中提取 state_dict")


def _init_distributed() -> Tuple[bool, int, int, int]:
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
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def _downsample_gt_field(gt_bvcthw: torch.Tensor, dec_h: int, dec_w: int) -> torch.Tensor:
    # gt_bvcthw: [B,V,C,T,H,W] -> tgt_bvcthw: [B,V,C,T,dec_h,dec_w]
    if gt_bvcthw.dim() != 6:
        raise ValueError(f"gt field dim must be 6, got {gt_bvcthw.dim()}")
    b, v, c, t, h_i, w_i = gt_bvcthw.shape
    x = gt_bvcthw.reshape(b * v * t, c, h_i, w_i)
    y = F.interpolate(x, size=(dec_h, dec_w), mode="bilinear", align_corners=False)
    tgt = y.view(b, v, t, c, dec_h, dec_w).permute(0, 1, 3, 2, 4, 5).contiguous()
    return tgt


def _to_color_map(img_2d: torch.Tensor) -> "np.ndarray":
    import numpy as np

    a = img_2d.detach().cpu().float().numpy()
    amin = float(a.min())
    amax = float(a.max())
    denom = max(1e-8, amax - amin)
    a01 = (a - amin) / denom
    u8 = (a01 * 255.0).clip(0, 255).astype("uint8")
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _save_field_video_compare(
    pred_bvcthw: torch.Tensor,
    tgt_bvcthw: torch.Tensor,
    out_dir: Path,
    *,
    field_name: str,
    view_idx: int = 0,
    fps: int = 8,
) -> None:
    """
    pred/tgt: [B,V,1,T,H,W]
    在 out_dir 下保存 pred_vs_gt 对比视频（逐帧，左pred右gt）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    b = int(pred_bvcthw.shape[0])
    if b <= 0:
        return
    v = int(pred_bvcthw.shape[1])
    view_idx = max(0, min(int(view_idx), v - 1))
    t = int(pred_bvcthw.shape[3])
    h = int(pred_bvcthw.shape[4])
    w = int(pred_bvcthw.shape[5])
    if t <= 0:
        return

    out_path = out_dir / f"{field_name}_pred_vs_gt.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, int(fps))),
        (w * 2, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out_path}")
    try:
        for ti in range(t):
            pred_2d = pred_bvcthw[0, view_idx, 0, ti]
            tgt_2d = tgt_bvcthw[0, view_idx, 0, ti]
            pred_color = _to_color_map(pred_2d)
            tgt_color = _to_color_map(tgt_2d)
            frame = cv2.hconcat([pred_color, tgt_color])
            writer.write(frame)
    finally:
        writer.release()


def evaluate_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dec_h: int,
    dec_w: int,
    vis_dir: Optional[Path],
    num_vis: int,
    vis_view: int,
    rank: int,
) -> Dict[str, float]:
    model.eval()
    total_count = 0

    # 参数：在 log 空间评估
    param_mae_sum = 0.0
    param_rmse_sum = 0.0

    # 场：在 (dec_h, dec_w) 空间评估（GT 仅空间下采样，保留时序）
    stress_mse_sum = 0.0
    flow_mse_sum = 0.0
    force_mse_sum = 0.0

    vis_done = 0
    seen_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            # return_sample_id=True 时：x,stress,flow,force,params,id
            if len(batch) == 6:
                x, stress_gt, flow_gt, force_gt, params_gt, sample_id = batch
            else:
                x, stress_gt, flow_gt, force_gt, params_gt = batch
                sample_id = ["unknown"] * int(x.shape[0])

            x = x.to(device, non_blocking=True)
            stress_gt = stress_gt.to(device, non_blocking=True)
            flow_gt = flow_gt.to(device, non_blocking=True)
            force_gt = force_gt.to(device, non_blocking=True)
            params_gt = params_gt.to(device, non_blocking=True)

            out = model(x)
            # 模型直接输出目标空间 [logE, nu, logDensity, logYield]
            param_pred = out["param_pred"]  # [B,4] (target-space)
            pred_raw = out.get("param_pred_raw")
            if pred_raw is None:
                pred_raw = torch.stack(
                    [
                        torch.expm1(torch.clamp(param_pred[:, 0], min=0.0)),
                        torch.clamp(param_pred[:, 1], min=0.0, max=0.5),
                        torch.expm1(torch.clamp(param_pred[:, 2], min=0.0)),
                        torch.expm1(torch.clamp(param_pred[:, 3], min=0.0)),
                    ],
                    dim=1,
                )

            pred_log = param_pred
            gt_log = _to_target_params(params_gt)
            diff = pred_log - gt_log
            mae_per_sample = diff.abs().mean(dim=1)  # [B]
            rmse_per_sample = (diff.pow(2).mean(dim=1)).sqrt()  # [B]

            stress_pred = out["stress_field_pred"]
            flow_pred = out["flow_field_pred"]
            force_pred = out["force_pred"]

            # GT 下采样到 dec_h/dec_w（保留时间维）
            stress_tgt = _downsample_gt_field(stress_gt, dec_h, dec_w)
            flow_tgt = _downsample_gt_field(flow_gt, dec_h, dec_w)
            force_tgt = _downsample_gt_field(force_gt, dec_h, dec_w)

            # per-sample mse: mean over [V,C,T,H,W]
            stress_mse = (stress_pred - stress_tgt).pow(2).mean(dim=(1, 2, 3, 4, 5))  # [B]
            flow_mse = (flow_pred - flow_tgt).pow(2).mean(dim=(1, 2, 3, 4, 5))
            force_mse = (force_pred - force_tgt).pow(2).mean(dim=(1, 2, 3, 4, 5))

            bsz = int(x.shape[0])
            total_count += bsz
            param_mae_sum += float(mae_per_sample.sum().item())
            param_rmse_sum += float(rmse_per_sample.sum().item())
            stress_mse_sum += float(stress_mse.sum().item())
            flow_mse_sum += float(flow_mse.sum().item())
            force_mse_sum += float(force_mse.sum().item())

            # rank0 负责可视化
            if rank == 0 and vis_dir is not None and vis_done < num_vis:
                b = int(x.shape[0])
                for i in range(b):
                    if vis_done >= num_vis:
                        break
                    sid = sample_id[i] if isinstance(sample_id, list) else str(sample_id)
                    sid = str(sid)
                    if sid in seen_ids:
                        continue
                    seen_ids.append(sid)
                    one_dir = vis_dir / sid
                    one_dir.mkdir(parents=True, exist_ok=True)

                    # fields：保存 pred_vs_gt 时序对比视频
                    _save_field_video_compare(
                        stress_pred[i : i + 1],
                        stress_tgt[i : i + 1],
                        one_dir / "stress",
                        field_name="stress",
                        view_idx=vis_view,
                    )
                    _save_field_video_compare(
                        flow_pred[i : i + 1],
                        flow_tgt[i : i + 1],
                        one_dir / "flow",
                        field_name="flow",
                        view_idx=vis_view,
                    )
                    _save_field_video_compare(
                        force_pred[i : i + 1],
                        force_tgt[i : i + 1],
                        one_dir / "force",
                        field_name="force",
                        view_idx=vis_view,
                    )

                    pred_params_i = pred_log[i].detach().cpu().tolist()
                    gt_params_i = gt_log[i].detach().cpu().tolist()
                    err_i = (pred_log[i] - gt_log[i]).abs().detach().cpu().tolist()
                    pred_params_raw_i = pred_raw[i].detach().cpu().tolist()
                    gt_params_raw_i = params_gt[i].detach().cpu().tolist()
                    (one_dir / "params.json").write_text(
                        json.dumps(
                            {
                                "sample_id": sid,
                                "pred_log_params": pred_params_i,
                                "gt_log_params": gt_params_i,
                                "abs_error_log": err_i,
                                "pred_raw_params": pred_params_raw_i,
                                "gt_raw_params": gt_params_raw_i,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    vis_done += 1

    return {
        "count": float(total_count),
        "param_mae_log_sum": float(param_mae_sum),
        "param_rmse_log_sum": float(param_rmse_sum),
        "stress_mse_sum": float(stress_mse_sum),
        "flow_mse_sum": float(flow_mse_sum),
        "force_mse_sum": float(force_mse_sum),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Arch4 多卡评估/推理")
    ap.add_argument("--config", type=str, default="my_model/configs.json", help="configs.json 路径（相对 PhysGaussian/ 或绝对路径）")
    ap.add_argument("--weights", type=str, required=True, help="模型权重 .pt/.pth 路径")

    ap.add_argument("--split_root", type=str, default=None, help="评估用数据根目录（覆盖 config.data.split_root）")
    ap.add_argument("--auto_output", type=str, default="auto_output", help="split_root 为相对路径时的父目录名")
    # 兼容旧参数：--test_ids_json（只取 test_ids）
    ap.add_argument("--test_ids_json", type=str, default=None, help="包含 test_ids 的 json 文件（train_test_split 输出）")
    # 新参数：通用采样 id 文件
    ap.add_argument("--sample_ids_json", type=str, default=None, help="包含 sample_ids 的 json 文件（train_test_split 输出）")
    ap.add_argument("--sample_ids_key", type=str, default="test_ids", help="sample_ids_json 中要取的 key（如 train_ids / test_ids）")

    ap.add_argument("--sample_limit", type=int, default=0, help="限制评估最多使用多少个样本（只对传入 sample_ids 时生效）")

    ap.add_argument("--out_dir", type=str, default=None, help="输出目录（默认自动创建）")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--num_vis", type=int, default=3, help="保存多少个样本的可视化（rank0）")
    ap.add_argument("--vis_view", type=int, default=0, help="可视化使用的视角索引")

    args = ap.parse_args()

    distributed, rank, local_rank, world_size = _init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    here = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = here / cfg_path
    cfg = _load_json(cfg_path)

    cfg_data = cfg.get("data") or {}
    cfg_model = cfg.get("model") or {}
    cfg_train = cfg.get("train") or {}
    preflight_eval = bool(cfg_train.get("preflight", True))

    split_root = args.split_root if args.split_root is not None else cfg_data.get("split_root")
    if not split_root:
        raise ValueError("split_root 未提供，且 config.data.split_root 为空")

    auto_output = args.auto_output
    split_root_path = resolve_flat_dataset_root(split_root, here / auto_output)

    # 读取 sample_ids 子集（训练/测试都可用同一套逻辑）
    sample_ids: Optional[List[str]] = None
    ids_json_arg = args.sample_ids_json if args.sample_ids_json is not None else args.test_ids_json
    ids_key = str(args.sample_ids_key or "test_ids")
    if ids_json_arg:
        tid = Path(ids_json_arg)
        if not tid.is_absolute():
            tid = here / tid
        sj = _load_json(tid)
        sample_ids = list(sj.get(ids_key) or [])

    sample_limit = int(args.sample_limit or 0)
    if sample_ids is not None and sample_limit > 0:
        sample_ids = sample_ids[:sample_limit]

    dec_h = int(cfg_model.get("dec_h", 56))
    dec_w = int(cfg_model.get("dec_w", 56))
    num_frames = int(cfg_model.get("num_frames", 16))
    max_views = int(cfg_model.get("num_views", 3))
    img_size = int(cfg_model.get("img_size", 224))
    input_mode_cfg = str(cfg_model.get("input_mode", "images")).strip().lower()
    in_channels = int(cfg_model.get("in_channels", 1))
    expected_in_channels = 1
    if int(in_channels) != int(expected_in_channels):
        if rank == 0:
            print(
                f"[warn] config mismatch: model.input_mode={input_mode_cfg!r} model.in_channels={in_channels} -> use {expected_in_channels}",
                flush=True,
            )
        in_channels = int(expected_in_channels)
    use_aux_field_heads = bool(cfg_model.get("use_aux_field_heads", True))

    # build model with accepted init kwargs
    accepted = set(torch.nn.modules.module.Module.__init__.__code__.co_varnames)  # no-op, just avoid mypy
    from inspect import signature

    sig = signature(Arch4VideoMAEPhysModel.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    model_kwargs = {k: v for k, v in cfg_model.items() if k in accepted}
    model_kwargs.update(
        {
            "num_views": max_views,
            "in_channels": in_channels,
            "num_frames": num_frames,
            "img_size": img_size,
            "dec_h": dec_h,
            "dec_w": dec_w,
            "use_aux_field_heads": use_aux_field_heads,
        }
    )
    model = build_arch4_model(**model_kwargs)

    weights_path = Path(args.weights).expanduser()
    if not weights_path.is_absolute():
        weights_path = (here / weights_path).resolve()
    ckpt = _load_checkpoint(weights_path)
    state_dict = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"[weights] missing={len(missing)} unexpected={len(unexpected)} strict=False")

    model.to(device)
    model.eval()

    # output dir
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (here / "output_eval" / f"arch4_eval_{ts}").resolve()
    vis_dir = out_dir / "vis" if rank == 0 else None

    if rank == 0:
        if out_dir.exists():
            # 避免污染旧结果：只清掉 vis
            (out_dir / "vis").mkdir(parents=True, exist_ok=True)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    if distributed:
        dist.barrier()

    ds = DatasetArch4(
        split_root_path,
        img_size=img_size,
        max_views=max_views,
        num_frames=num_frames,
        source="auto",
        preflight=preflight_eval,
        verbose=(rank == 0),
        sample_ids=sample_ids,
        input_mode="images",
        return_sample_id=True,
    )

    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if distributed else None

    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
    )

    stats = evaluate_one_epoch(
        model=model,
        loader=loader,
        device=device,
        dec_h=dec_h,
        dec_w=dec_w,
        vis_dir=vis_dir,
        num_vis=int(args.num_vis),
        vis_view=int(args.vis_view),
        rank=rank,
    )

    # all-reduce sums
    count_t = torch.tensor(stats["count"], device=device, dtype=torch.float64)
    mae_t = torch.tensor(stats["param_mae_log_sum"], device=device, dtype=torch.float64)
    rmse_t = torch.tensor(stats["param_rmse_log_sum"], device=device, dtype=torch.float64)
    stress_t = torch.tensor(stats["stress_mse_sum"], device=device, dtype=torch.float64)
    flow_t = torch.tensor(stats["flow_mse_sum"], device=device, dtype=torch.float64)
    force_t = torch.tensor(stats["force_mse_sum"], device=device, dtype=torch.float64)

    if distributed:
        for t in (count_t, mae_t, rmse_t, stress_t, flow_t, force_t):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    if rank == 0:
        count = float(count_t.item())
        metrics = {
            "count": count,
            "param_mae_log": float(mae_t.item() / max(count, 1.0)),
            "param_rmse_log": float(rmse_t.item() / max(count, 1.0)),
            "stress_mse": float(stress_t.item() / max(count, 1.0)),
            "flow_mse": float(flow_t.item() / max(count, 1.0)),
            "force_mse": float(force_t.item() / max(count, 1.0)),
            "dec_h": dec_h,
            "dec_w": dec_w,
            "num_views": max_views,
            "num_frames": num_frames,
            "img_size": img_size,
        }
        (out_dir / "eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # visual report
        vis_samples = sorted([p.name for p in (out_dir / "vis").iterdir() if p.is_dir()])
        report = {
            "weights": str(weights_path),
            "config": str(cfg_path),
            "metrics": metrics,
            "vis_samples": vis_samples,
        }
        lines = [
            "# Arch4 自动评估与可视化报告",
            "",
            f"- out_dir: `{out_dir}`",
            f"- weights: `{weights_path}`",
            "",
            "## Metrics",
            f"- count: {metrics['count']}",
            f"- param_mae_log: {metrics['param_mae_log']:.6f}",
            f"- param_rmse_log: {metrics['param_rmse_log']:.6f}",
            f"- stress_mse: {metrics['stress_mse']:.6f}",
            f"- flow_mse: {metrics['flow_mse']:.6f}",
            f"- force_mse: {metrics['force_mse']:.6f}",
            "",
            "## Visualized samples",
            *(f"- {sid}" for sid in vis_samples),
            "",
            "## How to inspect",
            "- 每个样本目录包含：`params.json`、`stress/flow/force/` 三个子目录下的可视化 PNG。",
        ]
        (out_dir / "visual_report.md").write_text("\n".join(lines), encoding="utf-8")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

