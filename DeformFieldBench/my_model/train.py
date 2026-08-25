# -*- coding: utf-8 -*-
"""
最小 Arch4 训练入口：6 通道输入（RGB+force_mask），主损失为物理参数回归，
辅助损失为 stress / flow / force 场 MSE（下采样 + 时间均值）。
运行示例（在 ``PhysGaussian`` 目录下）::

    python -m my_model.train --split_root auto_output/dataset_deformation_stress_500_new/train --epochs 1
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import inspect
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import sys
import subprocess
import signal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP

from .arch4_model import build_arch4_model
from .dataset import DatasetArch4, resolve_flat_dataset_root
from .losses import Arch4LossConfig, Arch4RegressionLoss, arch4_field_supervision_mse

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None


def _to_target_params(p: torch.Tensor) -> torch.Tensor:
    e = torch.log1p(torch.clamp(p[:, 0], min=0))
    nu = p[:, 1]
    density = torch.log1p(torch.clamp(p[:, 2], min=0))
    yield_stress = torch.log1p(torch.clamp(p[:, 3], min=0))
    return torch.stack([e, nu, density, yield_stress], dim=1)


def _target_to_raw_params(p: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.expm1(torch.clamp(p[:, 0], min=0)),
            torch.clamp(p[:, 1], min=0.0, max=0.5),
            torch.expm1(torch.clamp(p[:, 2], min=0)),
            torch.expm1(torch.clamp(p[:, 3], min=0)),
        ],
        dim=1,
    )


def _module_grad_norm(module: Optional[torch.nn.Module]) -> float:
    if module is None:
        return 0.0
    total = 0.0
    has_grad = False
    for p in module.parameters():
        if p.grad is None:
            continue
        has_grad = True
        g = p.grad.detach().float()
        total += float((g * g).sum().item())
    if not has_grad:
        return 0.0
    return float(total ** 0.5)


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a json object: {path}")
    return cfg


def _pick(d: Dict[str, Any], k: str, default: Any) -> Any:
    v = d.get(k, default) if isinstance(d, dict) else default
    return default if v is None else v


def _init_distributed() -> Tuple[bool, int, int, int]:
    """
    自动检测 torchrun 环境变量：
    - WORLD_SIZE > 1 => 初始化分布式
    """
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 配置文件（包含 data/model/train；优先从配置读取）",
    )
    ap.add_argument(
        "--split_root",
        type=str,
        default=None,
        help="扁平集 split 根目录，如 auto_output/<ds>/train",
    )
    ap.add_argument("--auto_output", type=str, default=None, help="含各数据集的父目录")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--max_views", type=int, default=None)
    ap.add_argument("--num_frames", type=int, default=None)
    ap.add_argument("--lambda_stress", type=float, default=None)
    ap.add_argument("--lambda_flow", type=float, default=None)
    ap.add_argument("--lambda_force", type=float, default=None)
    ap.add_argument("--head_dropout", type=float, default=None, help="覆盖参数头/模型头 dropout（overfit 推荐 0.0）")
    ap.add_argument("--no_log_scale", action="store_true", default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--overfit_num_samples", type=int, default=None, help="仅用前N个样本训练（推荐 1/8/16）")
    ap.add_argument("--overfit_no_shuffle", action="store_true", help="overfit 场景关闭 shuffle")
    ap.add_argument("--disable_aux_losses", action="store_true", help="禁用辅助损失（lambda_stress/flow/force 全置0）")
    ap.add_argument("--weak_aux_losses", action="store_true", help="弱辅助监督（0.02/0.02/0.01）")
    ap.add_argument("--debug_overfit", action="store_true", help="打印 overfit 调试信息（参数/损失/梯度/lr）")
    ap.add_argument("--debug_overfit_every_steps", type=int, default=None, help="debug_overfit 的打印间隔 step")
    ap.add_argument(
        "--ddp",
        action="store_true",
        default=False,
        help="强制启用 DDP（一般用 torchrun 时不需要显式指定）",
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[1]

    distributed, rank, local_rank, world_size = _init_distributed()
    if args.ddp and not distributed:
        raise ValueError("--ddp 被显式指定但未检测到 WORLD_SIZE>1 的分布式环境")

    # Ctrl+C 一次性关停：本进程收到 SIGINT/SIGTERM 后设置 stop 标志，
    # rank0 最终以非 0 退出码退出，从而让 torchrun 收掉其余 worker。
    stop_requested = False

    def _on_signal(signum, _frame):  # type: ignore[no-untyped-def]
        nonlocal stop_requested
        stop_requested = True
        if rank == 0:
            print(f"[signal] received {signum}, stopping...", flush=True)

    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
    except Exception:
        # 某些环境/线程下可能不允许注册信号处理器，忽略即可
        pass

    cfg: Dict[str, Any] = {}
    cfg_path_resolved: Optional[Path] = None
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = here / cfg_path
        cfg_path = cfg_path.resolve()
        cfg_path_resolved = cfg_path
        cfg = _load_config(cfg_path)
    else:
        # 若未提供 --config，则默认使用 my_model/configs.json（便于 quick_eval 调用）
        cfg_path_resolved = (here / "my_model" / "configs.json").resolve()

    cfg_data = cfg.get("data") or {}
    cfg_model = cfg.get("model") or {}
    cfg_train = cfg.get("train") or {}

    # regression debug print（只在 rank0 打印；用于小样本 overfit/消融时排查）
    cfg_dbg = cfg_train.get("debug_regression_print") if isinstance(cfg_train, dict) else None
    cfg_dbg = cfg_dbg if isinstance(cfg_dbg, dict) else {}
    dbg_enabled = bool(cfg_dbg.get("enabled", False))
    dbg_every_steps = max(1, int(cfg_dbg.get("every_steps", 50)))
    dbg_max_prints = max(0, int(cfg_dbg.get("max_prints", 5)))
    dbg_sample_idx = max(0, int(cfg_dbg.get("sample_index_in_batch", 0)))
    dbg_printed = 0
    debug_overfit = bool(args.debug_overfit or bool(cfg_train.get("debug_overfit", False)))
    debug_overfit_every_steps = int(
        args.debug_overfit_every_steps
        if args.debug_overfit_every_steps is not None
        else cfg_train.get("debug_overfit_every_steps", 20)
    )
    debug_overfit_every_steps = max(1, int(debug_overfit_every_steps))

    split_root = args.split_root if args.split_root is not None else cfg_data.get("split_root")
    if not split_root:
        raise ValueError("--split_root 未提供，且 config.data.split_root 为空")

    auto_output = args.auto_output if args.auto_output is not None else cfg_data.get("auto_output", "auto_output")
    root = resolve_flat_dataset_root(str(split_root), here / str(auto_output))

    img_size = args.img_size if args.img_size is not None else _pick(cfg_model, "img_size", 224)
    max_views = args.max_views if args.max_views is not None else _pick(cfg_model, "num_views", 3)
    num_frames = args.num_frames if args.num_frames is not None else _pick(cfg_model, "num_frames", 16)

    input_mode_cfg = str(cfg_model.get("input_mode", "images")).strip().lower()
    in_channels = _pick(cfg_model, "in_channels", 1)
    # 重构后输入固定为 images 单通道时序
    expected_in_channels = 1
    if int(in_channels) != int(expected_in_channels):
        if rank == 0:
            print(
                f"[warn] config mismatch: model.input_mode={input_mode_cfg!r} model.in_channels={in_channels} -> use {expected_in_channels}",
                flush=True,
            )
        in_channels = int(expected_in_channels)
    dec_h = _pick(cfg_model, "dec_h", 56)
    dec_w = _pick(cfg_model, "dec_w", 56)
    use_aux_field_heads = bool(_pick(cfg_model, "use_aux_field_heads", True))

    epochs = args.epochs if args.epochs is not None else _pick(cfg_train, "epochs", 1)
    batch_size = args.batch_size if args.batch_size is not None else _pick(cfg_train, "batch_size", 1)
    lr = args.lr if args.lr is not None else _pick(cfg_train, "lr", 3e-4)
    num_workers = args.num_workers if args.num_workers is not None else _pick(cfg_train, "num_workers", 0)

    preflight = bool(_pick(cfg_train, "preflight", True))

    lambda_stress = args.lambda_stress if args.lambda_stress is not None else _pick(cfg_train, "lambda_stress", 0.15)
    lambda_flow = args.lambda_flow if args.lambda_flow is not None else _pick(cfg_train, "lambda_flow", 0.15)
    lambda_force = args.lambda_force if args.lambda_force is not None else _pick(cfg_train, "lambda_force", 0.15)

    # overfit 模式：辅助损失开关（disable 优先于 weak）
    # 约定：
    # - disable_aux_losses: 强制三项全关
    # - weak_aux_losses: 若 JSON/CLI 未显式给 lambda_*，才回退到 0.02/0.02/0.01
    #   （若你在 config 里手动改了 lambda_*，会保留你的值）
    disable_aux_losses = bool(args.disable_aux_losses or bool(cfg_train.get("disable_aux_losses", False)))
    weak_aux_losses = bool(args.weak_aux_losses or bool(cfg_train.get("weak_aux_losses", False)))
    cli_has_lambda = (
        args.lambda_stress is not None
        or args.lambda_flow is not None
        or args.lambda_force is not None
    )
    cfg_has_lambda = (
        isinstance(cfg_train, dict)
        and (
            ("lambda_stress" in cfg_train)
            or ("lambda_flow" in cfg_train)
            or ("lambda_force" in cfg_train)
        )
    )
    if disable_aux_losses:
        lambda_stress = 0.0
        lambda_flow = 0.0
        lambda_force = 0.0
    elif weak_aux_losses and (not cli_has_lambda) and (not cfg_has_lambda):
        lambda_stress = 0.02
        lambda_flow = 0.02
        lambda_force = 0.01

    no_log_scale = args.no_log_scale if args.no_log_scale is not None else bool(_pick(cfg_train, "no_log_scale", False))

    overfit_num_samples = int(
        args.overfit_num_samples
        if args.overfit_num_samples is not None
        else cfg_train.get("overfit_num_samples", 0)
    )
    overfit_no_shuffle = bool(args.overfit_no_shuffle or bool(cfg_train.get("overfit_no_shuffle", False)))

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = args.device if args.device is not None else _pick(cfg_train, "device", default_device)
    if str(device_str).lower() == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    if distributed:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        device_str = str(device)
    else:
        device = torch.device(str(device_str))

    # 训练严格使用 train_test_split.json 里的 train_ids
    train_ids: Optional[List[str]] = None
    train_ids_json = cfg_train.get("train_ids_json") if isinstance(cfg_train, dict) else None
    if train_ids_json is not None:
        tids = Path(str(train_ids_json))
        if not tids.is_absolute():
            tids = here / tids
        tids = tids.resolve()
        sj = _load_config(tids)
        train_ids = list(sj.get("train_ids") or [])
        if rank == 0:
            print(f"[split] train_ids_json={tids} train_ids_count={len(train_ids)}", flush=True)

    ds = DatasetArch4(
        root,
        img_size=img_size,
        max_views=max_views,
        num_frames=num_frames,
        source="auto",
        preflight=preflight,
        verbose=(rank == 0),
        sample_ids=train_ids,
        input_mode="images",
    )

    train_dataset = ds
    if overfit_num_samples > 0:
        n_overfit = min(int(overfit_num_samples), len(ds))
        train_dataset = Subset(ds, list(range(n_overfit)))
        if rank == 0:
            print(
                f"[overfit] enabled: overfit_num_samples={overfit_num_samples} using={n_overfit}/{len(ds)}",
                flush=True,
            )

    sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(not overfit_no_shuffle),
            drop_last=False,
        )
        if distributed
        else None
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and (not overfit_no_shuffle)),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=str(device_str).startswith("cuda"),
        drop_last=False,
    )

    # 只把 json 里与模型 __init__ 参数匹配的字段传给 build_arch4_model
    from .arch4_model import Arch4VideoMAEPhysModel

    accepted = set(inspect.signature(Arch4VideoMAEPhysModel.__init__).parameters.keys()) - {"self"}
    model_kwargs = {k: v for k, v in cfg_model.items() if k in accepted}
    if args.head_dropout is not None:
        model_kwargs["head_dropout"] = float(args.head_dropout)
    # 这些关键字段用上面解析后的最终值，避免 config/cli 二义性
    model_kwargs.update(
        {
            "num_views": int(max_views),
            "in_channels": int(in_channels),
            "num_frames": int(num_frames),
            "img_size": int(img_size),
            "dec_h": int(dec_h),
            "dec_w": int(dec_w),
            "use_aux_field_heads": use_aux_field_heads,
        }
    )
    if rank == 0:
        print(
            f"[train_cfg] lambda_stress={float(lambda_stress):.4g} lambda_flow={float(lambda_flow):.4g} lambda_force={float(lambda_force):.4g} "
            f"overfit_no_shuffle={overfit_no_shuffle} num_workers={num_workers} "
            f"head_dropout={model_kwargs.get('head_dropout', 'default')} "
            f"disable_aux_losses={disable_aux_losses} weak_aux_losses={weak_aux_losses}",
            flush=True,
        )

    model = build_arch4_model(**model_kwargs)
    model.to(device)

    if distributed:
        # 输入消融/按 input_mode 关闭部分 loss 时，会出现“某些参数本轮未参与反传”。
        # DDP 默认要求每轮所有参数都有梯度，因此这里开启 find_unused_parameters 以避免报错。
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    loss_reg = Arch4RegressionLoss(Arch4LossConfig())

    # 注意：DDP 外壳不透传自定义属性（如 use_aux_field_heads）
    # 所以训练循环里用配置解析得到的布尔值而不是 model.use_aux_field_heads
    use_aux_field_heads_runtime = bool(use_aux_field_heads)

    # 新流程：输入仅 images，但输出始终监督 stress/flow/force_mask 三个时序场
    has_stress_input = True
    has_flow_input = True
    has_force_mask_input = True

    # checkpoint 配置（rank0 保存）
    cfg_ckpt = cfg_train.get("checkpoint") if isinstance(cfg_train, dict) else None
    cfg_ckpt = cfg_ckpt if isinstance(cfg_ckpt, dict) else {}
    save_every_epochs = int(cfg_ckpt.get("save_every_epochs", 1))
    save_dir_str = cfg_ckpt.get("save_dir")
    save_last = bool(cfg_ckpt.get("save_last", True))
    save_best = bool(cfg_ckpt.get("save_best", True))

    # quick eval 配置（rank0 调用 eval.py）
    cfg_quick_eval = cfg_train.get("quick_eval") if isinstance(cfg_train, dict) else None
    cfg_quick_eval = cfg_quick_eval if isinstance(cfg_quick_eval, dict) else {}
    quick_eval_enabled = bool(cfg_quick_eval.get("enabled", False))
    quick_eval_every_epochs = int(cfg_quick_eval.get("every_epochs", 100))
    quick_eval_weights = str(cfg_quick_eval.get("weights", "last"))
    quick_eval_batch_size = int(cfg_quick_eval.get("batch_size", 1))
    quick_eval_num_workers = int(cfg_quick_eval.get("num_workers", 0))
    quick_eval_num_vis = int(cfg_quick_eval.get("num_vis", 1))
    quick_eval_vis_view = int(cfg_quick_eval.get("vis_view", 0))
    quick_eval_split_root = cfg_quick_eval.get("split_root")
    quick_eval_train_ids_json = cfg_quick_eval.get("train_ids_json")
    quick_eval_test_ids_json = cfg_quick_eval.get("test_ids_json")
    quick_eval_sample_limit = int(cfg_quick_eval.get("sample_limit", 0))

    quick_eval_base_dir: Optional[Path] = None

    # resume
    resume_from = cfg_ckpt.get("resume_from")
    start_epoch = 0  # 0-based epoch index for range()
    resume_global_step = 0
    resume_best_avg_loss = float("inf")
    resume_model_state: Optional[Dict[str, Any]] = None
    resume_optimizer_state: Optional[Dict[str, Any]] = None
    if resume_from:
        resume_path = Path(str(resume_from))
        if not resume_path.is_absolute():
            resume_path = here / resume_path
        resume_path = resume_path.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"checkpoint resume_from not found: {resume_path}")

        ckpt_obj = torch.load(str(resume_path), map_location="cpu")
        if not isinstance(ckpt_obj, dict):
            raise ValueError(f"invalid checkpoint format: {resume_path}")

        start_epoch = int(ckpt_obj.get("epoch", 0))
        resume_global_step = int(ckpt_obj.get("global_step", 0))
        resume_best_avg_loss = float(ckpt_obj.get("avg_loss", float("inf")))

        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            resume_model_state = ckpt_obj["model"]
        else:
            # 兼容：若 checkpoint 直接保存的是 state_dict
            if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
                resume_model_state = ckpt_obj  # type: ignore[assignment]

        if "optimizer" in ckpt_obj and isinstance(ckpt_obj["optimizer"], dict):
            resume_optimizer_state = ckpt_obj["optimizer"]

        if resume_model_state is None:
            raise ValueError(f"checkpoint resume_from has no model weights: {resume_path}")

        if rank == 0:
            print(
                f"[resume] path={resume_path} start_epoch={start_epoch} global_step={resume_global_step} best_avg_loss={resume_best_avg_loss}",
                flush=True,
            )

    # resume：加载模型/optimizer 状态（使用 locals().get 避免偶发的未初始化变量问题）
    resume_model_state_rt = locals().get("resume_model_state", None)
    if resume_model_state_rt is not None:
        missing, unexpected = model.load_state_dict(resume_model_state_rt, strict=False)
        if rank == 0:
            print(
                f"[resume] model loaded missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

    resume_optimizer_state_rt = locals().get("resume_optimizer_state", None)
    if resume_optimizer_state_rt is not None:
        opt.load_state_dict(resume_optimizer_state_rt)
        if rank == 0:
            print("[resume] optimizer loaded", flush=True)

    if rank == 0:
        if save_dir_str:
            save_dir = Path(str(save_dir_str))
            if not save_dir.is_absolute():
                save_dir = here / save_dir
            save_dir = save_dir.resolve()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = (here / "output_checkpoints" / f"arch4_train_{ts}").resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        best_avg_loss = resume_best_avg_loss if resume_from else float("inf")

        tb_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_root = (here / "tb_logs" / f"arch4_train_{tb_ts}_epochs_{int(epochs)}").resolve()
        tb_root.mkdir(parents=True, exist_ok=True)

        if quick_eval_enabled:
            qt = datetime.now().strftime("%Y%m%d_%H%M%S")
            quick_eval_base_dir = (here / "output_eval" / f"arch4_quick_eval_{qt}").resolve()
            quick_eval_base_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = here / "output_checkpoints" / "arch4_train_unused"
        best_avg_loss = float("inf")
        tb_root = here / "tb_logs" / "arch4_train_unused"
        quick_eval_base_dir = None

    try:
        writer: Optional[Any] = None
        global_step = resume_global_step
        if rank == 0 and SummaryWriter is not None:
            writer = SummaryWriter(log_dir=str(tb_root))

        for epoch in range(int(start_epoch), int(epochs)):
            if stop_requested:
                break
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            total = 0.0
            n = 0
            total_reg = 0.0
            total_stress = 0.0
            total_flow = 0.0
            total_force = 0.0

            batch_iter = loader
            if rank == 0 and tqdm is not None:
                total_batches = len(loader) if hasattr(loader, "__len__") else None
                batch_iter = tqdm(
                    loader,
                    total=total_batches,
                    desc=f"epoch {epoch + 1}/{int(epochs)}",
                    dynamic_ncols=False,
                    ncols=240,
                    leave=True,
                    file=sys.stdout,
                    mininterval=1.0,
                )

            for bi, (x, stress_gt, flow_gt, force_gt, params_gt) in enumerate(batch_iter):
                if stop_requested:
                    break
                x = x.to(device)
                stress_gt = stress_gt.to(device)
                flow_gt = flow_gt.to(device)
                force_gt = force_gt.to(device)
                params_gt = params_gt.to(device)
                opt.zero_grad()
                out = model(x)
                pred = out["param_pred"]
                pred_raw_runtime = out.get("param_pred_raw")
                if pred_raw_runtime is None:
                    pred_raw_runtime = _target_to_raw_params(pred)

                # yield_stress=0 的材质不计入该维度回归损失
                valid_mask = torch.ones_like(pred)
                valid_mask[:, 3] = (params_gt[:, 3] > 0).to(valid_mask.dtype)
                gt_train_space = _to_target_params(params_gt)

                # Debug：打印回归每个参数的 raw/log GT 与预测，以及逐维误差与 loss（rank0）
                if (
                    rank == 0
                    and dbg_enabled
                    and dbg_printed < dbg_max_prints
                    and (global_step % dbg_every_steps == 0)
                ):
                    try:
                        si = min(int(dbg_sample_idx), int(pred.shape[0]) - 1)
                        pred_raw = pred_raw_runtime.detach()[si].float().cpu()
                        gt_raw = params_gt.detach()[si].float().cpu()
                        pred_log = pred.detach()[si].float().cpu()
                        gt_log = gt_train_space.detach()[si].float().cpu()

                        abs_raw = (pred_raw - gt_raw).abs()
                        abs_log = (pred_log - gt_log).abs()
                        mse_raw = (pred_raw - gt_raw).pow(2)
                        mse_log = (pred_log - gt_log).pow(2)

                        # SmoothL1（与 Arch4RegressionLoss 对齐，用于逐维展示；不含可选 target_weights/异方差项）
                        beta = 1.0
                        try:
                            beta = float(getattr(loss_reg, "cfg", None).smoothl1_beta)  # type: ignore[attr-defined]
                        except Exception:
                            beta = 1.0
                        sl1_raw = F.smooth_l1_loss(pred_raw, gt_raw, beta=beta, reduction="none")
                        sl1_log = F.smooth_l1_loss(pred_log, gt_log, beta=beta, reduction="none")

                        names = ["E", "nu", "density", "yield_stress"]
                        print(
                            "\n".join(
                                [
                                    f"[reg_debug] step={global_step} sample_idx={si} no_log_scale={no_log_scale} beta={beta} yield_valid={bool(valid_mask[si,3].item() > 0)}",
                                    "  raw_pred: " + ", ".join([f"{names[i]}={pred_raw[i].item():.6g}" for i in range(4)]),
                                    "  raw_gt  : " + ", ".join([f"{names[i]}={gt_raw[i].item():.6g}" for i in range(4)]),
                                    "  log_pred: " + ", ".join([f"{names[i]}={pred_log[i].item():.6g}" for i in range(4)]),
                                    "  log_gt  : " + ", ".join([f"{names[i]}={gt_log[i].item():.6g}" for i in range(4)]),
                                    "  abs_raw : " + ", ".join([f"{names[i]}={abs_raw[i].item():.6g}" for i in range(4)]),
                                    "  abs_log : " + ", ".join([f"{names[i]}={abs_log[i].item():.6g}" for i in range(4)]),
                                    "  mse_raw : " + ", ".join([f"{names[i]}={mse_raw[i].item():.6g}" for i in range(4)]),
                                    "  mse_log : " + ", ".join([f"{names[i]}={mse_log[i].item():.6g}" for i in range(4)]),
                                    "  sl1_raw : " + ", ".join([f"{names[i]}={sl1_raw[i].item():.6g}" for i in range(4)]),
                                    "  sl1_log : " + ", ".join([f"{names[i]}={sl1_log[i].item():.6g}" for i in range(4)]),
                                ]
                            ),
                            flush=True,
                        )
                        dbg_printed += 1
                    except Exception as e:
                        print(f"[reg_debug] failed to print: {e}", flush=True)
                if not no_log_scale:
                    # 模型直接输出目标空间 [logE, nu, logDensity, logYield]
                    loss_p = loss_reg(pred, gt_train_space, out["logvar"], valid_mask=valid_mask)
                else:
                    # 兼容：若要求在 raw 空间监督，则使用反变换后的 raw 预测
                    loss_p = loss_reg(pred_raw_runtime, params_gt, out["logvar"], valid_mask=valid_mask)

                # 主损失：参数回归
                loss_reg_part = loss_p

                # 辅助损失：场预测
                loss_stress_part = (
                    arch4_field_supervision_mse(out["stress_field_pred"], stress_gt, dec_h, dec_w)
                    if (use_aux_field_heads_runtime and has_stress_input)
                    else loss_reg_part.new_tensor(0.0)
                )
                loss_flow_part = (
                    arch4_field_supervision_mse(out["flow_field_pred"], flow_gt, dec_h, dec_w)
                    if (use_aux_field_heads_runtime and has_flow_input)
                    else loss_reg_part.new_tensor(0.0)
                )
                loss_force_part = (
                    arch4_field_supervision_mse(out["force_pred"], force_gt, dec_h, dec_w)
                    if (use_aux_field_heads_runtime and has_force_mask_input)
                    else loss_reg_part.new_tensor(0.0)
                )

                # 总损失 = 主损失 + 加权辅助损失
                loss_total = (
                    loss_reg_part
                    + float(lambda_stress) * loss_stress_part
                    + float(lambda_flow) * loss_flow_part
                    + float(lambda_force) * loss_force_part
                )

                if rank == 0 and debug_overfit and (global_step % debug_overfit_every_steps == 0):
                    si = 0
                    raw_pred_dbg = pred_raw_runtime.detach()[si].float().cpu()
                    raw_gt_dbg = params_gt.detach()[si].float().cpu()
                    train_pred_dbg = pred.detach()[si].float().cpu()
                    train_gt_dbg = gt_train_space.detach()[si].float().cpu()
                    abs_raw_dbg = (raw_pred_dbg - raw_gt_dbg).abs()
                    abs_train_dbg = (train_pred_dbg - train_gt_dbg).abs()
                    print(
                        "\n".join(
                            [
                                f"[debug_overfit] step={global_step} sample_idx={si}",
                                f"  raw_pred={raw_pred_dbg.tolist()}",
                                f"  raw_gt={raw_gt_dbg.tolist()}",
                                f"  abs_raw={abs_raw_dbg.tolist()}",
                                f"  pred_train_space={train_pred_dbg.tolist()}",
                                f"  gt_train_space={train_gt_dbg.tolist()}",
                                f"  abs_train_space={abs_train_dbg.tolist()}",
                                (
                                    "  losses: "
                                    f"loss_total={float(loss_total.item()):.6g}, "
                                    f"loss_reg={float(loss_reg_part.item()):.6g}, "
                                    f"loss_stress={float(loss_stress_part.item()):.6g}, "
                                    f"loss_flow={float(loss_flow_part.item()):.6g}, "
                                    f"loss_force={float(loss_force_part.item()):.6g}"
                                ),
                            ]
                        ),
                        flush=True,
                    )

                # tqdm 显示主要信息
                if rank == 0 and tqdm is not None and hasattr(batch_iter, "set_postfix"):
                    # 显示与 total 一致的“加权分项”（避免 flow 因四舍五入显示为 0）
                    total_v = float(loss_total.item())
                    reg_v = float(loss_reg_part.item())
                    stress_v = float(lambda_stress) * float(loss_stress_part.item())
                    flow_v = float(lambda_flow) * float(loss_flow_part.item())
                    force_v = float(lambda_force) * float(loss_force_part.item())
                    batch_iter.set_postfix(
                        {
                            "total": f"{total_v:.4e}",
                            "reg": f"{reg_v:.4e}",
                            "stress": f"{stress_v:.4e}",
                            "flow": f"{flow_v:.4e}",
                            "force_mask": f"{force_v:.4e}",
                        }
                    )

                loss_total.backward()

                if rank == 0 and debug_overfit and (global_step % debug_overfit_every_steps == 0):
                    model_ref = model.module if isinstance(model, DDP) else model
                    backbone_last = None
                    try:
                        blocks = getattr(getattr(model_ref, "encoder", None), "blocks", None)
                        if blocks is not None and len(blocks) > 0:
                            backbone_last = blocks[-1]
                    except Exception:
                        backbone_last = None
                    fusion_mod = getattr(model_ref, "fusion", None)
                    param_head_mod = getattr(model_ref, "param_head", None)
                    stress_head_mod = getattr(model_ref, "stress_head", None)
                    flow_head_mod = getattr(model_ref, "flow_head", None)
                    force_head_mod = getattr(model_ref, "force_head", None)
                    lr_cur = float(opt.param_groups[0]["lr"]) if len(opt.param_groups) > 0 else 0.0
                    print(
                        "\n".join(
                            [
                                f"  grad_norm/backbone={_module_grad_norm(backbone_last):.6g}",
                                f"  grad_norm/fusion={_module_grad_norm(fusion_mod):.6g}",
                                f"  grad_norm/param_head={_module_grad_norm(param_head_mod):.6g}",
                                f"  grad_norm/stress_head={_module_grad_norm(stress_head_mod):.6g}",
                                f"  grad_norm/flow_head={_module_grad_norm(flow_head_mod):.6g}",
                                f"  grad_norm/force_head={_module_grad_norm(force_head_mod):.6g}",
                                f"  lr={lr_cur:.6g}",
                            ]
                        ),
                        flush=True,
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                bsz = x.size(0)
                total += loss_total.item() * bsz
                n += bsz
                total_reg += loss_reg_part.item() * bsz
                total_stress += loss_stress_part.item() * bsz
                total_flow += loss_flow_part.item() * bsz
                total_force += loss_force_part.item() * bsz

                if writer is not None:
                    writer.add_scalar("train/batch_loss", float(loss_total.item()), global_step)
                    writer.add_scalar("train/batch_loss_reg", float(loss_reg_part.item()), global_step)
                    writer.add_scalar("train/batch_loss_stress", float(loss_stress_part.item()), global_step)
                    writer.add_scalar("train/batch_loss_flow", float(loss_flow_part.item()), global_step)
                    writer.add_scalar("train/batch_loss_force", float(loss_force_part.item()), global_step)
                global_step += 1

            if stop_requested:
                # 直接跳过本 epoch 的 all-reduce/barrier/ckpt/quick_eval
                break
            # 汇总全卡 epoch loss
            if distributed:
                t_total = torch.tensor([total], device=device, dtype=torch.float64)
                t_n = torch.tensor([n], device=device, dtype=torch.float64)
                t_reg = torch.tensor([total_reg], device=device, dtype=torch.float64)
                t_stress = torch.tensor([total_stress], device=device, dtype=torch.float64)
                t_flow = torch.tensor([total_flow], device=device, dtype=torch.float64)
                t_force = torch.tensor([total_force], device=device, dtype=torch.float64)
                dist.all_reduce(t_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(t_n, op=dist.ReduceOp.SUM)
                dist.all_reduce(t_reg, op=dist.ReduceOp.SUM)
                dist.all_reduce(t_stress, op=dist.ReduceOp.SUM)
                dist.all_reduce(t_flow, op=dist.ReduceOp.SUM)
                dist.all_reduce(t_force, op=dist.ReduceOp.SUM)
                total_g = float(t_total.item())
                n_g = float(t_n.item())
                total_reg_g = float(t_reg.item())
                total_stress_g = float(t_stress.item())
                total_flow_g = float(t_flow.item())
                total_force_g = float(t_force.item())
            else:
                total_g = float(total)
                n_g = float(n)
                total_reg_g = float(total_reg)
                total_stress_g = float(total_stress)
                total_flow_g = float(total_flow)
                total_force_g = float(total_force)

            if rank == 0:
                print(f"epoch {epoch + 1}/{epochs} loss={total_g / max(n_g, 1.0):.6f}", flush=True)

                avg_loss = total_g / max(n_g, 1.0)
                avg_loss_reg = total_reg_g / max(n_g, 1.0)
                avg_loss_stress = total_stress_g / max(n_g, 1.0)
                avg_loss_flow = total_flow_g / max(n_g, 1.0)
                avg_loss_force = total_force_g / max(n_g, 1.0)

                if writer is not None:
                    writer.add_scalar("train/avg_loss", float(avg_loss), epoch)
                    writer.add_scalar("train/avg_loss_reg", float(avg_loss_reg), epoch)
                    writer.add_scalar("train/avg_loss_stress", float(avg_loss_stress), epoch)
                    writer.add_scalar("train/avg_loss_flow", float(avg_loss_flow), epoch)
                    writer.add_scalar("train/avg_loss_force", float(avg_loss_force), epoch)
                    writer.flush()
                # 保存 checkpoint（只在 rank0 写盘）
                to_save_model = model.module if distributed else model
                if save_every_epochs > 0 and ((epoch + 1) % int(save_every_epochs) == 0):
                    ckpt_path = save_dir / f"epoch_{epoch + 1:04d}.pt"
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "avg_loss": avg_loss,
                            "global_step": global_step,
                            "model": to_save_model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "model_kwargs": model_kwargs,
                            "config": cfg,
                        },
                        str(ckpt_path),
                    )

                if save_last:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "avg_loss": avg_loss,
                            "global_step": global_step,
                            "model": to_save_model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "model_kwargs": model_kwargs,
                            "config": cfg,
                        },
                        str(save_dir / "last.pt"),
                    )

                if save_best and avg_loss < best_avg_loss:
                    best_avg_loss = avg_loss
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "avg_loss": avg_loss,
                            "global_step": global_step,
                            "model": to_save_model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "model_kwargs": model_kwargs,
                            "config": cfg,
                        },
                        str(save_dir / "best.pt"),
                    )

                # quick eval：每 N 个 epoch 由 rank0 调用 eval.py
                if (
                    (not stop_requested)
                    and quick_eval_enabled
                    and quick_eval_every_epochs > 0
                    and ((epoch + 1) % quick_eval_every_epochs == 0)
                ):
                    assert quick_eval_base_dir is not None  # rank0 且 enabled 时应创建

                    weights_eval_path: Optional[Path] = None
                    if quick_eval_weights == "epoch":
                        p = save_dir / f"epoch_{epoch + 1:04d}.pt"
                        if p.is_file():
                            weights_eval_path = p
                    elif quick_eval_weights == "best":
                        p = save_dir / "best.pt"
                        if p.is_file():
                            weights_eval_path = p
                    else:
                        # 默认 last
                        p = save_dir / "last.pt"
                        if p.is_file():
                            weights_eval_path = p

                    # 兜底：last 优先
                    if weights_eval_path is None:
                        p = save_dir / "last.pt"
                        if p.is_file():
                            weights_eval_path = p

                    if weights_eval_path is None:
                        print("[quick_eval] skip: no suitable checkpoint found", flush=True)
                    else:
                        epoch_tag = epoch + 1
                        cmd_common = [
                            sys.executable,
                            "-m",
                            "my_model.eval",
                            "--config",
                            str(cfg_path_resolved),
                            "--weights",
                            str(weights_eval_path),
                            "--batch_size",
                            str(quick_eval_batch_size),
                            "--num_workers",
                            str(quick_eval_num_workers),
                            "--num_vis",
                            str(quick_eval_num_vis),
                            "--vis_view",
                            str(quick_eval_vis_view),
                        ]
                        if quick_eval_split_root is not None:
                            cmd_common += ["--split_root", str(quick_eval_split_root)]
                            # quick_eval_split_root 若是相对路径（含 auto_output/...），这里透传同一父目录名
                            auto_out = cfg_data.get("auto_output", "auto_output")
                            cmd_common += ["--auto_output", str(auto_out)]

                        ran_any = False

                        # quick eval 分别在 train_ids 与 test_ids 上跑
                        if quick_eval_train_ids_json is not None:
                            train_ids_p = Path(str(quick_eval_train_ids_json))
                            if not train_ids_p.is_absolute():
                                train_ids_p = here / train_ids_p
                            out_dir_train = quick_eval_base_dir / "train" / f"epoch_{epoch_tag:04d}"
                            cmd_train = cmd_common + [
                                "--out_dir",
                                str(out_dir_train),
                                "--sample_ids_json",
                                str(train_ids_p.resolve()),
                                "--sample_ids_key",
                                "train_ids",
                            ]
                            if quick_eval_sample_limit > 0:
                                cmd_train += ["--sample_limit", str(quick_eval_sample_limit)]
                            print(
                                f"[quick_eval] train epoch={epoch_tag} weights={weights_eval_path.name} -> {out_dir_train}",
                                flush=True,
                            )
                            env = os.environ.copy()
                            env["WORLD_SIZE"] = "1"
                            env["RANK"] = "0"
                            env["LOCAL_RANK"] = "0"
                            env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
                            try:
                                subprocess.run(cmd_train, env=env, check=True, cwd=str(here))
                            except Exception as e:
                                print(f"[quick_eval] train failed: {e}", flush=True)
                            ran_any = True

                        if quick_eval_test_ids_json is not None:
                            test_ids_p = Path(str(quick_eval_test_ids_json))
                            if not test_ids_p.is_absolute():
                                test_ids_p = here / test_ids_p
                            out_dir_test = quick_eval_base_dir / "test" / f"epoch_{epoch_tag:04d}"
                            cmd_test = cmd_common + [
                                "--out_dir",
                                str(out_dir_test),
                                "--sample_ids_json",
                                str(test_ids_p.resolve()),
                                "--sample_ids_key",
                                "test_ids",
                            ]
                            if quick_eval_sample_limit > 0:
                                cmd_test += ["--sample_limit", str(quick_eval_sample_limit)]
                            print(
                                f"[quick_eval] test epoch={epoch_tag} weights={weights_eval_path.name} -> {out_dir_test}",
                                flush=True,
                            )
                            env = os.environ.copy()
                            env["WORLD_SIZE"] = "1"
                            env["RANK"] = "0"
                            env["LOCAL_RANK"] = "0"
                            env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
                            try:
                                subprocess.run(cmd_test, env=env, check=True, cwd=str(here))
                            except Exception as e:
                                print(f"[quick_eval] test failed: {e}", flush=True)
                            ran_any = True

                        if not ran_any:
                            print("[quick_eval] skip: train_ids_json/test_ids_json 未配置", flush=True)

            # 中断时不要 barrier（最容易卡死）
            if distributed and (not stop_requested):
                dist.barrier()
    except KeyboardInterrupt:
        # 某些情况下 SIGINT 不会进入 signal handler（例如在部分阻塞调用里），这里兜底
        stop_requested = True
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        if rank == 0 and writer is not None:
            writer.close()

    # 让 torchrun 感知中断并清理所有 worker
    if stop_requested:
        # 所有 rank 都以非0退出码退出，torchrun 会立即终止整个作业
        raise SystemExit(130)


if __name__ == "__main__":
    main()
