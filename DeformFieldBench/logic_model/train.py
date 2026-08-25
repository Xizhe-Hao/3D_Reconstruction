from __future__ import annotations

"""
单机多卡训练（DDP）示例：在 **Phys 仓库根目录** 执行，配置文件在 ``logic_model/configs/`` 下::

    torchrun --nproc_per_node=8 -m logic_model.train --config logic_model/configs/logic_train_dataset_mask_1000.json

单卡::

    python -m logic_model.train --config logic_model/configs/logic_train_dataset_mask_1000.json

若在 ``Phys/logic_model`` 目录下执行，可用 ``--config configs/logic_train_dataset_mask_1000.json``。

**Eval 与多卡**：eval 仅计算 loss（总 loss 与各分项），不写图或视频。默认仅 **rank0** 跑 eval，其它 rank 在 ``dist.barrier()`` 等待；若设 ``train.eval_use_distributed_sampler: true``，则各卡分片前向后 ``all_reduce`` 聚合指标。

**AMP**：在 **CUDA** 上默认启用 **fp16**（``torch.cuda.amp.autocast`` + ``GradScaler``）。关闭：CLI ``--no_amp`` 或 JSON ``"use_amp": false``。

**DataLoader（CUDA）**：``pin_memory=True``、``.to(..., non_blocking=True)``；若 ``train.num_workers`` > 0 则 ``persistent_workers=True``，``prefetch_factor`` 见 ``train.prefetch_factor``（默认 2，且至少为 2）。
"""

import argparse
from collections import deque
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from logic_model.dataset import LmdbGtDataset, _SharedSampleCache, collate_lmdb_gt_batch
from logic_model.losses import (
    Arch4LossConfig,
    Arch4RegressionLoss,
    PhysicsConsistencyConfig,
    action_classification_loss,
    binary_dice_loss_with_logits,
    compute_physics_consistency_losses,
    weighted_bce_with_logits_loss,
    weighted_edge_l1_loss_bvcthw,
)
from logic_model.model import LogicPhysModel
from logic_model.model2 import LogicPhysModel2


def _pick(d: Dict[str, Any], k: str, default: Any) -> Any:
    v = d.get(k, default) if isinstance(d, dict) else default
    return default if v is None else v


def _amp_autocast(*, use_amp: bool, device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16, enabled=bool(use_amp))
    return torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=False)


def _cli_or_pick_cfg(arg_val: Any, cfg: Dict[str, Any], key: str, fallback: Any) -> Any:
    """
    配置读取优先级（与仅用 argparse 默认值冲突时以本函数为准）：
    1) 命令行显式传入（非 None）→ 用命令行
    2) 否则若 ``cfg[key]`` 存在且非 None → 用配置
    3) 否则 ``fallback``

    说明：若 CLI 某项 ``default`` 为非 None 的常数，则无法区分「用户未传」与「用户传了默认值」；
    因此对依赖 JSON 配置的项，CLI 默认改为 None，未传时才读 config。
    """
    if arg_val is not None:
        return arg_val
    return _pick(cfg, key, fallback)


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be json object: {path}")
    return cfg


def _resolve_path_relative_to_config(rel: str, config_path: str | None) -> Path:
    """
    相对路径解析顺序：
    1) 相对 **配置文件所在目录**（如 ``logic_model/configs/*.json`` 旁的兄弟路径）；
    2) 相对 **当前工作目录**（便于 ``data.split_root`` 写 ``auto_output/...`` 时在 Phys 根目录运行）。
    """
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p.resolve()
    if config_path:
        cand = (Path(config_path).resolve().parent / p).resolve()
        if cand.exists():
            return cand
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return cwd_p
    return cwd_p


def _load_split_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"split json 顶层必须是 object: {path}")
    return raw


def _default_project_torchhub_dir() -> Path:
    # Phys/logic_model/train.py -> Phys/.torch/hub
    return (Path(__file__).resolve().parent.parent / ".torch" / "hub").resolve()


def _resolve_torchhub_dir(cfg_model: Dict[str, Any]) -> Path:
    cfg_dir = str(_pick(cfg_model, "torchhub_dir", "") or "").strip()
    if cfg_dir:
        p = Path(cfg_dir).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (Path.cwd() / p).resolve()
    env_home = str(os.environ.get("TORCH_HOME", "") or "").strip()
    if env_home:
        return (Path(env_home).expanduser().resolve() / "hub").resolve()
    return _default_project_torchhub_dir()


def _configure_torchhub_dir(hub_dir: Path) -> Path:
    hub_dir = Path(hub_dir).expanduser().resolve()
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(hub_dir.parent)
    torch.hub.set_dir(str(hub_dir))
    return hub_dir


def _expected_torchhub_repo_dir(repo: str) -> str:
    # torch.hub: "owner/name:ref" -> "owner_name_ref"
    s = str(repo).replace(":", "_").replace("/", "_")
    return s


def _expected_dino_ckpt_basename(backbone_name: str) -> str:
    name = str(backbone_name).strip().lower()
    table = {
        "dinov2_vits14": "dinov2_vits14_pretrain.pth",
        "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
        "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
        "dinov2_vitg14": "dinov2_vitg14_pretrain.pth",
        "dinov2_vits14_reg": "dinov2_vits14_reg4_pretrain.pth",
        "dinov2_vitb14_reg": "dinov2_vitb14_reg4_pretrain.pth",
        "dinov2_vitl14_reg": "dinov2_vitl14_reg4_pretrain.pth",
        "dinov2_vitg14_reg": "dinov2_vitg14_reg4_pretrain.pth",
    }
    return table.get(name, "")


def _prewarm_torchhub_dino(cfg_model: Dict[str, Any], hub_dir: Path) -> None:
    """
    仅做本地缓存检查，不触发在线下载。
    """
    source = str(_pick(cfg_model, "dino_backbone_source", "torchhub")).lower().strip()
    if source not in ("torchhub", "hub"):
        return
    repo = str(_pick(cfg_model, "dino_torchhub_repo", "facebookresearch/dinov2:main"))
    repo_dir = hub_dir / _expected_torchhub_repo_dir(repo)
    if not repo_dir.is_dir():
        raise FileNotFoundError(
            f"torchhub repo 缓存缺失: {repo_dir}。请先手动准备本地 hub 缓存（不走在线下载）。"
        )

    if bool(_pick(cfg_model, "dino_backbone_pretrained", True)):
        ckpt_name = _expected_dino_ckpt_basename(str(_pick(cfg_model, "dino_backbone_name", "dinov2_vits14")))
        if ckpt_name:
            ckpt_path = hub_dir / "checkpoints" / ckpt_name
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"DINO checkpoint 缓存缺失: {ckpt_path}。请手动下载到 Phys/.torch/hub/checkpoints。"
                )


def _init_distributed() -> Tuple[bool, int, int, int]:
    """仅解析 WORLD_SIZE/RANK/LOCAL_RANK，返回 (distributed, rank, local_rank, world_size)。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return True, rank, local_rank, world_size


def _startup_phase_log(
    phase: str,
    status: str,
    *,
    rank: Optional[int] = None,
    extra: str = "",
) -> None:
    return


def _safe_barrier(local_rank: Optional[int] = None) -> None:
    """
    分布式同步的安全包装：
    - 未初始化/单卡时直接返回
    - CUDA 多卡时显式传 device_ids，避免 NCCL barrier 设备推断异常
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    backend = str(dist.get_backend())
    if backend == "nccl":
        # 某些环境的 NCCL barrier 会触发 CUDA invalid argument；
        # 用 1 元 all_reduce 作为等价同步更稳。
        if not torch.cuda.is_available():
            dist.barrier()
            return
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        t = torch.zeros(1, device=dev, dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return
    dist.barrier()


def _distributed_all_true(flag: bool, device: torch.device) -> tuple[bool, int]:
    """
    多卡一致性检查：
    - 返回 (all_true, true_count)
    - 单卡/未初始化时直接返回当前结果
    """
    local_true = 1 if bool(flag) else 0
    if not dist.is_available() or not dist.is_initialized():
        return bool(local_true), int(local_true)
    t = torch.tensor([local_true], device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    true_count = int(t.item())
    world = int(dist.get_world_size())
    return true_count == world, true_count


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def _prebuild_shared_memmap_cache_rank0(
    *,
    split_root: str,
    lmdb_env_subdir: str,
    sample_pack_name: str,
    sample_storage_backend: str,
    allow_missing_sample_ids: bool,
    purify_force_mask_on_read: bool,
    force_mask_purify_mode: str,
    force_mask_keep_three_channels: bool,
    force_mask_single_channel: bool,
    force_mask_binary_threshold: float,
    max_views: int,
    num_frames: int,
    img_size: int,
    sample_ids: Optional[List[str]],
    cache_root: Any,
    cache_version: str,
    cache_force_rebuild: bool,
    cache_log: bool,
    prebuild_num_workers: int,
) -> None:
    ds_pre = LmdbGtDataset(
        split_root=split_root,
        lmdb_env_subdir=lmdb_env_subdir,
        sample_pack_name=sample_pack_name,
        sample_storage_backend=sample_storage_backend,
        sample_pack_deep_validate=False,
        allow_missing_sample_ids=allow_missing_sample_ids,
        purify_force_mask_on_read=purify_force_mask_on_read,
        force_mask_purify_mode=force_mask_purify_mode,
        force_mask_keep_three_channels=force_mask_keep_three_channels,
        force_mask_single_channel=force_mask_single_channel,
        force_mask_binary_threshold=force_mask_binary_threshold,
        max_views=max_views,
        num_frames=(None if num_frames <= 0 else num_frames),
        img_size=(None if img_size <= 0 else img_size),
        return_action_name=True,
        sample_ids=sample_ids,
        cache_backend="shared_memmap",
        cache_root=cache_root,
        cache_version=cache_version,
        cache_readonly=False,
        cache_force_rebuild=bool(cache_force_rebuild),
        cache_log=bool(cache_log),
    )
    n_all = len(ds_pre)
    t0 = time.perf_counter()
    # 先做完整性快检：若已全量 READY，直接跳过 prebuild。
    if (not bool(cache_force_rebuild)) and (ds_pre._shared_cache is not None):
        cache_obj = ds_pre._shared_cache
        ready_cnt = 0
        nf = ds_pre.num_frames
        for d in ds_pre.samples:
            payload = {
                "sample_id": d.name,
                "sample_path": str(d),
                "sample_storage_backend": ds_pre._storage_backend_by_sample.get(d.name, "lmdb"),
                "lmdb_env_subdir": ds_pre.lmdb_env_subdir,
                "sample_pack_name": ds_pre.sample_pack_name,
                "max_views": int(ds_pre.max_views),
                "num_frames": (None if nf is None or int(nf) <= 0 else int(nf)),
                "img_size": (None if ds_pre.img_size is None or int(ds_pre.img_size) <= 0 else int(ds_pre.img_size)),
                "cache_version": ds_pre.cache_version,
                "purify_force_mask_on_read": bool(ds_pre.purify_force_mask_on_read),
                "force_mask_purify_mode": str(ds_pre.force_mask_purify_mode),
                "force_mask_keep_three_channels": bool(ds_pre.force_mask_keep_three_channels),
                "force_mask_single_channel": bool(ds_pre.force_mask_single_channel),
                "force_mask_binary_threshold": float(ds_pre.force_mask_binary_threshold),
            }
            skey = _SharedSampleCache.build_cache_key(payload)
            sdir = cache_obj._sample_cache_dir(skey)
            if cache_obj._is_ready_complete(sdir):
                ready_cnt += 1
        if ready_cnt == n_all:
            return
    _pre_dl_extras: Dict[str, Any] = {}
    if int(prebuild_num_workers) > 0:
        _pre_dl_extras["persistent_workers"] = True
        _pre_dl_extras["prefetch_factor"] = 2
    pre_loader = DataLoader(
        ds_pre,
        batch_size=1,
        shuffle=False,
        num_workers=int(prebuild_num_workers),
        collate_fn=collate_lmdb_gt_batch,
        drop_last=False,
        pin_memory=False,
        **_pre_dl_extras,
    )
    pbar = tqdm(
        pre_loader,
        total=n_all,
        desc="prebuild-cache",
        leave=False,
        dynamic_ncols=True,
    )
    for _ in pbar:
        pass
    elapsed = time.perf_counter() - t0
    _ = elapsed


def _to_target_params(p: torch.Tensor) -> torch.Tensor:
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    e = torch.log1p(torch.clamp(p[:, 0], min=0))
    nu = p[:, 1]
    density = torch.log1p(torch.clamp(p[:, 2], min=0))
    yield_stress = torch.log1p(torch.clamp(p[:, 3], min=0))
    return torch.stack([e, nu, density, yield_stress], dim=1)


def _sanitize_raw_params_and_valid_mask(raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    清洗原始参数并构造逐目标 valid_mask，避免 NaN/Inf 污染回归损失。
    raw: [B,4] -> [E, nu, density, yield_stress]
    """
    p = raw.float()
    finite = torch.isfinite(p)
    p_safe = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    valid = finite.to(dtype=p_safe.dtype)
    valid[:, 0] = valid[:, 0] * (p_safe[:, 0] > 0).to(dtype=p_safe.dtype)
    valid[:, 1] = valid[:, 1] * (p_safe[:, 1] > 0).to(dtype=p_safe.dtype) * (p_safe[:, 1] <= 0.5).to(dtype=p_safe.dtype)
    valid[:, 2] = valid[:, 2] * (p_safe[:, 2] > 0).to(dtype=p_safe.dtype)
    valid[:, 3] = valid[:, 3] * (p_safe[:, 3] > 0).to(dtype=p_safe.dtype)
    return p_safe, valid


_REG_TARGET_NAMES = ("logE", "nu", "logDensity", "logYield")
_REG_MAE_SUM_START = 12
_REG_COUNT_START = 16
_REG_EDGE_STRESS = 20
_REG_EDGE_FLOW = 21
_REG_STRESS_RECON = 22
_REG_FLOW_RECON = 23
_REG_FORCE_BCE = 24
_REG_FORCE_DICE = 25
_REG_SUMS_LEN = 26


def _extract_raw_params_from_sample(sample: Dict[str, Any]) -> List[float]:
    params = sample.get("params")
    if isinstance(params, torch.Tensor):
        flat = params.detach().cpu().view(-1).tolist()
        return [float(x) for x in flat[:4]]
    if isinstance(params, (list, tuple)):
        return [float(x) for x in list(params)[:4]]
    raise ValueError("sample 缺少可读 params")


def _collect_raw_param_rows(data_source: Any) -> List[List[float]]:
    if isinstance(data_source, Subset):
        base = data_source.dataset
        indices = [int(i) for i in data_source.indices]
        if isinstance(base, LmdbGtDataset):
            rows: List[List[float]] = []
            for idx in indices:
                sample_dir = base.samples[int(idx)]
                gt = base._gt_by_sample[sample_dir.name]
                rows.append(
                    [
                        float(gt["E"]),
                        float(gt["nu"]),
                        float(gt["density"]),
                        float(gt["yield_stress"]),
                    ]
                )
            return rows
        return [_extract_raw_params_from_sample(base[int(i)]) for i in indices]
    if isinstance(data_source, LmdbGtDataset):
        rows = []
        for sample_dir in data_source.samples:
            gt = data_source._gt_by_sample[sample_dir.name]
            rows.append(
                [
                    float(gt["E"]),
                    float(gt["nu"]),
                    float(gt["density"]),
                    float(gt["yield_stress"]),
                ]
            )
        return rows
    return [_extract_raw_params_from_sample(data_source[int(i)]) for i in range(len(data_source))]


def _build_regression_target_stats(data_source: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = _collect_raw_param_rows(data_source)
    if not rows:
        raise ValueError("无法为回归目标计算标准化统计：训练集为空")
    raw = torch.tensor(rows, dtype=torch.float32)
    raw_safe, valid_mask = _sanitize_raw_params_and_valid_mask(raw)
    target = _to_target_params(raw_safe)

    mean = torch.zeros(target.shape[1], dtype=torch.float32)
    std = torch.ones(target.shape[1], dtype=torch.float32)
    counts = valid_mask.sum(dim=0)
    for i in range(int(target.shape[1])):
        sel = valid_mask[:, i] > 0.5
        if bool(sel.any()):
            vals = target[sel, i]
            mean[i] = vals.mean()
            std_i = vals.std(unbiased=False)
            std[i] = torch.clamp(std_i, min=1e-6)
    return mean, std, counts


def _normalize_regression_targets(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (x - mean.to(device=x.device, dtype=x.dtype)) / std.to(device=x.device, dtype=x.dtype)


def _per_target_abs_error_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    err = (pred - target).abs()
    mask = valid_mask.to(device=err.device, dtype=err.dtype)
    return (err * mask).sum(dim=0), mask.sum(dim=0)


def _per_target_mae_from_sums(sums: List[float], idx: int) -> float:
    den = max(float(sums[_REG_COUNT_START + idx]), 1.0)
    return float(sums[_REG_MAE_SUM_START + idx]) / den


def _resample_time_bvcthw(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """
    x: [B,V,C,T,H,W] -> [B,V,C,target_t,H,W]
    """
    t0 = int(x.shape[3])
    if t0 == int(target_t):
        return x
    b, v, c, _, h, w = x.shape
    y = x.permute(0, 1, 2, 4, 5, 3).contiguous().view(b * v * c * h * w, 1, t0)
    y = torch.nn.functional.interpolate(y, size=int(target_t), mode="linear", align_corners=False)
    y = y.view(b, v, c, h, w, int(target_t)).permute(0, 1, 2, 5, 3, 4).contiguous()
    return y


class _TimingTracker:
    """轻量 timing 统计：均值 + 近期窗口均值 + EMA。"""

    def __init__(self, *, window_size: int = 50, ema_alpha: float = 0.1) -> None:
        self.window_size = max(1, int(window_size))
        self.ema_alpha = float(ema_alpha)
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.ema: Dict[str, float] = {}
        self.recent: Dict[str, Deque[float]] = {}
        self.startup_data_wait: Optional[float] = None

    def update(self, metrics: Dict[str, float], *, is_startup: bool = False) -> None:
        if is_startup and self.startup_data_wait is None and "data_wait" in metrics:
            self.startup_data_wait = float(metrics["data_wait"])
        for k, v in metrics.items():
            x = float(v)
            self.sums[k] = self.sums.get(k, 0.0) + x
            self.counts[k] = self.counts.get(k, 0) + 1
            if k not in self.ema:
                self.ema[k] = x
            else:
                self.ema[k] = self.ema_alpha * x + (1.0 - self.ema_alpha) * self.ema[k]
            if k not in self.recent:
                self.recent[k] = deque(maxlen=self.window_size)
            self.recent[k].append(x)

    def avg(self, k: str) -> float:
        c = max(1, int(self.counts.get(k, 0)))
        return float(self.sums.get(k, 0.0) / c)

    def recent_avg(self, k: str) -> float:
        q = self.recent.get(k)
        if not q:
            return 0.0
        return float(sum(q) / max(1, len(q)))


def _align_bvcthw_to_ref(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    src/ref: [B,V,C,T,H,W]，按 ref 的 [T,H,W] 对齐 src（时间+空间）。
    """
    if src.shape == ref.shape:
        return src
    if src.dim() != 6 or ref.dim() != 6:
        raise ValueError(f"expect 6D [B,V,C,T,H,W], got src={tuple(src.shape)} ref={tuple(ref.shape)}")
    b, v, c, ts, hs, ws = src.shape
    tr, hr, wr = int(ref.shape[3]), int(ref.shape[4]), int(ref.shape[5])
    x = src.view(b * v, c, ts, hs, ws)
    x = F.interpolate(x, size=(tr, hr, wr), mode="trilinear", align_corners=False)
    return x.view(b, v, c, tr, hr, wr)


def _object_mask_weighted_field_loss(
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor,
    *,
    fg_weight: float,
    bg_weight: float,
    bg_black: bool,
) -> torch.Tensor:
    """
    前景加权场监督:
    - 前景(由 object_mask 定义): 拟合 gt
    - 背景: 默认拟合黑色(0)，抑制背景噪声
    """
    pred = pred_bvcthw
    gt = _align_bvcthw_to_ref(gt_bvcthw, pred)
    obj = _align_bvcthw_to_ref(object_mask_bvcthw, pred)
    obj = obj.mean(dim=2, keepdim=True).clamp(0.0, 1.0)
    obj = obj.expand(-1, -1, int(pred.shape[2]), -1, -1, -1)

    err_fg = (pred - gt).pow(2)
    if bg_black:
        err_bg = pred.pow(2)
    else:
        err_bg = (pred - gt).pow(2)

    w_fg = float(max(0.0, fg_weight))
    w_bg = float(max(0.0, bg_weight))
    num = w_fg * (obj * err_fg).sum() + w_bg * ((1.0 - obj) * err_bg).sum()
    den = w_fg * obj.sum() + w_bg * (1.0 - obj).sum()
    if den <= 1e-12:
        return pred.sum() * 0.0
    return num / den


def _forward_model_batch(
    model: torch.nn.Module,
    mcore: torch.nn.Module,
    x: torch.Tensor,
    *,
    stage_name: str,
    object_gt: torch.Tensor,
    param_stress_gt: Optional[torch.Tensor] = None,
    param_flow_gt: Optional[torch.Tensor] = None,
    param_stress_use_gt: Optional[torch.Tensor] = None,
    param_flow_use_gt: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    前向包装：对 LogicPhysModel2 传入 ``object_mask``，在 stress 融合与参数 field 编码前乘 mask；
    旧架构因无该参数而回退。
    """
    if hasattr(mcore, "set_training_stage"):
        try:
            return model(
                x,
                stage=stage_name,
                object_mask=object_gt,
                param_stress_gt=param_stress_gt,
                param_flow_gt=param_flow_gt,
                param_stress_use_gt=param_stress_use_gt,
                param_flow_use_gt=param_flow_use_gt,
            )
        except TypeError:
            return model(x, stage=stage_name)
    try:
        return model(x, object_mask=object_gt)
    except TypeError:
        return model(x)


def _compose_total_loss(
    *,
    loss_reg_part: torch.Tensor,
    loss_stress_part: torch.Tensor,
    loss_flow_part: torch.Tensor,
    loss_force_part: torch.Tensor,
    loss_action_part: torch.Tensor,
    loss_phys_part: torch.Tensor,
    lambda_stress: float,
    lambda_flow: float,
    lambda_force: float,
    lambda_action: float,
    lambda_phys: float,
    use_reg_loss: bool,
    use_stress_loss: bool,
    use_flow_loss: bool,
    use_force_loss: bool,
    use_action_loss: bool,
    use_phys_loss: bool,
) -> torch.Tensor:
    reg_w = 1.0 if use_reg_loss else 0.0
    stress_w = float(lambda_stress) if use_stress_loss else 0.0
    flow_w = float(lambda_flow) if use_flow_loss else 0.0
    force_w = float(lambda_force) if use_force_loss else 0.0
    action_w = float(lambda_action) if use_action_loss else 0.0
    phys_w = float(lambda_phys) if use_phys_loss else 0.0
    return (
        reg_w * loss_reg_part
        + stress_w * loss_stress_part
        + flow_w * loss_flow_part
        + force_w * loss_force_part
        + action_w * loss_action_part
        + phys_w * loss_phys_part
    )


def _compute_eval_batch_losses(
    model: torch.nn.Module,
    ev_m: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    loss_reg: Arch4RegressionLoss,
    reg_target_mean: torch.Tensor,
    reg_target_std: torch.Tensor,
    phys_cfg: PhysicsConsistencyConfig,
    *,
    lambda_stress: float,
    lambda_flow: float,
    lambda_force: float,
    lambda_action: float,
    lambda_phys: float,
    use_reg_loss: bool,
    use_stress_loss: bool,
    use_flow_loss: bool,
    use_force_loss: bool,
    use_action_loss: bool,
    use_phys_loss: bool,
    stage_name: str = "joint",
    object_mask_fg_weight: float,
    object_mask_bg_weight: float,
    object_mask_bg_black: bool,
    lambda_stress_edge: float = 0.0,
    lambda_flow_edge: float = 0.0,
    edge_boost: float = 4.0,
    force_bce_weight: float = 0.7,
    force_dice_weight: float = 0.3,
    use_amp: bool = False,
    non_blocking: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """eval 前向一步，返回总 loss 与各分项（与训练时加权方式一致）。"""
    rgb = batch["rgb"].to(device, non_blocking=non_blocking)
    stress_gt = batch["stress"].to(device, non_blocking=non_blocking)
    flow_gt = batch["flow"].to(device, non_blocking=non_blocking)
    force_gt = batch["force_mask"].to(device, non_blocking=non_blocking)
    object_gt = batch["object_mask"].to(device, non_blocking=non_blocking)
    params_gt_raw = batch["params"].to(device, non_blocking=non_blocking)
    action_label = batch["action_label"].to(device, non_blocking=non_blocking)

    if int(ev_m.in_channels) == 1:
        x = rgb[:, :, :1, :, :, :]
    else:
        x = rgb[:, :, : int(ev_m.in_channels), :, :, :]
    if int(x.shape[3]) != int(ev_m.num_frames):
        x = _resample_time_bvcthw(x, int(ev_m.num_frames))
        stress_gt = _resample_time_bvcthw(stress_gt, int(ev_m.num_frames))
        flow_gt = _resample_time_bvcthw(flow_gt, int(ev_m.num_frames))
        force_gt = _resample_time_bvcthw(force_gt, int(ev_m.num_frames))
        object_gt = _resample_time_bvcthw(object_gt, int(ev_m.num_frames))

    _amp = _amp_autocast(use_amp=use_amp, device=device)
    with _amp:
        out = _forward_model_batch(model, ev_m, x, stage_name=stage_name, object_gt=object_gt)
    params_gt_safe, valid_mask = _sanitize_raw_params_and_valid_mask(params_gt_raw)
    gt_train_space = _to_target_params(params_gt_safe).float()
    pred_param = out["param_pred"].float()
    zero = out["stress_field_pred"].float().sum() * 0.0
    if use_reg_loss:
        pred_reg_std = _normalize_regression_targets(pred_param, reg_target_mean, reg_target_std)
        gt_reg_std = _normalize_regression_targets(gt_train_space, reg_target_mean, reg_target_std)
        reg_mae_sum, reg_count = _per_target_abs_error_sums(pred_param, gt_train_space, valid_mask)
        loss_reg_part = loss_reg(pred_reg_std, gt_reg_std, out["logvar"].float(), valid_mask=valid_mask)
    else:
        reg_mae_sum = torch.zeros(4, device=device, dtype=torch.float32)
        reg_count = torch.zeros(4, device=device, dtype=torch.float32)
        loss_reg_part = zero

    loss_stress_recon = (
        _object_mask_weighted_field_loss(
            out["stress_field_pred"].float(),
            stress_gt.float(),
            object_gt.float(),
            fg_weight=object_mask_fg_weight,
            bg_weight=object_mask_bg_weight,
            bg_black=object_mask_bg_black,
        )
        if use_stress_loss
        else zero
    )
    loss_stress_edge = (
        weighted_edge_l1_loss_bvcthw(
            out["stress_field_pred"].float(),
            stress_gt.float(),
            object_gt.float(),
            edge_boost=float(edge_boost),
            bg_weight=float(object_mask_bg_weight),
        )
        if use_stress_loss and float(lambda_stress_edge) > 0.0
        else zero
    )
    loss_stress_part = loss_stress_recon + float(lambda_stress_edge) * loss_stress_edge

    loss_flow_recon = (
        _object_mask_weighted_field_loss(
            out["flow_field_pred"].float(),
            flow_gt.float(),
            object_gt.float(),
            fg_weight=object_mask_fg_weight,
            bg_weight=object_mask_bg_weight,
            bg_black=object_mask_bg_black,
        )
        if use_flow_loss
        else zero
    )
    loss_flow_edge = (
        weighted_edge_l1_loss_bvcthw(
            out["flow_field_pred"].float(),
            flow_gt.float(),
            object_gt.float(),
            edge_boost=float(edge_boost),
            bg_weight=float(object_mask_bg_weight),
        )
        if use_flow_loss and float(lambda_flow_edge) > 0.0
        else zero
    )
    loss_flow_part = loss_flow_recon + float(lambda_flow_edge) * loss_flow_edge

    loss_force_bce = zero
    loss_force_dice = zero
    if use_force_loss:
        if "force_logits" in out:
            loss_force_bce = weighted_bce_with_logits_loss(
                out["force_logits"].float(),
                force_gt.float(),
                object_gt.float(),
                fg_weight=object_mask_fg_weight,
                bg_weight=object_mask_bg_weight,
                bg_black=object_mask_bg_black,
            )
            loss_force_dice = binary_dice_loss_with_logits(
                out["force_logits"].float(),
                force_gt.float(),
                object_gt.float(),
            )
            loss_force_part = float(force_bce_weight) * loss_force_bce + float(force_dice_weight) * loss_force_dice
        else:
            loss_force_part = _object_mask_weighted_field_loss(
                out["force_pred"].float(),
                force_gt.float(),
                object_gt.float(),
                fg_weight=object_mask_fg_weight,
                bg_weight=object_mask_bg_weight,
                bg_black=object_mask_bg_black,
            )
    else:
        loss_force_part = zero
    if use_action_loss:
        action_ret = action_classification_loss(out["action_logits"].float(), action_label)
    else:
        action_ret = {"loss_action": zero, "action_acc": zero}
    if use_phys_loss:
        phys_ret = compute_physics_consistency_losses(
            stress_pred=out["stress_field_pred"].float(),
            flow_pred=out["flow_field_pred"].float(),
            force_pred=out["force_pred"].float(),
            force_gt=force_gt.float(),
            cfg=phys_cfg,
        )
    else:
        phys_ret = {
            "loss_stress_flow_consistency": zero,
            "loss_force_stress_consistency": zero,
            "loss_force_flow_consistency": zero,
            "loss_phys_total": zero,
        }
    loss_total = _compose_total_loss(
        loss_reg_part=loss_reg_part,
        loss_stress_part=loss_stress_part,
        loss_flow_part=loss_flow_part,
        loss_force_part=loss_force_part,
        loss_action_part=action_ret["loss_action"],
        loss_phys_part=phys_ret["loss_phys_total"],
        lambda_stress=lambda_stress,
        lambda_flow=lambda_flow,
        lambda_force=lambda_force,
        lambda_action=lambda_action,
        lambda_phys=lambda_phys,
        use_reg_loss=use_reg_loss,
        use_stress_loss=use_stress_loss,
        use_flow_loss=use_flow_loss,
        use_force_loss=use_force_loss,
        use_action_loss=use_action_loss,
        use_phys_loss=use_phys_loss,
    )
    parts = {
        "loss_reg": loss_reg_part,
        "reg_mae_sum": reg_mae_sum,
        "reg_count": reg_count,
        "param_pred_raw": out["param_pred_raw"].float(),
        "loss_stress": loss_stress_part,
        "loss_flow": loss_flow_part,
        "loss_stress_edge": loss_stress_edge,
        "loss_flow_edge": loss_flow_edge,
        "loss_stress_recon": loss_stress_recon,
        "loss_flow_recon": loss_flow_recon,
        "loss_force_bce": loss_force_bce,
        "loss_force_dice": loss_force_dice,
        "loss_force": loss_force_part,
        "loss_action": action_ret["loss_action"],
        "loss_phys_total": phys_ret["loss_phys_total"],
        "loss_phys_sf": phys_ret["loss_stress_flow_consistency"],
        "loss_phys_fs": phys_ret["loss_force_stress_consistency"],
        "loss_phys_ff": phys_ret["loss_force_flow_consistency"],
        "action_acc": action_ret["action_acc"],
    }
    return loss_total, parts


def _eval_sums_to_record(
    sums: List[float],
    *,
    lambda_stress: float,
    lambda_flow: float,
    lambda_force: float,
    lambda_action: float,
    lambda_phys: float,
) -> Dict[str, Any]:
    """将 eval 累积向量转为 JSON/日志用 dict。"""
    n = max(float(sums[11]), 1.0)

    def a(i: int) -> float:
        return float(sums[i]) / n

    out = {
        "avg_loss": a(0),
        "avg_loss_reg": a(1),
        "avg_loss_stress": a(2),
        "avg_loss_flow": a(3),
        "avg_loss_force": a(4),
        "avg_loss_action": a(5),
        "avg_loss_phys_total": a(6),
        "avg_loss_phys_sf": a(7),
        "avg_loss_phys_fs": a(8),
        "avg_loss_phys_ff": a(9),
        "avg_action_acc": a(10),
        "weighted_reg": a(1),
        "weighted_stress": float(lambda_stress) * a(2),
        "weighted_flow": float(lambda_flow) * a(3),
        "weighted_force": float(lambda_force) * a(4),
        "weighted_action": float(lambda_action) * a(5),
        "weighted_phys": float(lambda_phys) * a(6),
        "num_samples": int(round(n)),
        "avg_loss_stress_edge": a(_REG_EDGE_STRESS),
        "avg_loss_flow_edge": a(_REG_EDGE_FLOW),
        "avg_loss_stress_recon": a(_REG_STRESS_RECON),
        "avg_loss_flow_recon": a(_REG_FLOW_RECON),
        "avg_loss_force_bce": a(_REG_FORCE_BCE),
        "avg_loss_force_dice": a(_REG_FORCE_DICE),
    }
    for i, name in enumerate(_REG_TARGET_NAMES):
        out[f"avg_reg_mae_{name}"] = _per_target_mae_from_sums(sums, i)
        out[f"num_valid_{name}"] = int(round(float(sums[_REG_COUNT_START + i])))
    return out


def _eval_loss_breakdown_kwargs(
    metrics: Dict[str, Any],
    *,
    eval_loss_switches: Dict[str, bool],
    lambda_stress_edge: float,
    lambda_flow_edge: float,
    force_bce_weight: float,
    force_dice_weight: float,
) -> Dict[str, Any]:
    """供 eval 汇总日志 / tqdm 将 loss 子项传入 ``_stage_metric_parts``。"""
    _fb = float(metrics.get("avg_loss_force_bce", 0.0))
    _fd = float(metrics.get("avg_loss_force_dice", 0.0))
    return {
        "stress_recon": float(metrics["avg_loss_stress_recon"])
        if bool(eval_loss_switches.get("use_stress_loss"))
        else None,
        "stress_edge": float(metrics["avg_loss_stress_edge"])
        if (bool(eval_loss_switches.get("use_stress_loss")) and float(lambda_stress_edge) > 0.0)
        else None,
        "flow_recon": float(metrics["avg_loss_flow_recon"])
        if bool(eval_loss_switches.get("use_flow_loss"))
        else None,
        "flow_edge": float(metrics["avg_loss_flow_edge"])
        if (bool(eval_loss_switches.get("use_flow_loss")) and float(lambda_flow_edge) > 0.0)
        else None,
        "force_bce": (_fb if (_fb + _fd) > 1e-12 else None),
        "force_dice": (_fd if (_fb + _fd) > 1e-12 else None),
        "lambda_stress_edge": float(lambda_stress_edge),
        "lambda_flow_edge": float(lambda_flow_edge),
        "force_bce_weight": float(force_bce_weight),
        "force_dice_weight": float(force_dice_weight),
    }


def _eval_partial_sums_breakdown_kwargs(
    sums_t: torch.Tensor,
    *,
    eval_loss_switches: Dict[str, bool],
    lambda_stress_edge: float,
    lambda_flow_edge: float,
    force_bce_weight: float,
    force_dice_weight: float,
) -> Dict[str, Any]:
    """eval 分片跑批过程中，用当前卡上累积的 sums 近似子项（仅 tqdm postfix）。"""
    _sn = float(sums_t[11].item())
    d = max(_sn, 1.0)
    _fb = float(sums_t[_REG_FORCE_BCE].item() / d)
    _fd = float(sums_t[_REG_FORCE_DICE].item() / d)
    return {
        "stress_recon": float(sums_t[_REG_STRESS_RECON].item() / d)
        if bool(eval_loss_switches.get("use_stress_loss"))
        else None,
        "stress_edge": float(sums_t[_REG_EDGE_STRESS].item() / d)
        if (bool(eval_loss_switches.get("use_stress_loss")) and float(lambda_stress_edge) > 0.0)
        else None,
        "flow_recon": float(sums_t[_REG_FLOW_RECON].item() / d)
        if bool(eval_loss_switches.get("use_flow_loss"))
        else None,
        "flow_edge": float(sums_t[_REG_EDGE_FLOW].item() / d)
        if (bool(eval_loss_switches.get("use_flow_loss")) and float(lambda_flow_edge) > 0.0)
        else None,
        "force_bce": (_fb if (_fb + _fd) > 1e-12 else None),
        "force_dice": (_fd if (_fb + _fd) > 1e-12 else None),
        "lambda_stress_edge": float(lambda_stress_edge),
        "lambda_flow_edge": float(lambda_flow_edge),
        "force_bce_weight": float(force_bce_weight),
        "force_dice_weight": float(force_dice_weight),
    }


def _eval_partial_list_breakdown_kwargs(
    sums_list: List[float],
    *,
    eval_loss_switches: Dict[str, bool],
    lambda_stress_edge: float,
    lambda_flow_edge: float,
    force_bce_weight: float,
    force_dice_weight: float,
) -> Dict[str, Any]:
    """单卡 eval 跑批时，用 Python list 累积的 sums 近似子项（tqdm postfix）。"""
    d = max(sums_list[11], 1.0)
    _fb = float(sums_list[_REG_FORCE_BCE] / d)
    _fd = float(sums_list[_REG_FORCE_DICE] / d)
    return {
        "stress_recon": float(sums_list[_REG_STRESS_RECON] / d)
        if bool(eval_loss_switches.get("use_stress_loss"))
        else None,
        "stress_edge": float(sums_list[_REG_EDGE_STRESS] / d)
        if (bool(eval_loss_switches.get("use_stress_loss")) and float(lambda_stress_edge) > 0.0)
        else None,
        "flow_recon": float(sums_list[_REG_FLOW_RECON] / d)
        if bool(eval_loss_switches.get("use_flow_loss"))
        else None,
        "flow_edge": float(sums_list[_REG_EDGE_FLOW] / d)
        if (bool(eval_loss_switches.get("use_flow_loss")) and float(lambda_flow_edge) > 0.0)
        else None,
        "force_bce": (_fb if (_fb + _fd) > 1e-12 else None),
        "force_dice": (_fd if (_fb + _fd) > 1e-12 else None),
        "lambda_stress_edge": float(lambda_stress_edge),
        "lambda_flow_edge": float(lambda_flow_edge),
        "force_bce_weight": float(force_bce_weight),
        "force_dice_weight": float(force_dice_weight),
    }


def _tqdm_log(msg: str) -> None:
    tqdm.write(str(msg))


def _append_loss_breakdown_parts(
    parts: List[str],
    *,
    prefix: str,
    stress_recon: Optional[float],
    stress_edge: Optional[float],
    flow_recon: Optional[float],
    flow_edge: Optional[float],
    force_bce: Optional[float],
    force_dice: Optional[float],
    lambda_stress_edge: float,
    lambda_flow_edge: float,
    force_bce_weight: float,
    force_dice_weight: float,
) -> None:
    """在已有分项后追加：场重建 / edge 原项 / force BCE·Dice 子项（若传入非 None）。"""
    p = prefix
    if stress_recon is not None:
        parts.append(f"{p}s_rec={float(stress_recon):.4f}")
    if stress_edge is not None and float(lambda_stress_edge) > 0.0:
        parts.append(f"{p}s_edge={float(stress_edge):.4f}")
        parts.append(f"{p}s_wedge={float(lambda_stress_edge) * float(stress_edge):.4f}")
    if flow_recon is not None:
        parts.append(f"{p}f_rec={float(flow_recon):.4f}")
    if flow_edge is not None and float(lambda_flow_edge) > 0.0:
        parts.append(f"{p}f_edge={float(flow_edge):.4f}")
        parts.append(f"{p}f_wedge={float(lambda_flow_edge) * float(flow_edge):.4f}")
    if force_bce is not None and force_dice is not None:
        parts.append(f"{p}F_bce={float(force_bce):.4f}")
        parts.append(f"{p}F_dice={float(force_dice):.4f}")
        parts.append(
            f"{p}F_mix={float(force_bce_weight) * float(force_bce) + float(force_dice_weight) * float(force_dice):.4f}"
        )


def _stage_metric_parts(
    *,
    stage_name: str,
    total_loss: float,
    reg_loss: float,
    stress_loss: float,
    flow_loss: float,
    force_loss: float,
    action_loss: float = 0.0,
    phys_loss: float = 0.0,
    action_acc: Optional[float] = None,
    prefix: str = "",
    include_field_metrics_in_param_stage: bool = False,
    stress_recon: Optional[float] = None,
    stress_edge: Optional[float] = None,
    flow_recon: Optional[float] = None,
    flow_edge: Optional[float] = None,
    force_bce: Optional[float] = None,
    force_dice: Optional[float] = None,
    lambda_stress_edge: float = 0.0,
    lambda_flow_edge: float = 0.0,
    force_bce_weight: float = 0.7,
    force_dice_weight: float = 0.3,
    show_breakdown: bool = True,
) -> List[str]:
    p = str(prefix)
    parts = [f"{p}loss={float(total_loss):.4f}"]
    s = str(stage_name).strip().lower()
    if s == "flow_force":
        parts.extend(
            [
                f"{p}flow={float(flow_loss):.4f}",
                f"{p}force={float(force_loss):.4f}",
            ]
        )
        if show_breakdown:
            _append_loss_breakdown_parts(
                parts,
                prefix=p,
                stress_recon=None,
                stress_edge=None,
                flow_recon=flow_recon,
                flow_edge=flow_edge,
                force_bce=force_bce,
                force_dice=force_dice,
                lambda_stress_edge=0.0,
                lambda_flow_edge=lambda_flow_edge,
                force_bce_weight=force_bce_weight,
                force_dice_weight=force_dice_weight,
            )
        return parts
    if s == "stress":
        parts.append(f"{p}stress={float(stress_loss):.4f}")
        if show_breakdown:
            _append_loss_breakdown_parts(
                parts,
                prefix=p,
                stress_recon=stress_recon,
                stress_edge=stress_edge,
                flow_recon=None,
                flow_edge=None,
                force_bce=None,
                force_dice=None,
                lambda_stress_edge=lambda_stress_edge,
                lambda_flow_edge=0.0,
                force_bce_weight=force_bce_weight,
                force_dice_weight=force_dice_weight,
            )
        return parts
    if s == "field":
        parts.extend(
            [
                f"{p}stress={float(stress_loss):.4f}",
                f"{p}flow={float(flow_loss):.4f}",
                f"{p}force={float(force_loss):.4f}",
            ]
        )
        if float(phys_loss) != 0.0:
            parts.append(f"{p}phys={float(phys_loss):.4f}")
        if show_breakdown:
            _append_loss_breakdown_parts(
                parts,
                prefix=p,
                stress_recon=stress_recon,
                stress_edge=stress_edge,
                flow_recon=flow_recon,
                flow_edge=flow_edge,
                force_bce=force_bce,
                force_dice=force_dice,
                lambda_stress_edge=lambda_stress_edge,
                lambda_flow_edge=lambda_flow_edge,
                force_bce_weight=force_bce_weight,
                force_dice_weight=force_dice_weight,
            )
        return parts
    if s == "param":
        parts.append(f"{p}reg={float(reg_loss):.4f}")
        if include_field_metrics_in_param_stage:
            parts.extend(
                [
                    f"{p}stress={float(stress_loss):.4f}",
                    f"{p}flow={float(flow_loss):.4f}",
                    f"{p}force={float(force_loss):.4f}",
                ]
            )
            if show_breakdown:
                _append_loss_breakdown_parts(
                    parts,
                    prefix=p,
                    stress_recon=stress_recon,
                    stress_edge=stress_edge,
                    flow_recon=flow_recon,
                    flow_edge=flow_edge,
                    force_bce=force_bce,
                    force_dice=force_dice,
                    lambda_stress_edge=lambda_stress_edge,
                    lambda_flow_edge=lambda_flow_edge,
                    force_bce_weight=force_bce_weight,
                    force_dice_weight=force_dice_weight,
                )
        return parts
    parts.extend(
        [
            f"{p}reg={float(reg_loss):.4f}",
            f"{p}stress={float(stress_loss):.4f}",
            f"{p}flow={float(flow_loss):.4f}",
            f"{p}force={float(force_loss):.4f}",
        ]
    )
    if float(action_loss) != 0.0:
        parts.append(f"{p}action={float(action_loss):.4f}")
    if float(phys_loss) != 0.0:
        parts.append(f"{p}phys={float(phys_loss):.4f}")
    if action_acc is not None and float(action_loss) != 0.0:
        parts.append(f"{p}acc={float(action_acc):.4f}")
    if show_breakdown:
        _append_loss_breakdown_parts(
            parts,
            prefix=p,
            stress_recon=stress_recon,
            stress_edge=stress_edge,
            flow_recon=flow_recon,
            flow_edge=flow_edge,
            force_bce=force_bce,
            force_dice=force_dice,
            lambda_stress_edge=lambda_stress_edge,
            lambda_flow_edge=lambda_flow_edge,
            force_bce_weight=force_bce_weight,
            force_dice_weight=force_dice_weight,
        )
    return parts


def _tensorboard_log_eval_metrics(tb: SummaryWriter, m: Dict[str, Any], epoch_1based: int) -> None:
    for key in (
        "avg_loss",
        "avg_loss_reg",
        "avg_loss_stress",
        "avg_loss_flow",
        "avg_loss_force",
        "avg_loss_action",
        "avg_loss_phys_total",
        "avg_loss_phys_sf",
        "avg_loss_phys_fs",
        "avg_loss_phys_ff",
        "avg_loss_stress_edge",
        "avg_loss_flow_edge",
        "avg_loss_stress_recon",
        "avg_loss_flow_recon",
        "avg_loss_force_bce",
        "avg_loss_force_dice",
        "avg_action_acc",
        "weighted_reg",
        "weighted_stress",
        "weighted_flow",
        "weighted_force",
        "weighted_action",
        "weighted_phys",
        "avg_reg_mae_logE",
        "avg_reg_mae_nu",
        "avg_reg_mae_logDensity",
        "avg_reg_mae_logYield",
    ):
        if key in m:
            tb.add_scalar(f"eval/{key}", float(m[key]), epoch_1based)


def _build_checkpoint_payload(
    *,
    epoch_1based: int,
    avg_loss: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    use_amp: bool,
    action_to_id: Dict[str, int],
    model_arch: str,
    max_views: int,
    dec_h: int,
    dec_w: int,
    cfg_model: Dict[str, Any],
    eval_metrics: Optional[Dict[str, Any]] = None,
    stage_name: Optional[str] = None,
    checkpoint_kind: str = "periodic",
) -> Dict[str, Any]:
    mcore = _unwrap_model(model)
    model_hparams = {
        "arch": str(model_arch),
        "num_views": int(max_views),
        "in_channels": int(mcore.in_channels),
        "num_frames": int(mcore.num_frames),
        "dec_h": int(dec_h),
        "dec_w": int(dec_w),
    }
    if model_arch in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
        model_hparams.update(
            {
                "dino_backbone_name": str(_pick(cfg_model, "dino_backbone_name", "dinov2_vits14")),
                "dino_backbone_pretrained": bool(_pick(cfg_model, "dino_backbone_pretrained", True)),
                "dino_backbone_source": str(_pick(cfg_model, "dino_backbone_source", "torchhub")),
                "dino_torchhub_repo": str(_pick(cfg_model, "dino_torchhub_repo", "facebookresearch/dinov2:main")),
                "torchhub_dir": str(torch.hub.get_dir()),
                "dino_force_reload": bool(_pick(cfg_model, "dino_force_reload", False)),
                "dino_trust_repo": bool(_pick(cfg_model, "dino_trust_repo", True)),
                "dino_skip_validation": bool(_pick(cfg_model, "dino_skip_validation", True)),
                "dino_hub_verbose": bool(_pick(cfg_model, "dino_hub_verbose", False)),
                "dino_log_torchhub_dir": bool(_pick(cfg_model, "dino_log_torchhub_dir", False)),
                "dino_out_dim": int(_pick(cfg_model, "dino_out_dim", 384)),
                "temporal_adapter_type": str(_pick(cfg_model, "temporal_adapter_type", "transformer")),
                "temporal_adapter_layers": int(_pick(cfg_model, "temporal_adapter_layers", 2)),
                "temporal_adapter_heads": int(_pick(cfg_model, "temporal_adapter_heads", 6)),
                "temporal_adapter_dropout": float(_pick(cfg_model, "temporal_adapter_dropout", 0.1)),
                "frame_pool": str(_pick(cfg_model, "frame_pool", "mean")),
                "freeze_backbone": bool(_pick(cfg_model, "freeze_backbone", True)),
                "field_head_mode": str(_pick(cfg_model, "field_head_mode", "independent")),
                "field_token_dim": int(_pick(cfg_model, "field_token_dim", 512)),
                "field_base_channels": int(_pick(cfg_model, "field_base_channels", 128)),
                "field_shared_channels": int(_pick(cfg_model, "field_shared_channels", 64)),
                "field_temporal_layers": int(_pick(cfg_model, "field_temporal_layers", 0)),
                "field_spatial_channels": int(_pick(cfg_model, "field_spatial_channels", 0)),
                "field_use_multiscale_spatial": bool(_pick(cfg_model, "field_use_multiscale_spatial", False)),
                "field_use_shared_task_phys": bool(_pick(cfg_model, "field_use_shared_task_phys", False)),
                "field_use_geometry_residual": bool(_pick(cfg_model, "field_use_geometry_residual", False)),
                "field_sequential_stress": bool(_pick(cfg_model, "field_sequential_stress", False)),
                "use_stress_spatial_enhancer": bool(_pick(cfg_model, "use_stress_spatial_enhancer", False)),
                "use_flow_spatial_enhancer": cfg_model.get("use_flow_spatial_enhancer"),
                "flow_output_scale_init": float(_pick(cfg_model, "flow_output_scale_init", 0.5)),
                "flow_output_bias_init": float(_pick(cfg_model, "flow_output_bias_init", -0.5)),
                "force_out_channels": int(_pick(cfg_model, "force_out_channels", 3)),
                "field_use_patch_tokens": bool(_pick(cfg_model, "field_use_patch_tokens", False)),
                "field_patch_dim": int(_pick(cfg_model, "field_patch_dim", 256)),
                "param_chain_mode": str(_pick(cfg_model, "param_chain_mode", "baseline")),
                "param_token_dim": int(_pick(cfg_model, "param_token_dim", 256)),
                "param_mixer_layers": int(_pick(cfg_model, "param_mixer_layers", 2)),
                "param_mixer_heads": int(_pick(cfg_model, "param_mixer_heads", 8)),
                "param_mixer_dropout": float(
                    _pick(cfg_model, "param_mixer_dropout", _pick(cfg_model, "head_dropout", 0.1))
                ),
                "param_use_rgb_static": bool(_pick(cfg_model, "param_use_rgb_static", True)),
                "param_use_rgb_residual": bool(_pick(cfg_model, "param_use_rgb_residual", True)),
                "param_use_masked_field_tokens": bool(_pick(cfg_model, "param_use_masked_field_tokens", True)),
            }
        )
    payload: Dict[str, Any] = {
        "epoch": int(epoch_1based),
        "avg_loss": float(avg_loss),
        "stage_name": (None if stage_name is None else str(stage_name)),
        "checkpoint_kind": str(checkpoint_kind),
        "model_state": mcore.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if use_amp else None,
        "action_to_id": action_to_id,
        "model_hparams": model_hparams,
    }
    if eval_metrics is not None:
        payload["eval_metrics"] = dict(eval_metrics)
    return payload


def _resolve_eval_loss_switches(
    *,
    stage_name: str,
    use_reg_loss: bool,
    use_stress_loss: bool,
    use_flow_loss: bool,
    use_force_loss: bool,
    use_action_loss: bool,
    use_phys_loss: bool,
) -> Dict[str, bool]:
    s = str(stage_name).strip().lower()
    if s == "flow_force":
        return {
            "use_reg_loss": False,
            "use_stress_loss": False,
            "use_flow_loss": True,
            "use_force_loss": True,
            "use_action_loss": False,
            "use_phys_loss": False,
        }
    if s == "stress":
        return {
            "use_reg_loss": False,
            "use_stress_loss": True,
            "use_flow_loss": False,
            "use_force_loss": False,
            "use_action_loss": False,
            "use_phys_loss": False,
        }
    if s == "joint":
        return {
            "use_reg_loss": bool(use_reg_loss),
            "use_stress_loss": bool(use_stress_loss),
            "use_flow_loss": bool(use_flow_loss),
            "use_force_loss": bool(use_force_loss),
            "use_action_loss": bool(use_action_loss),
            "use_phys_loss": bool(use_phys_loss),
        }
    if s == "field":
        return {
            "use_reg_loss": False,
            "use_stress_loss": bool(use_stress_loss),
            "use_flow_loss": bool(use_flow_loss),
            "use_force_loss": bool(use_force_loss),
            "use_action_loss": False,
            "use_phys_loss": False,
        }
    if s == "param":
        return {
            "use_reg_loss": bool(use_reg_loss),
            "use_stress_loss": False,
            "use_flow_loss": False,
            "use_force_loss": False,
            "use_action_loss": False,
            "use_phys_loss": False,
        }
    return {
        "use_reg_loss": bool(use_reg_loss),
        "use_stress_loss": bool(use_stress_loss),
        "use_flow_loss": bool(use_flow_loss),
        "use_force_loss": bool(use_force_loss),
        "use_action_loss": bool(use_action_loss),
        "use_phys_loss": bool(use_phys_loss),
    }


def _axis_lo_hi(
    xs: np.ndarray,
    ys: np.ndarray,
    pad_frac: float,
    axis_percentiles: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    npt = int(xs.shape[0])
    if axis_percentiles is not None and npt >= 2:
        pl, ph = float(axis_percentiles[0]), float(axis_percentiles[1])
        lo = float(min(np.percentile(xs, pl), np.percentile(ys, pl)))
        hi = float(max(np.percentile(xs, ph), np.percentile(ys, ph)))
    else:
        lo = float(min(xs.min(), ys.min()))
        hi = float(max(xs.max(), ys.max()))
    span = hi - lo
    if span < 1e-12:
        return lo - 0.5, hi + 0.5
    pad = float(max(0.0, pad_frac)) * span
    return lo - pad, hi + pad


def _save_grouped_param_scatter_figure(
    records: List[Dict[str, Any]],
    out_path: Path,
    *,
    group_key: str,
    group_label: str,
    space: str,
    pad_frac: float = 0.02,
    axis_percentiles: Optional[Tuple[float, float]] = None,
) -> None:
    if not records:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:
        raise RuntimeError("参数散点图需要 matplotlib：pip install matplotlib") from e

    gt_raw = np.asarray([r["param_gt_raw"] for r in records], dtype=np.float64)
    pred_raw = np.asarray([r["param_pred_raw"] for r in records], dtype=np.float64)
    if gt_raw.size == 0 or pred_raw.size == 0:
        return
    if str(space).strip().lower() == "log":
        gt = _to_target_params(torch.from_numpy(gt_raw).float()).cpu().numpy().astype(np.float64)
        pred = _to_target_params(torch.from_numpy(pred_raw).float()).cpu().numpy().astype(np.float64)
        names = ("logE", "nu", "logDensity", "logYield")
    else:
        gt = gt_raw
        pred = pred_raw
        names = ("E", "nu", "density", "yield_stress")

    groups = np.asarray(
        [str(r.get(group_key, "unknown")).strip() or "unknown" for r in records],
        dtype=object,
    )
    uniq_groups = sorted(set(groups.tolist()))
    if not uniq_groups:
        uniq_groups = ["unknown"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    cmap = plt.cm.get_cmap("tab20", max(20, len(uniq_groups)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
    style_map: Dict[str, Tuple[Any, str]] = {}
    for idx, name in enumerate(uniq_groups):
        style_map[name] = (cmap(idx % cmap.N), markers[idx % len(markers)])

    for ax, i, name in zip(axes.flat, range(4), names):
        gx = gt[:, i].astype(np.float64)
        py = pred[:, i].astype(np.float64)
        valid = np.isfinite(gx) & np.isfinite(py)
        if str(space).strip().lower() == "raw" and name == "yield_stress":
            valid = valid & (gx >= 0) & (py >= 0)
        if not np.any(valid):
            ax.set_visible(False)
            continue
        xs, ys = gx[valid], py[valid]
        lo, hi = _axis_lo_hi(xs, ys, pad_frac, axis_percentiles)
        for gname in uniq_groups:
            gm = valid & (groups == gname)
            if not np.any(gm):
                continue
            color, marker = style_map[gname]
            ax.scatter(
                gx[gm],
                py[gm],
                s=16,
                alpha=0.7,
                c=[color],
                marker=marker,
                edgecolors="none",
            )
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("GT")
        ax.set_ylabel("Pred")
        ax.set_title(f"{name}  N={int(valid.sum())}")

    handles = []
    for gname in uniq_groups:
        color, marker = style_map[gname]
        count = sum(1 for r in records if (str(r.get(group_key, "unknown")).strip() or "unknown") == gname)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=6,
                label=f"{gname} ({count})",
            )
        )
    fig.suptitle(f"Param GT vs Pred ({space}) grouped by {group_label}", fontsize=12)
    fig.subplots_adjust(right=0.78)
    if handles:
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.80, 0.5),
            fontsize=7,
            frameon=False,
        )
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_eval_visual_artifacts_rank0(
    *,
    model: torch.nn.Module,
    eval_ds: torch.utils.data.Dataset,
    train_vis_ds: Optional[torch.utils.data.Dataset],
    device: torch.device,
    mcore: torch.nn.Module,
    eval_dir: Path,
    epoch_1based: int,
    stage_name: str,
    eval_video_num_samples: int,
    eval_train_video_num_samples: int,
    param_eval_records: List[Dict[str, Any]],
    eval_param_scatter: bool,
    eval_param_scatter_log: bool,
    eval_scatter_pad_frac: float,
    eval_scatter_axis_percentiles: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    visual_dir = eval_dir / f"epoch_{epoch_1based:04d}_visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    out["visual_dir"] = str(visual_dir)

    if int(eval_video_num_samples) > 0:
        from logic_model.eval_visual import export_field_videos_rank0

        export_field_videos_rank0(
            model=model,
            eval_ds=eval_ds,
            device=device,
            mcore=mcore,
            out_root=visual_dir,
            max_samples=int(eval_video_num_samples),
            fps=8,
            view_idx=0,
            color_mode="rgb",
        )
        out["field_videos_dir"] = str(visual_dir / "field_videos")

    if train_vis_ds is not None and int(eval_train_video_num_samples) > 0:
        from logic_model.eval_visual import export_field_videos_rank0

        train_visual_dir = visual_dir / "train_samples"
        export_field_videos_rank0(
            model=model,
            eval_ds=train_vis_ds,
            device=device,
            mcore=mcore,
            out_root=train_visual_dir,
            max_samples=int(eval_train_video_num_samples),
            fps=8,
            view_idx=0,
            color_mode="rgb",
        )
        out["train_field_videos_dir"] = str(train_visual_dir / "field_videos")

    if str(stage_name).strip().lower() == "param" and bool(eval_param_scatter) and param_eval_records:
        scatter_dir = visual_dir / "param_scatter"
        scatter_dir.mkdir(parents=True, exist_ok=True)
        (scatter_dir / "records.json").write_text(
            json.dumps(param_eval_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        for group_key, group_label in (
            ("object", "object"),
            ("material", "material"),
            ("action", "action"),
        ):
            _save_grouped_param_scatter_figure(
                param_eval_records,
                scatter_dir / f"{group_key}_raw.png",
                group_key=group_key,
                group_label=group_label,
                space="raw",
                pad_frac=float(eval_scatter_pad_frac),
                axis_percentiles=eval_scatter_axis_percentiles,
            )
            if bool(eval_param_scatter_log):
                _save_grouped_param_scatter_figure(
                    param_eval_records,
                    scatter_dir / f"{group_key}_log.png",
                    group_key=group_key,
                    group_label=group_label,
                    space="log",
                    pad_frac=float(eval_scatter_pad_frac),
                    axis_percentiles=eval_scatter_axis_percentiles,
                )
        out["param_scatter_dir"] = str(scatter_dir)
    return out


def _auto_pick_least_utilized_gpu() -> int | None:
    """
    使用 nvidia-smi 按 utilization.gpu 选择最空闲 GPU 的“逻辑编号”。
    - 若设置了 CUDA_VISIBLE_DEVICES，会先在可见物理卡内选择，再映射为逻辑序号。
    - 若查询失败，返回 None。
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None

    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            util = int(parts[1])
            mem = int(parts[2])
        except ValueError:
            continue
        rows.append((util, mem, idx))

    if not rows:
        return None

    # 默认: 物理 index -> (util, mem)
    phys_stats = {int(idx): (int(util), int(mem)) for util, mem, idx in rows}

    # 若设置了 CUDA_VISIBLE_DEVICES，按可见物理卡过滤，再映射到逻辑 index
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_env:
        visible_phys: list[int] = []
        for tok in visible_env.split(","):
            t = tok.strip()
            if not t:
                continue
            try:
                visible_phys.append(int(t))
            except ValueError:
                # UUID/MIG 格式时不做物理映射，退回 torch 逻辑索引
                visible_phys = []
                break

        if visible_phys:
            cands = []
            for logical_idx, phys_idx in enumerate(visible_phys):
                st = phys_stats.get(int(phys_idx))
                if st is None:
                    continue
                util, mem = st
                cands.append((util, mem, logical_idx))
            if cands:
                cands.sort()
                return int(cands[0][2])
            return 0

    # 未设置 CUDA_VISIBLE_DEVICES：一般逻辑 index 与物理 index 一致
    visible_count = int(torch.cuda.device_count())
    if visible_count <= 0:
        return None
    cands = []
    for phys_idx, (util, mem) in phys_stats.items():
        if 0 <= int(phys_idx) < visible_count:
            cands.append((util, mem, int(phys_idx)))
    if not cands:
        return 0
    cands.sort()
    return int(cands[0][2])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("logic_model minimal trainer")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument(
        "--split_root",
        type=str,
        default=None,
        help="数据根目录；缺省时用 config 的 data.split_root",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="未传时读 train.epochs；仍缺省则为 1000",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="未传时读 train.batch_size；仍缺省则为 1",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="未传时读 train.max_samples；>0 仅前 N 个样本；0=全部",
    )
    ap.add_argument("--lr", type=float, default=None, help="未传时读 train.lr；缺省 3e-4")
    ap.add_argument("--num_workers", type=int, default=None, help="未传时读 train.num_workers；缺省 0")
    ap.add_argument("--max_views", type=int, default=None, help="未传时读 model.num_views；缺省 4")
    ap.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="未传时读 model.num_frames；0=按 LMDB；CLI 缺省视为未传",
    )
    ap.add_argument("--img_size", type=int, default=None, help="未传时读 model.img_size")
    ap.add_argument("--dec_h", type=int, default=None, help="未传时读 model.dec_h；缺省 112")
    ap.add_argument("--dec_w", type=int, default=None, help="未传时读 model.dec_w；缺省 112")
    ap.add_argument("--device", type=str, default=None, help="未传时读 train.device；缺省 cuda")
    ap.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="覆盖 train.output_root（如多副本并行时按副本区分目录）；未传时读 JSON",
    )
    # loss weights（未传时读 train.*；显式 CLI 覆盖 JSON）
    ap.add_argument("--lambda_stress", type=float, default=None)
    ap.add_argument("--lambda_flow", type=float, default=None)
    ap.add_argument("--lambda_force", type=float, default=None)
    ap.add_argument("--lambda_action", type=float, default=None)
    ap.add_argument("--lambda_phys", type=float, default=None)
    ap.add_argument("--disable_action_loss", action="store_true", help="从总 loss 中移除 action loss")
    ap.add_argument("--disable_phys_loss", action="store_true", help="从总 loss 中移除 physics consistency loss")
    ap.add_argument("--stage1_field_epochs", type=int, default=None, help="两阶段训练：第一阶段仅场头训练的 epoch 数")
    ap.add_argument("--use_pred_force_mask_for_phys", action="store_true")
    ap.add_argument(
        "--save_every_epochs",
        type=int,
        default=None,
        help="覆盖 JSON：仅建议写 train.checkpoint.save_every_epochs；未传 CLI 时只读 checkpoint.save_every_epochs，再缺省 100000",
    )
    ap.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="从 checkpoint 续训；未传时读 train.checkpoint.resume_from",
    )
    ap.add_argument(
        "--eval_every_epochs",
        type=int,
        default=None,
        help="未传时读 train.eval_every_epochs；缺省 100；旧配置仅设 quick_eval.every_epochs 时仍兼容并告警",
    )
    ap.add_argument(
        "--eval_batches",
        type=int,
        default=None,
        help="未传时读 train.eval_max_batches；0=跑满 eval_loader；CLI 缺省同未传",
    )
    ap.add_argument("--object_mask_fg_weight", type=float, default=None, help="未传时读 train.object_mask_fg_weight")
    ap.add_argument("--object_mask_bg_weight", type=float, default=None, help="未传时读 train.object_mask_bg_weight")
    ap.add_argument("--object_mask_bg_black", type=int, default=None, help="未传时读 train.object_mask_bg_black")
    ap.add_argument(
        "--ddp",
        action="store_true",
        help="显式要求 DDP（需 torchrun WORLD_SIZE>1；一般可不写，自动检测）",
    )
    ap.add_argument(
        "--no_amp",
        action="store_true",
        help="关闭 CUDA AMP（默认训练在 GPU 上使用 fp16 autocast + GradScaler）",
    )
    ap.add_argument(
        "--debug_timing",
        action="store_true",
        help="开启训练 timing 调试（默认关闭；仅采样步做更精确 GPU 计时）",
    )
    ap.add_argument(
        "--debug_timing_every_steps",
        type=int,
        default=None,
        help="timing 采样步长；未传时读 train.debug_timing_every_steps，默认 50",
    )
    ap.add_argument(
        "--profile_timing",
        action="store_true",
        help="开启 sampled timing profiling（默认关闭）",
    )
    ap.add_argument(
        "--profile_timing_every",
        type=int,
        default=None,
        help="profile 采样步长；未传时读 train.profile_timing_every，默认 50",
    )
    ap.add_argument(
        "--timing_log_first_n",
        type=int,
        default=None,
        help="每个 epoch 前 N 个 batch 打印 timing；未传时读 train.timing_log_first_n，默认 5",
    )
    ap.add_argument(
        "--timing_log_every",
        type=int,
        default=None,
        help="每 K 个 batch 打印一次 timing；未传时读 train.timing_log_every，默认 50；<=0 表示关闭",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    cfg_data = cfg.get("data") or {}
    cfg_model = cfg.get("model") or {}
    cfg_train = cfg.get("train") or {}
    startup_t0 = time.perf_counter()
    rank_env = int(os.environ.get("RANK", "0"))

    split_root = (args.split_root or "").strip() or str(_pick(cfg_data, "split_root", "") or "").strip()
    if not split_root:
        split_root = "auto_output/dataset_deformation_stress_500_new/train"
    split_root = str(Path(split_root).expanduser())

    max_views = int(_cli_or_pick_cfg(args.max_views, cfg_model, "num_views", 4))
    num_frames = int(_cli_or_pick_cfg(args.num_frames, cfg_model, "num_frames", 0))
    img_size = int(_cli_or_pick_cfg(args.img_size, cfg_model, "img_size", 0))
    dec_h = int(_cli_or_pick_cfg(args.dec_h, cfg_model, "dec_h", 112))
    dec_w = int(_cli_or_pick_cfg(args.dec_w, cfg_model, "dec_w", 112))

    lmdb_env_subdir = str(
        _pick(cfg_data, "lmdb_env_subdir", "arch4_data.lmdb") or "arch4_data.lmdb"
    ).strip() or "arch4_data.lmdb"
    sample_pack_name = str(
        _pick(cfg_data, "sample_pack_name", "sample_pack.npz") or "sample_pack.npz"
    ).strip() or "sample_pack.npz"
    sample_storage_backend = str(
        _pick(cfg_data, "sample_storage_backend", "auto") or "auto"
    ).strip() or "auto"
    purify_force_mask_on_read = bool(_pick(cfg_data, "purify_force_mask_on_read", False))
    force_mask_purify_mode = str(
        _pick(cfg_data, "force_mask_purify_mode", "red_minus_others") or "red_minus_others"
    ).strip() or "red_minus_others"
    force_mask_keep_three_channels = bool(_pick(cfg_data, "force_mask_keep_three_channels", True))
    force_mask_single_channel = bool(_pick(cfg_data, "force_mask_single_channel", False))
    force_mask_binary_threshold = float(_pick(cfg_data, "force_mask_binary_threshold", 0.5))
    allow_missing_sample_ids = bool(
        _pick(
            cfg_train,
            "allow_missing_sample_ids",
            _pick(cfg_data, "allow_missing_sample_ids", False),
        )
    )
    cache_backend = cfg_train.get("cache_backend", cfg_data.get("cache_backend"))
    cache_root = cfg_train.get("cache_root", cfg_data.get("cache_root"))
    cache_version = str(cfg_train.get("cache_version", _pick(cfg_data, "cache_version", "v1")))
    cache_readonly = bool(cfg_train.get("cache_readonly", _pick(cfg_data, "cache_readonly", False)))
    cache_force_rebuild = bool(
        cfg_train.get("cache_force_rebuild", _pick(cfg_data, "cache_force_rebuild", False))
    )
    cache_log = bool(cfg_train.get("cache_log", _pick(cfg_data, "cache_log", False)))
    prebuild_cache_first = bool(_pick(cfg_train, "prebuild_cache_first", False))
    prebuild_num_workers = int(_pick(cfg_train, "prebuild_num_workers", 0))
    cache_readonly_after_prebuild = bool(_pick(cfg_train, "cache_readonly_after_prebuild", True))

    train_ids_json = cfg_data.get("train_ids_json") or cfg_train.get("train_ids_json")
    train_id_list: Optional[List[str]] = None
    test_id_list: Optional[List[str]] = None
    split_meta: Dict[str, Any] = {}
    if train_ids_json:
        _startup_phase_log("split_json", "start", rank=rank_env, extra=f"path={train_ids_json}")
        jp = _resolve_path_relative_to_config(str(train_ids_json), args.config)
        if not jp.is_file():
            raise FileNotFoundError(f"train_ids_json 不存在: {jp}")
        split_meta = _load_split_json(jp)
        train_id_list = [str(x) for x in (split_meta.get("train_ids") or [])]
        test_id_list = [str(x) for x in (split_meta.get("test_ids") or [])]
        _startup_phase_log(
            "split_json",
            "done",
            rank=rank_env,
            extra=(
                f"train_ids={len(train_id_list)} test_ids={len(test_id_list)} "
                f"elapsed={time.perf_counter() - startup_t0:.3f}s"
            ),
        )
        sr_json = str(split_meta.get("split_root") or "").strip()
        if sr_json and not (args.split_root or "").strip() and not cfg_data.get("split_root"):
            split_root = sr_json
        lm = split_meta.get("lmdb_env_subdir")
        if lm and not cfg_data.get("lmdb_env_subdir"):
            lmdb_env_subdir = str(lm).strip() or lmdb_env_subdir

    if args.max_samples is not None:
        max_samples = int(args.max_samples)
    elif cfg_train.get("max_samples") is not None:
        max_samples = int(cfg_train["max_samples"])
    elif bool(cfg_train.get("debug_overfit")) and cfg_train.get("overfit_num_samples") is not None:
        max_samples = int(cfg_train["overfit_num_samples"])
    else:
        max_samples = 0

    phase_t0 = time.perf_counter()
    _startup_phase_log("ddp_init", "start", rank=rank_env)
    distributed, rank, local_rank, world_size = _init_distributed()
    if bool(getattr(args, "ddp", False)) and not distributed:
        raise ValueError("--ddp 已指定但未检测到 WORLD_SIZE>1，请使用 torchrun 启动")

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP 训练需要 CUDA")
        # 先绑定当前进程的本地 GPU，再初始化进程组，避免后续 collective 使用错误设备。
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_available():
            raise RuntimeError("torch.distributed 不可用")
        if not dist.is_initialized():
            backend = str(os.environ.get("DIST_BACKEND", "nccl")).strip() or "nccl"
            init_kwargs: Dict[str, Any] = {"backend": backend, "init_method": "env://"}
            # torch 新版本支持 device_id；支持时可减少 "No device id ..." 警告。
            try:
                dist.init_process_group(**init_kwargs, device_id=device)
            except TypeError:
                dist.init_process_group(**init_kwargs)
    else:
        device_str = str(_cli_or_pick_cfg(args.device, cfg_train, "device", "cuda"))
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        elif device_str in ("cuda", "auto"):
            picked_gpu = _auto_pick_least_utilized_gpu()
            if picked_gpu is None:
                picked_gpu = 0
            device_str = f"cuda:{picked_gpu}"
        device = torch.device(device_str)
    _startup_phase_log(
        "ddp_init",
        "done",
        rank=rank,
        extra=(
            f"distributed={distributed} world_size={world_size} device={device} "
            f"elapsed={time.perf_counter() - phase_t0:.3f}s"
        ),
    )

    use_shared_cache = str(cache_backend or "").strip().lower() in ("shared_memmap", "memmap", "shared")
    if use_shared_cache and prebuild_cache_first:
        if rank == 0:
            _prebuild_shared_memmap_cache_rank0(
                split_root=split_root,
                lmdb_env_subdir=lmdb_env_subdir,
                sample_pack_name=sample_pack_name,
                sample_storage_backend=sample_storage_backend,
                allow_missing_sample_ids=allow_missing_sample_ids,
                purify_force_mask_on_read=purify_force_mask_on_read,
                force_mask_purify_mode=force_mask_purify_mode,
                force_mask_keep_three_channels=force_mask_keep_three_channels,
                force_mask_single_channel=force_mask_single_channel,
                force_mask_binary_threshold=force_mask_binary_threshold,
                max_views=max_views,
                num_frames=num_frames,
                img_size=img_size,
                sample_ids=train_id_list,
                cache_root=cache_root,
                cache_version=cache_version,
                cache_force_rebuild=cache_force_rebuild,
                cache_log=cache_log,
                prebuild_num_workers=prebuild_num_workers,
            )
        if distributed:
            _safe_barrier(local_rank if distributed else None)
    effective_cache_readonly = bool(cache_readonly)
    if use_shared_cache and prebuild_cache_first and cache_readonly_after_prebuild:
        effective_cache_readonly = True

    phase_t0 = time.perf_counter()
    _startup_phase_log(
        "train_ds",
        "start",
        rank=rank,
        extra=(
            f"split_root={split_root} sample_ids={len(train_id_list or [])} "
            f"backend={sample_storage_backend} sample_pack_deep_validate=false "
            f"allow_missing_sample_ids={allow_missing_sample_ids}"
        ),
    )
    ds = LmdbGtDataset(
        split_root=split_root,
        lmdb_env_subdir=lmdb_env_subdir,
        sample_pack_name=sample_pack_name,
        sample_storage_backend=sample_storage_backend,
        sample_pack_deep_validate=False,
        allow_missing_sample_ids=allow_missing_sample_ids,
        max_views=max_views,
        num_frames=(None if num_frames <= 0 else num_frames),
        img_size=(None if img_size <= 0 else img_size),
        return_action_name=True,
        sample_ids=train_id_list,
        cache_backend=cache_backend,
        cache_root=cache_root,
        cache_version=cache_version,
        cache_readonly=effective_cache_readonly,
        cache_force_rebuild=False if effective_cache_readonly else cache_force_rebuild,
        cache_log=cache_log,
        purify_force_mask_on_read=purify_force_mask_on_read,
        force_mask_purify_mode=force_mask_purify_mode,
        force_mask_keep_three_channels=force_mask_keep_three_channels,
        force_mask_single_channel=force_mask_single_channel,
        force_mask_binary_threshold=force_mask_binary_threshold,
    )
    _startup_phase_log(
        "train_ds",
        "done",
        rank=rank,
        extra=(
            f"len={len(ds)} missing_skipped={len(getattr(ds, 'missing_sample_ids', []))} "
            f"elapsed={time.perf_counter() - phase_t0:.3f}s"
        ),
    )
    train_source = (
        Subset(ds, list(range(min(int(max_samples), len(ds)))))
        if int(max_samples) > 0
        else ds
    )
    eval_split = str(_pick(cfg_train, "eval_split", "train")).strip().lower()
    eval_use_distributed_sampler = bool(_pick(cfg_train, "eval_use_distributed_sampler", False))
    eval_sharded = eval_use_distributed_sampler and distributed
    reg_target_mean_cpu, reg_target_std_cpu, reg_target_count_cpu = _build_regression_target_stats(train_source)
    reg_target_mean = reg_target_mean_cpu.to(device)
    reg_target_std = reg_target_std_cpu.to(device)
    shuffle_train = not bool(cfg_train.get("overfit_no_shuffle", False))
    bs = int(_cli_or_pick_cfg(args.batch_size, cfg_train, "batch_size", 1))
    nw = int(_cli_or_pick_cfg(args.num_workers, cfg_train, "num_workers", 0))
    train_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_source,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle_train,
        )

    pin_memory = device.type == "cuda"
    nb = pin_memory
    prefetch_factor = max(2, int(_pick(cfg_train, "prefetch_factor", 2)))
    _dl_extras: Dict[str, Any] = {}
    if nw > 0:
        # rank0-only eval 时，其它 rank 会在 barrier 等待；训练 DataLoader 的 persistent workers
        # 在 sample_pack/网络存储场景下更容易在下一轮首 batch 卡住，表现成 eval 后的 DDP broadcast 超时。
        _dl_extras["persistent_workers"] = not (distributed and not eval_sharded)
        _dl_extras["prefetch_factor"] = prefetch_factor

    loader = DataLoader(
        train_source,
        batch_size=bs,
        shuffle=(shuffle_train and not distributed),
        sampler=train_sampler,
        num_workers=nw,
        collate_fn=collate_lmdb_gt_batch,
        drop_last=False,
        pin_memory=pin_memory,
        **_dl_extras,
    )

    if eval_split == "test":
        if not test_id_list:
            if rank == 0:
                _tqdm_log(
                    "[logic_train] WARN eval_split=test 但 split json 无 test_ids，"
                    "回退为 eval_split=train"
                )
            eval_split = "train"
    eval_ds = train_source
    if eval_split == "test" and test_id_list:
        phase_t0 = time.perf_counter()
        _startup_phase_log(
            "test_ds",
            "start",
            rank=rank,
            extra=(
                f"split_root={split_root} sample_ids={len(test_id_list)} "
                f"backend={sample_storage_backend} sample_pack_deep_validate=false "
                f"allow_missing_sample_ids={allow_missing_sample_ids}"
            ),
        )
        ds_eval = LmdbGtDataset(
            split_root=split_root,
            lmdb_env_subdir=lmdb_env_subdir,
            sample_pack_name=sample_pack_name,
            sample_storage_backend=sample_storage_backend,
            sample_pack_deep_validate=False,
            allow_missing_sample_ids=allow_missing_sample_ids,
            max_views=max_views,
            num_frames=(None if num_frames <= 0 else num_frames),
            img_size=(None if img_size <= 0 else img_size),
            return_action_name=True,
            sample_ids=test_id_list,
            action_to_id=ds.action_to_id,
            cache_backend=cache_backend,
            cache_root=cache_root,
            cache_version=cache_version,
            cache_readonly=effective_cache_readonly,
            cache_force_rebuild=False if effective_cache_readonly else cache_force_rebuild,
            cache_log=cache_log,
            purify_force_mask_on_read=purify_force_mask_on_read,
            force_mask_purify_mode=force_mask_purify_mode,
            force_mask_keep_three_channels=force_mask_keep_three_channels,
            force_mask_single_channel=force_mask_single_channel,
            force_mask_binary_threshold=force_mask_binary_threshold,
        )
        _startup_phase_log(
            "test_ds",
            "done",
            rank=rank,
            extra=(
                f"len={len(ds_eval)} missing_skipped={len(getattr(ds_eval, 'missing_sample_ids', []))} "
                f"elapsed={time.perf_counter() - phase_t0:.3f}s"
            ),
        )
        eval_ds = ds_eval
    eval_sampler_eval: Optional[DistributedSampler] = None
    if eval_sharded:
        eval_sampler_eval = DistributedSampler(
            eval_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    eval_loader: Optional[DataLoader] = None
    if eval_sharded:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=bs,
            shuffle=False,
            sampler=eval_sampler_eval,
            num_workers=nw,
            collate_fn=collate_lmdb_gt_batch,
            pin_memory=pin_memory,
            **_dl_extras,
        )
    elif rank == 0:
        # rank0-only eval 不需要为所有 rank 常驻一套 eval workers；
        # 否则多卡下会额外占满大量 worker，容易在 epoch 切换时把数据侧拖住。
        eval_loader = DataLoader(
            eval_ds,
            batch_size=bs,
            shuffle=False,
            sampler=None,
            num_workers=0,
            collate_fn=collate_lmdb_gt_batch,
            pin_memory=pin_memory,
        )

    if bool(getattr(args, "no_amp", False)):
        use_amp = False
    elif device.type != "cuda":
        use_amp = False
    else:
        use_amp = bool(_pick(cfg_train, "use_amp", True))
    legacy_debug_timing = bool(args.debug_timing) or bool(_pick(cfg_train, "debug_timing", False))
    profile_timing = bool(args.profile_timing) or bool(_pick(cfg_train, "profile_timing", False)) or legacy_debug_timing
    if args.profile_timing_every is not None:
        profile_timing_every = max(1, int(args.profile_timing_every))
    elif args.debug_timing_every_steps is not None:
        profile_timing_every = max(1, int(args.debug_timing_every_steps))
    elif ("profile_timing_every" in cfg_train) and (cfg_train.get("profile_timing_every") is not None):
        profile_timing_every = max(1, int(_pick(cfg_train, "profile_timing_every", 50)))
    else:
        profile_timing_every = max(1, int(_pick(cfg_train, "debug_timing_every_steps", 50)))
    if args.timing_log_first_n is not None:
        timing_log_first_n = max(0, int(args.timing_log_first_n))
    else:
        timing_log_first_n = max(0, int(_pick(cfg_train, "timing_log_first_n", 5)))
    if args.timing_log_every is not None:
        timing_log_every = int(args.timing_log_every)
    else:
        timing_log_every = int(_pick(cfg_train, "timing_log_every", 50))
    timing_recent_window = max(1, int(_pick(cfg_train, "timing_recent_window", 50)))
    timing_ema_alpha = float(_pick(cfg_train, "timing_ema_alpha", 0.1))
    timing_tb = bool(_pick(cfg_train, "timing_tensorboard", True))

    if num_frames > 0:
        model_num_frames = int(num_frames)
    else:
        # full-frames 模式下，使用第一个样本帧数作为模型固定时间长度
        sample0 = ds[0]
        model_num_frames = int(sample0["rgb"].shape[2])

    model_arch = str(_pick(cfg_model, "arch", "logic_v1")).strip().lower()
    if model_arch in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
        phase_t0 = time.perf_counter()
        _startup_phase_log("dino_prewarm", "start", rank=rank)
        hub_dir = _configure_torchhub_dir(_resolve_torchhub_dir(cfg_model))
        do_prewarm = bool(_pick(cfg_train, "torchhub_prewarm", True))
        require_cache = bool(_pick(cfg_train, "torchhub_require_cache", True))
        if distributed and do_prewarm:
            if rank == 0:
                if require_cache:
                    _prewarm_torchhub_dino(cfg_model, hub_dir)
            _safe_barrier(local_rank if distributed else None)
        elif (not distributed) and do_prewarm:
            if require_cache:
                _prewarm_torchhub_dino(cfg_model, hub_dir)
        _startup_phase_log(
            "dino_prewarm",
            "done",
            rank=rank,
            extra=f"elapsed={time.perf_counter() - phase_t0:.3f}s",
        )
    if model_arch in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
        phase_t0 = time.perf_counter()
        _startup_phase_log("dino_load", "start", rank=rank, extra=f"arch={model_arch}")
        model = LogicPhysModel2(
            num_views=max_views,
            in_channels=int(_pick(cfg_model, "in_channels", 3)),
            num_frames=int(_pick(cfg_model, "num_frames", model_num_frames)),
            img_size=int(_pick(cfg_model, "img_size", img_size if img_size > 0 else 224)),
            num_targets=4,
            num_actions=ds.num_actions,
            dec_h=dec_h,
            dec_w=dec_w,
            fusion_dim=int(_pick(cfg_model, "fusion_dim", 512)),
            fusion_heads=int(_pick(cfg_model, "fusion_heads", 8)),
            head_dropout=float(_pick(cfg_model, "head_dropout", 0.1)),
            use_uncertainty=bool(_pick(cfg_model, "use_uncertainty", False)),
            bottleneck_dim=int(_pick(cfg_model, "bottleneck_dim", 128)),
            dino_backbone_name=str(_pick(cfg_model, "dino_backbone_name", "dinov2_vits14")),
            dino_backbone_pretrained=bool(_pick(cfg_model, "dino_backbone_pretrained", True)),
            dino_backbone_source=str(_pick(cfg_model, "dino_backbone_source", "torchhub")),
            dino_out_dim=int(_pick(cfg_model, "dino_out_dim", 384)),
            temporal_adapter_type=str(_pick(cfg_model, "temporal_adapter_type", "transformer")),
            temporal_adapter_layers=int(_pick(cfg_model, "temporal_adapter_layers", 2)),
            temporal_adapter_heads=int(_pick(cfg_model, "temporal_adapter_heads", 6)),
            temporal_adapter_dropout=float(_pick(cfg_model, "temporal_adapter_dropout", 0.1)),
            frame_pool=str(_pick(cfg_model, "frame_pool", "mean")),
            freeze_backbone=bool(_pick(cfg_model, "freeze_backbone", True)),
            torchhub_dir=str(torch.hub.get_dir()),
            dino_torchhub_repo=str(_pick(cfg_model, "dino_torchhub_repo", "facebookresearch/dinov2:main")),
            dino_force_reload=bool(_pick(cfg_model, "dino_force_reload", False)),
            dino_trust_repo=bool(_pick(cfg_model, "dino_trust_repo", True)),
            dino_skip_validation=bool(_pick(cfg_model, "dino_skip_validation", True)),
            dino_hub_verbose=bool(_pick(cfg_model, "dino_hub_verbose", False)),
            dino_log_torchhub_dir=bool(_pick(cfg_model, "dino_log_torchhub_dir", False)),
            field_head_mode=str(_pick(cfg_model, "field_head_mode", "independent")),
            field_token_dim=int(_pick(cfg_model, "field_token_dim", 512)),
            field_base_channels=int(_pick(cfg_model, "field_base_channels", 128)),
            field_shared_channels=int(_pick(cfg_model, "field_shared_channels", 64)),
            field_temporal_layers=int(_pick(cfg_model, "field_temporal_layers", 0)),
            field_spatial_channels=int(_pick(cfg_model, "field_spatial_channels", 0)),
            field_use_multiscale_spatial=bool(_pick(cfg_model, "field_use_multiscale_spatial", False)),
            field_use_shared_task_phys=bool(_pick(cfg_model, "field_use_shared_task_phys", False)),
            field_use_geometry_residual=bool(_pick(cfg_model, "field_use_geometry_residual", False)),
            field_sequential_stress=bool(_pick(cfg_model, "field_sequential_stress", False)),
            use_stress_spatial_enhancer=bool(_pick(cfg_model, "use_stress_spatial_enhancer", False)),
            use_flow_spatial_enhancer=cfg_model.get("use_flow_spatial_enhancer"),
            flow_output_scale_init=float(_pick(cfg_model, "flow_output_scale_init", 0.5)),
            flow_output_bias_init=float(_pick(cfg_model, "flow_output_bias_init", -0.5)),
            force_out_channels=int(_pick(cfg_model, "force_out_channels", 3)),
            field_use_patch_tokens=bool(_pick(cfg_model, "field_use_patch_tokens", False)),
            field_patch_dim=int(_pick(cfg_model, "field_patch_dim", 256)),
            param_chain_mode=str(_pick(cfg_model, "param_chain_mode", "baseline")),
            param_token_dim=int(_pick(cfg_model, "param_token_dim", 256)),
            param_mixer_layers=int(_pick(cfg_model, "param_mixer_layers", 2)),
            param_mixer_heads=int(_pick(cfg_model, "param_mixer_heads", 8)),
            param_mixer_dropout=float(
                _pick(cfg_model, "param_mixer_dropout", _pick(cfg_model, "head_dropout", 0.1))
            ),
            param_use_rgb_static=bool(_pick(cfg_model, "param_use_rgb_static", True)),
            param_use_rgb_residual=bool(_pick(cfg_model, "param_use_rgb_residual", True)),
            param_use_masked_field_tokens=bool(_pick(cfg_model, "param_use_masked_field_tokens", True)),
        ).to(device)
        _startup_phase_log(
            "dino_load",
            "done",
            rank=rank,
            extra=f"elapsed={time.perf_counter() - phase_t0:.3f}s",
        )
    else:
        model = LogicPhysModel(
            num_views=max_views,
            in_channels=int(_pick(cfg_model, "in_channels", 3)),
            num_frames=int(_pick(cfg_model, "num_frames", model_num_frames)),
            img_size=int(_pick(cfg_model, "img_size", img_size if img_size > 0 else 224)),
            num_targets=4,
            num_actions=ds.num_actions,
            dec_h=dec_h,
            dec_w=dec_w,
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
        ).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    base_lr = float(_cli_or_pick_cfg(args.lr, cfg_train, "lr", 3e-4))
    stage1_field_lr = float(_pick(cfg_train, "stage1_field_lr", 5e-5))
    use_param_groups = bool(_pick(cfg_train, "use_param_groups", False))
    backbone_lr: Optional[float] = None
    head_lr: Optional[float] = None
    wd_backbone: Optional[float] = None
    wd_head: Optional[float] = None
    if use_param_groups and model_arch in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
        head_lr = float(_pick(cfg_train, "head_lr", base_lr))
        backbone_lr = float(_pick(cfg_train, "backbone_lr", head_lr * 0.1))
        wd_head = float(_pick(cfg_train, "weight_decay_head", 0.01))
        wd_backbone = float(_pick(cfg_train, "weight_decay_backbone", wd_head))

        mopt = _unwrap_model(model)
        backbone_params: List[torch.nn.Parameter] = []
        head_params: List[torch.nn.Parameter] = []
        for n, p in mopt.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("frame_encoder."):
                backbone_params.append(p)
            else:
                head_params.append(p)

        param_groups: List[Dict[str, Any]] = []
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": wd_backbone}
            )
        if head_params:
            param_groups.append(
                {"params": head_params, "lr": head_lr, "weight_decay": wd_head}
            )
        opt = torch.optim.AdamW(param_groups, lr=head_lr)
    else:
        if use_param_groups and rank == 0 and model_arch not in ("logic_v2_dino", "logic_v2", "dino", "dinov2"):
            _tqdm_log(
                f"[logic_train] WARN use_param_groups=true 但 model_arch={model_arch} 非 dino，回退为单学习率 AdamW。"
            )
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
        )
    for pg in opt.param_groups:
        pg["initial_lr"] = float(pg.get("lr", base_lr))
    scaler = GradScaler(enabled=use_amp)

    loss_reg = Arch4RegressionLoss(Arch4LossConfig())
    phys_cfg = PhysicsConsistencyConfig(
        # 仅保留 lambda_phys 作为总权重，三个 physics 子项固定等权。
        lambda_stress_flow_consistency=1.0,
        lambda_force_stress_consistency=1.0,
        lambda_force_flow_consistency=1.0,
        use_pred_force_mask=bool(_pick(cfg_train, "use_pred_force_mask_for_phys", args.use_pred_force_mask_for_phys)),
    )

    lambda_stress = float(_cli_or_pick_cfg(args.lambda_stress, cfg_train, "lambda_stress", 10.0))
    lambda_flow = float(_cli_or_pick_cfg(args.lambda_flow, cfg_train, "lambda_flow", 10.0))
    lambda_force = float(_cli_or_pick_cfg(args.lambda_force, cfg_train, "lambda_force", 10.0))
    lambda_action = float(_cli_or_pick_cfg(args.lambda_action, cfg_train, "lambda_action", 1.0))
    lambda_phys = float(_cli_or_pick_cfg(args.lambda_phys, cfg_train, "lambda_phys", 0.001))
    use_action_loss = bool(_pick(cfg_train, "use_action_loss", True))
    use_phys_loss = bool(_pick(cfg_train, "use_phys_loss", True))
    if args.disable_action_loss:
        use_action_loss = False
    if args.disable_phys_loss:
        use_phys_loss = False
    # 当前训练默认始终启用 regression / field 三个主监督；
    # stage 切换时再按 epoch 覆盖。这里显式声明，供 eval 分流逻辑复用。
    use_reg_loss = True
    use_stress_loss = True
    use_flow_loss = True
    use_force_loss = True
    effective_lambda_action = float(lambda_action) if use_action_loss else 0.0
    effective_lambda_phys = float(lambda_phys) if use_phys_loss else 0.0
    stage1_field_epochs = int(_cli_or_pick_cfg(args.stage1_field_epochs, cfg_train, "stage1_field_epochs", 0))
    stage_flow_force_epochs = int(_pick(cfg_train, "stage_flow_force_epochs", 0))
    stage_stress_epochs = int(_pick(cfg_train, "stage_stress_epochs", 0))
    stage_joint_epochs = int(_pick(cfg_train, "stage_joint_epochs", 0))
    stage_stress_lr = float(_pick(cfg_train, "stage_stress_lr", 5e-5))
    _sjl = cfg_train.get("stage_joint_lr")
    stage_joint_lr = float(_sjl) if _sjl is not None else float(base_lr)
    _sfflr = cfg_train.get("stage_flow_force_lr")
    stage_flow_force_lr = float(_sfflr) if _sfflr is not None else float(stage1_field_lr)
    _splr = cfg_train.get("stage_param_lr")
    stage_param_lr = float(_splr) if _splr is not None else float(base_lr)
    lambda_stress_edge = float(_pick(cfg_train, "lambda_stress_edge", 0.0))
    lambda_flow_edge = float(_pick(cfg_train, "lambda_flow_edge", 0.0))
    edge_boost = float(_pick(cfg_train, "edge_boost", 4.0))
    force_bce_weight = float(_pick(cfg_train, "force_bce_weight", 0.7))
    force_dice_weight = float(_pick(cfg_train, "force_dice_weight", 0.3))
    stage1_use_phys_loss = bool(_pick(cfg_train, "stage1_use_phys_loss", False))
    param_field_source_mode = str(_pick(cfg_train, "param_field_source_mode", "pred")).strip().lower()
    param_gt_field_ratio = float(_pick(cfg_train, "param_gt_field_ratio", 0.0))
    param_gt_field_ratio = min(1.0, max(0.0, param_gt_field_ratio))
    _param_apply_raw = _pick(cfg_train, "param_gt_field_apply_to", ["stress", "flow"])
    if isinstance(_param_apply_raw, str):
        param_gt_field_apply_to = {x.strip().lower() for x in _param_apply_raw.split(",") if x.strip()}
    elif isinstance(_param_apply_raw, (list, tuple, set)):
        param_gt_field_apply_to = {str(x).strip().lower() for x in _param_apply_raw if str(x).strip()}
    else:
        param_gt_field_apply_to = {"stress", "flow"}
    param_gt_field_granularity = str(_pick(cfg_train, "param_gt_field_granularity", "sample")).strip().lower()
    if param_gt_field_granularity != "sample":
        raise ValueError(f"unsupported param_gt_field_granularity: {param_gt_field_granularity}")
    if param_field_source_mode not in ("pred", "mixed_gt_pred"):
        raise ValueError(f"unsupported param_field_source_mode: {param_field_source_mode}")
    object_mask_fg_weight = float(_cli_or_pick_cfg(args.object_mask_fg_weight, cfg_train, "object_mask_fg_weight", 10.0))
    object_mask_bg_weight = float(_cli_or_pick_cfg(args.object_mask_bg_weight, cfg_train, "object_mask_bg_weight", 1.0))
    object_mask_bg_black = bool(
        int(_cli_or_pick_cfg(args.object_mask_bg_black, cfg_train, "object_mask_bg_black", 1))
    )

    train_out_cli = str(args.output_root or "").strip()
    if train_out_cli:
        train_out = train_out_cli
    else:
        train_out = str(_pick(cfg_train, "output_root", "") or "").strip()
    if train_out:
        output_root = Path(train_out).expanduser().resolve()
    else:
        output_root = (Path("logic_model") / "output").resolve()
    ckpt_dir = output_root / "checkpoints"
    eval_dir = output_root / "eval"
    tensorboard_dir = output_root / "tensorboard"
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    best_eval_loss_by_stage: Dict[str, float] = {
        "field": float("inf"),
        "flow_force": float("inf"),
        "stress": float("inf"),
        "joint": float("inf"),
        "param": float("inf"),
    }
    best_eval_epoch_by_stage: Dict[str, int] = {
        "field": 0,
        "flow_force": 0,
        "stress": 0,
        "joint": 0,
        "param": 0,
    }
    tb_writer: Optional[SummaryWriter] = None
    if rank == 0:
        tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))

    ckpt_cfg = cfg_train.get("checkpoint")
    if not isinstance(ckpt_cfg, dict):
        ckpt_cfg = {}
    if args.save_every_epochs is not None:
        save_every_epochs = max(1, int(args.save_every_epochs))
    else:
        sv_e = ckpt_cfg.get("save_every_epochs")
        if sv_e is not None:
            save_every_epochs = max(1, int(sv_e))
        else:
            save_every_epochs = max(1, int(100000))
    resume_from_raw = str(
        args.resume_from
        if args.resume_from is not None
        else (ckpt_cfg.get("resume_from") if ckpt_cfg.get("resume_from") is not None else "")
    ).strip()
    resume_allow_partial = bool(_pick(cfg_train, "resume_allow_partial", False))
    resume_optimizer_state = bool(
        _pick(cfg_train, "resume_optimizer_state", (not resume_allow_partial))
    )
    resume_ckpt_path: Optional[Path] = None
    resume_epoch_1based = 0
    resume_stage_name: Optional[str] = None
    if resume_from_raw:
        resume_ckpt_path = _resolve_path_relative_to_config(resume_from_raw, args.config)
        if not resume_ckpt_path.is_file():
            raise FileNotFoundError(f"resume checkpoint 不存在: {resume_ckpt_path}")
        try:
            resume_ckpt = torch.load(str(resume_ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:
            resume_ckpt = torch.load(str(resume_ckpt_path), map_location="cpu")
        if not isinstance(resume_ckpt, dict):
            raise ValueError(f"resume checkpoint 顶层必须是 dict: {resume_ckpt_path}")
        model_state = resume_ckpt.get("model_state")
        if not isinstance(model_state, dict):
            raise KeyError(f"resume checkpoint 缺少 model_state: {resume_ckpt_path}")
        if resume_allow_partial:
            model_ref = _unwrap_model(model)
            current_state = model_ref.state_dict()
            allowed_missing_prefixes = (
                "shared_field_param_encoder.",
                "param_field_fusion.0.weight",
                "phys_token_proj.",
                "rgb_static_proj.",
                "rgb_residual_phys_proj.",
                "rgb_residual_proj.",
                "field_token_proj.",
                "stress_temporal_param_encoder.",
                "flow_temporal_param_encoder.",
                "strong_param_transformer.",
            )
            filtered_state = {}
            skipped_mismatch_keys = []
            unexpected_keys = []
            for key, value in model_state.items():
                cur = current_state.get(key)
                if cur is None:
                    unexpected_keys.append(key)
                    continue
                if getattr(cur, "shape", None) != getattr(value, "shape", None):
                    if any(key.startswith(prefix) for prefix in allowed_missing_prefixes):
                        skipped_mismatch_keys.append(key)
                        continue
                    raise RuntimeError(
                        f"partial resume 发现非预期 shape mismatch: {key} ckpt={tuple(value.shape)} model={tuple(cur.shape)}"
                    )
                filtered_state[key] = value
            incompatible = model_ref.load_state_dict(filtered_state, strict=False)
            missing_keys = list(getattr(incompatible, "missing_keys", []))
            unexpected_keys.extend(list(getattr(incompatible, "unexpected_keys", [])))
            bad_missing = [
                k for k in missing_keys if not any(k.startswith(prefix) for prefix in allowed_missing_prefixes)
            ]
            if bad_missing:
                raise RuntimeError(
                    f"partial resume 缺少非预期参数，请检查结构兼容性: {bad_missing}"
                )
            if unexpected_keys:
                raise RuntimeError(
                    f"partial resume 出现 checkpoint 多余参数，请检查结构兼容性: {unexpected_keys}"
                )
            if rank == 0 and missing_keys:
                _tqdm_log(
                    f"[logic_train] partial resume: 新模块将随机初始化 missing_keys={missing_keys}"
                )
            if rank == 0 and skipped_mismatch_keys:
                _tqdm_log(
                    f"[logic_train] partial resume: 跳过 shape mismatch keys={skipped_mismatch_keys}"
                )
        else:
            _unwrap_model(model).load_state_dict(model_state, strict=True)
        opt_state = resume_ckpt.get("optimizer_state")
        if resume_optimizer_state and isinstance(opt_state, dict):
            opt.load_state_dict(opt_state)
        elif rank == 0:
            if resume_optimizer_state:
                _tqdm_log(f"[logic_train] WARN resume checkpoint 无 optimizer_state：{resume_ckpt_path}")
            else:
                _tqdm_log("[logic_train] skip loading optimizer_state by config")
        scaler_state = resume_ckpt.get("scaler_state")
        if use_amp and resume_optimizer_state and isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        resume_epoch_1based = max(0, int(resume_ckpt.get("epoch", 0) or 0))
        resume_stage_name = (
            None if resume_ckpt.get("stage_name") is None else str(resume_ckpt.get("stage_name"))
        )
        if rank == 0:
            _tqdm_log(
                f"[logic_train] resume loaded ckpt={resume_ckpt_path} "
                f"epoch={resume_epoch_1based} stage={resume_stage_name or '<unknown>'}"
            )
        if distributed:
            _safe_barrier(local_rank if distributed else None)

    ev_e = cfg_train.get("eval_every_epochs")
    _eval_from_quick_eval = False
    if ev_e is None:
        qe = cfg_train.get("quick_eval")
        if isinstance(qe, dict) and qe.get("every_epochs") is not None:
            ev_e = qe["every_epochs"]
            _eval_from_quick_eval = True
    if args.eval_every_epochs is not None:
        eval_every_epochs = max(1, int(args.eval_every_epochs))
    else:
        eval_every_epochs = max(1, int(ev_e if ev_e is not None else 100))
    if rank == 0 and _eval_from_quick_eval:
        _tqdm_log(
            "[logic_train] WARN train.quick_eval.every_epochs 已废弃，请改用 train.eval_every_epochs",
        )
    eval_loader_len = (
        len(eval_loader)
        if eval_loader is not None
        else max(1, int(math.ceil(float(len(eval_ds)) / float(max(1, bs)))))
    )
    if args.eval_batches is not None:
        _eb = int(args.eval_batches)
        eval_max_batches = eval_loader_len if _eb <= 0 else max(1, _eb)
    elif cfg_train.get("eval_max_batches") is not None:
        _emi = int(cfg_train["eval_max_batches"])
        eval_max_batches = eval_loader_len if _emi <= 0 else max(1, _emi)
    else:
        eval_max_batches = eval_loader_len
    eval_video_num_samples = max(0, int(_pick(cfg_train, "eval_video_num_samples", 0)))
    eval_train_video_num_samples = max(0, int(_pick(cfg_train, "eval_train_video_num_samples", 0)))
    eval_param_scatter = bool(_pick(cfg_train, "eval_param_scatter", False))
    eval_param_scatter_log = bool(_pick(cfg_train, "eval_param_scatter_log", False))
    eval_scatter_pad_frac = float(_pick(cfg_train, "eval_scatter_pad_frac", 0.02))
    eval_scatter_axis_percentiles_raw = _pick(cfg_train, "eval_scatter_axis_percentiles", None)
    eval_scatter_axis_percentiles: Optional[Tuple[float, float]] = None
    if isinstance(eval_scatter_axis_percentiles_raw, (list, tuple)) and len(eval_scatter_axis_percentiles_raw) >= 2:
        eval_scatter_axis_percentiles = (
            float(eval_scatter_axis_percentiles_raw[0]),
            float(eval_scatter_axis_percentiles_raw[1]),
        )
    epochs = int(_cli_or_pick_cfg(args.epochs, cfg_train, "epochs", 1000))
    start_epoch = int(resume_epoch_1based)
    if start_epoch > epochs:
        raise ValueError(
            f"resume epoch ({start_epoch}) 大于总 epochs ({epochs})，请增大 train.epochs 或改用更早 checkpoint"
        )
    if rank == 0:
        _tqdm_log(
            f"[logic_train] start arch={model_arch} train={len(ds)} eval_split={eval_split} "
            f"epochs={epochs} ddp={distributed} world_size={world_size} amp={use_amp} "
            f"lr={base_lr} out={output_root}"
        )
        _tqdm_log(
            f"[logic_train] loss_weight reg=1 stress={lambda_stress} flow={lambda_flow} "
            f"force={lambda_force} action={effective_lambda_action} phys={effective_lambda_phys} "
            f"stage1_field_epochs={stage1_field_epochs} stage1_field_lr={stage1_field_lr} "
            f"stage_flow_force_epochs={stage_flow_force_epochs} stage_stress_epochs={stage_stress_epochs} "
            f"stage_joint_epochs={stage_joint_epochs} stage_joint_lr={stage_joint_lr} "
            f"stage_flow_force_lr={stage_flow_force_lr} stage_stress_lr={stage_stress_lr} "
            f"stage_param_lr={stage_param_lr} "
            f"lambda_stress_edge={lambda_stress_edge} lambda_flow_edge={lambda_flow_edge} edge_boost={edge_boost} "
            f"stage1_use_phys={stage1_use_phys_loss}"
        )
        _tqdm_log(
            f"[logic_train] eval_visual video_samples={eval_video_num_samples} "
            f"train_video_samples={eval_train_video_num_samples} "
            f"param_scatter={eval_param_scatter} param_scatter_log={eval_param_scatter_log} "
            f"scatter_pct={eval_scatter_axis_percentiles}"
        )
        _tqdm_log(
            f"[logic_train] param_field_source mode={param_field_source_mode} "
            f"gt_ratio={param_gt_field_ratio} apply_to={sorted(param_gt_field_apply_to)} "
            f"granularity={param_gt_field_granularity}"
        )

    if rank == 0:
        (output_root / "action_to_id.json").write_text(
            json.dumps(ds.action_to_id, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_root / "run_config.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "cfg_data": cfg_data,
                    "cfg_model": cfg_model,
                    "cfg_train": cfg_train,
                    "resolved": {
                        "distributed": distributed,
                        "world_size": world_size,
                        "rank": rank,
                        "split_root": split_root,
                        "lmdb_env_subdir": lmdb_env_subdir,
                        "train_ids_json": str(train_ids_json or ""),
                        "n_train_ids": len(train_id_list or []),
                        "n_test_ids": len(test_id_list or []),
                        "max_samples": int(max_samples),
                        "eval_split": eval_split,
                        "max_views": max_views,
                        "num_frames": num_frames,
                        "img_size": img_size,
                        "dec_h": dec_h,
                        "dec_w": dec_w,
                        "device": str(device),
                        "use_amp_fp16": bool(use_amp),
                        "profile_timing": bool(profile_timing),
                        "profile_timing_every": int(profile_timing_every),
                        "timing_log_first_n": int(timing_log_first_n),
                        "timing_log_every": int(timing_log_every),
                        "dataloader_num_workers": int(nw),
                        "dataloader_pin_memory": bool(pin_memory),
                        "dataloader_persistent_workers": bool(nw > 0),
                        "dataloader_prefetch_factor": int(prefetch_factor) if nw > 0 else None,
                        "model_num_frames": int(_unwrap_model(model).num_frames),
                        "model_arch": str(model_arch),
                        "optimizer_use_param_groups": bool(use_param_groups),
                        "optimizer_base_lr": float(base_lr),
                        "optimizer_backbone_lr": (None if backbone_lr is None else float(backbone_lr)),
                        "optimizer_head_lr": (None if head_lr is None else float(head_lr)),
                        "optimizer_weight_decay_backbone": (None if wd_backbone is None else float(wd_backbone)),
                        "optimizer_weight_decay_head": (None if wd_head is None else float(wd_head)),
                        "output_root": str(output_root),
                        "tensorboard_dir": str(tensorboard_dir),
                        "epochs": int(epochs),
                        "start_epoch": int(start_epoch),
                        "resume_checkpoint": (None if resume_ckpt_path is None else str(resume_ckpt_path)),
                        "resume_stage_name": resume_stage_name,
                        "save_every_epochs": int(save_every_epochs),
                        "eval_every_epochs": int(eval_every_epochs),
                        "eval_max_batches": int(eval_max_batches),
                        "eval_sharded": bool(eval_sharded),
                        "lambda_stress": float(lambda_stress),
                        "lambda_flow": float(lambda_flow),
                        "lambda_force": float(lambda_force),
                        "lambda_action": float(lambda_action),
                        "lambda_phys": float(lambda_phys),
                        "use_action_loss": bool(use_action_loss),
                        "use_phys_loss": bool(use_phys_loss),
                        "effective_lambda_action": float(effective_lambda_action),
                        "effective_lambda_phys": float(effective_lambda_phys),
                        "stage1_field_epochs": int(stage1_field_epochs),
                        "stage_flow_force_epochs": int(stage_flow_force_epochs),
                        "stage_stress_epochs": int(stage_stress_epochs),
                        "stage_joint_epochs": int(stage_joint_epochs),
                        "stage_joint_lr": float(stage_joint_lr),
                        "stage_flow_force_lr": float(stage_flow_force_lr),
                        "stage_stress_lr": float(stage_stress_lr),
                        "stage_param_lr": float(stage_param_lr),
                        "lambda_stress_edge": float(lambda_stress_edge),
                        "lambda_flow_edge": float(lambda_flow_edge),
                        "edge_boost": float(edge_boost),
                        "force_bce_weight": float(force_bce_weight),
                        "force_dice_weight": float(force_dice_weight),
                        "stage1_use_phys_loss": bool(stage1_use_phys_loss),
                        "param_field_source_mode": str(param_field_source_mode),
                        "param_gt_field_ratio": float(param_gt_field_ratio),
                        "param_gt_field_apply_to": sorted(param_gt_field_apply_to),
                        "param_gt_field_granularity": str(param_gt_field_granularity),
                        "reg_target_mean": [float(x) for x in reg_target_mean_cpu.tolist()],
                        "reg_target_std": [float(x) for x in reg_target_std_cpu.tolist()],
                        "reg_target_valid_counts": [int(round(float(x))) for x in reg_target_count_cpu.tolist()],
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    epoch_pbar = tqdm(
        range(start_epoch, epochs),
        desc="epoch",
        disable=rank != 0,
        dynamic_ncols=True,
        leave=True,
    )
    for epoch in epoch_pbar:
        stage_name = "joint"
        sff_i = int(stage_flow_force_epochs)
        sse_i = int(stage_stress_epochs)
        sjt_i = int(stage_joint_epochs)
        use_staged_schedule = (sff_i + sse_i + sjt_i) > 0
        if use_staged_schedule:
            e1 = int(epoch + 1)
            if e1 <= sff_i:
                stage_name = "flow_force"
            elif e1 <= sff_i + sse_i:
                stage_name = "stress"
            elif sjt_i > 0 and e1 <= sff_i + sse_i + sjt_i:
                stage_name = "joint"
            else:
                stage_name = "param"
        elif int(stage1_field_epochs) > 0:
            stage_name = "field" if (epoch + 1) <= int(stage1_field_epochs) else "param"
        if hasattr(_unwrap_model(model), "set_training_stage"):
            _unwrap_model(model).set_training_stage(stage_name)
        lr_scale = 1.0
        if use_staged_schedule:
            if stage_name == "flow_force":
                lr_scale = float(stage_flow_force_lr) / max(float(base_lr), 1e-12)
            elif stage_name == "stress":
                lr_scale = float(stage_stress_lr) / max(float(base_lr), 1e-12)
            elif stage_name == "joint":
                lr_scale = float(stage_joint_lr) / max(float(base_lr), 1e-12)
            elif stage_name == "param":
                lr_scale = float(stage_param_lr) / max(float(base_lr), 1e-12)
        elif stage_name == "field" and int(stage1_field_epochs) > 0:
            lr_scale = float(stage1_field_lr) / max(float(base_lr), 1e-12)
        for pg in opt.param_groups:
            pg["lr"] = float(pg.get("initial_lr", base_lr)) * lr_scale
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        mcore = _unwrap_model(model)
        use_reg_loss_epoch = True
        use_stress_loss_epoch = True
        use_flow_loss_epoch = True
        use_force_loss_epoch = True
        use_action_loss_epoch = bool(use_action_loss)
        use_phys_loss_epoch = bool(use_phys_loss)
        if use_staged_schedule:
            if stage_name == "flow_force":
                use_reg_loss_epoch = False
                use_stress_loss_epoch = False
                use_flow_loss_epoch = True
                use_force_loss_epoch = True
                use_action_loss_epoch = False
                use_phys_loss_epoch = bool(stage1_use_phys_loss)
            elif stage_name == "stress":
                use_reg_loss_epoch = False
                use_stress_loss_epoch = True
                use_flow_loss_epoch = False
                use_force_loss_epoch = False
                use_action_loss_epoch = False
                use_phys_loss_epoch = False
            elif stage_name == "joint":
                use_reg_loss_epoch = True
                use_stress_loss_epoch = True
                use_flow_loss_epoch = True
                use_force_loss_epoch = True
                use_action_loss_epoch = bool(use_action_loss)
                use_phys_loss_epoch = bool(use_phys_loss)
            elif stage_name == "param":
                use_stress_loss_epoch = False
                use_flow_loss_epoch = False
                use_force_loss_epoch = False
                use_action_loss_epoch = False
                use_phys_loss_epoch = False
        elif stage_name == "field":
            use_reg_loss_epoch = False
            use_action_loss_epoch = False
            use_phys_loss_epoch = bool(stage1_use_phys_loss)
        elif stage_name == "param":
            use_stress_loss_epoch = False
            use_flow_loss_epoch = False
            use_force_loss_epoch = False
            use_action_loss_epoch = False
            use_phys_loss_epoch = False

        eff_lambda_stress_epoch = float(lambda_stress) if use_stress_loss_epoch else 0.0
        eff_lambda_flow_epoch = float(lambda_flow) if use_flow_loss_epoch else 0.0
        eff_lambda_force_epoch = float(lambda_force) if use_force_loss_epoch else 0.0
        eff_lambda_action_epoch = float(lambda_action) if use_action_loss_epoch else 0.0
        eff_lambda_phys_epoch = float(lambda_phys) if use_phys_loss_epoch else 0.0
        train_sums = [0.0] * _REG_SUMS_LEN
        train_pbar = tqdm(
            loader,
            desc=f"train {epoch + 1}/{epochs}",
            disable=rank != 0,
            total=len(loader),
            leave=False,
            dynamic_ncols=True,
        )
        last_end = time.perf_counter()
        timing_tracker = _TimingTracker(window_size=timing_recent_window, ema_alpha=timing_ema_alpha)
        for bi, batch in enumerate(train_pbar):
            t_batch_ready = time.perf_counter()
            data_wait = t_batch_ready - last_end
            step_wall_start = t_batch_ready
            startup_data_wait = data_wait if bi == 0 else 0.0

            sampled_timing = bool(profile_timing and ((bi % profile_timing_every) == 0))
            use_cuda_event_timing = bool(sampled_timing and device.type == "cuda")
            ev_h2d_s = ev_h2d_e = None
            ev_prep_s = ev_prep_e = None
            ev_fwd_s = ev_fwd_e = None
            ev_bwd_s = ev_bwd_e = None
            ev_opt_s = ev_opt_e = None
            if use_cuda_event_timing:
                ev_h2d_s = torch.cuda.Event(enable_timing=True)
                ev_h2d_e = torch.cuda.Event(enable_timing=True)
                ev_prep_s = torch.cuda.Event(enable_timing=True)
                ev_prep_e = torch.cuda.Event(enable_timing=True)
                ev_fwd_s = torch.cuda.Event(enable_timing=True)
                ev_fwd_e = torch.cuda.Event(enable_timing=True)
                ev_bwd_s = torch.cuda.Event(enable_timing=True)
                ev_bwd_e = torch.cuda.Event(enable_timing=True)
                ev_opt_s = torch.cuda.Event(enable_timing=True)
                ev_opt_e = torch.cuda.Event(enable_timing=True)

            h2d_wall_start = time.perf_counter()
            if use_cuda_event_timing and ev_h2d_s is not None:
                ev_h2d_s.record()
            rgb = batch["rgb"].to(device, non_blocking=nb)
            stress_gt = batch["stress"].to(device, non_blocking=nb)
            flow_gt = batch["flow"].to(device, non_blocking=nb)
            force_gt = batch["force_mask"].to(device, non_blocking=nb)
            object_gt = batch["object_mask"].to(device, non_blocking=nb)
            params_gt_raw = batch["params"].to(device, non_blocking=nb)
            action_label = batch["action_label"].to(device, non_blocking=nb)
            if use_cuda_event_timing and ev_h2d_e is not None:
                ev_h2d_e.record()
            h2d_time_wall = time.perf_counter() - h2d_wall_start

            preprocess_wall_start = time.perf_counter()
            if use_cuda_event_timing and ev_prep_s is not None:
                ev_prep_s.record()
            # [B,V,3,T,H,W] -> [B,V,C,T,H,W]，兼容 in_channels
            if int(mcore.in_channels) == 1:
                x = rgb[:, :, :1, :, :, :]
            else:
                x = rgb[:, :, : int(mcore.in_channels), :, :, :]

            # 兼容 full-frames 数据读取：训练时对齐到模型固定 num_frames
            if int(x.shape[3]) != int(mcore.num_frames):
                x = _resample_time_bvcthw(x, int(mcore.num_frames))
                stress_gt = _resample_time_bvcthw(stress_gt, int(mcore.num_frames))
                flow_gt = _resample_time_bvcthw(flow_gt, int(mcore.num_frames))
                force_gt = _resample_time_bvcthw(force_gt, int(mcore.num_frames))
                object_gt = _resample_time_bvcthw(object_gt, int(mcore.num_frames))
            if use_cuda_event_timing and ev_prep_e is not None:
                ev_prep_e.record()
            preprocess_gpu_time_wall = time.perf_counter() - preprocess_wall_start

            forward_wall_start = time.perf_counter()
            if use_cuda_event_timing and ev_fwd_s is not None:
                ev_fwd_s.record()
            param_stress_use_gt = None
            param_flow_use_gt = None
            if (
                stage_name == "param"
                and param_field_source_mode == "mixed_gt_pred"
                and float(param_gt_field_ratio) > 0.0
            ):
                gt_field_mask = torch.rand(int(x.shape[0]), device=device) < float(param_gt_field_ratio)
                if "stress" in param_gt_field_apply_to:
                    param_stress_use_gt = gt_field_mask
                if "flow" in param_gt_field_apply_to:
                    param_flow_use_gt = gt_field_mask
            with _amp_autocast(use_amp=use_amp, device=device):
                out = _forward_model_batch(
                    model,
                    mcore,
                    x,
                    stage_name=stage_name,
                    object_gt=object_gt,
                    param_stress_gt=stress_gt,
                    param_flow_gt=flow_gt,
                    param_stress_use_gt=param_stress_use_gt,
                    param_flow_use_gt=param_flow_use_gt,
                )

            zero = out["stress_field_pred"].float().sum() * 0.0
            params_gt_safe, valid_mask = _sanitize_raw_params_and_valid_mask(params_gt_raw)
            gt_train_space = _to_target_params(params_gt_safe).float()
            pred_param = out["param_pred"].float()
            if use_reg_loss_epoch:
                pred_reg_std = _normalize_regression_targets(pred_param, reg_target_mean, reg_target_std)
                gt_reg_std = _normalize_regression_targets(gt_train_space, reg_target_mean, reg_target_std)
                loss_reg_part = loss_reg(pred_reg_std, gt_reg_std, out["logvar"].float(), valid_mask=valid_mask)
            else:
                loss_reg_part = zero

            loss_stress_recon = (
                _object_mask_weighted_field_loss(
                    out["stress_field_pred"].float(),
                    stress_gt.float(),
                    object_gt.float(),
                    fg_weight=object_mask_fg_weight,
                    bg_weight=object_mask_bg_weight,
                    bg_black=object_mask_bg_black,
                )
                if use_stress_loss_epoch
                else zero
            )
            loss_stress_edge = (
                weighted_edge_l1_loss_bvcthw(
                    out["stress_field_pred"].float(),
                    stress_gt.float(),
                    object_gt.float(),
                    edge_boost=float(edge_boost),
                    bg_weight=float(object_mask_bg_weight),
                )
                if use_stress_loss_epoch and float(lambda_stress_edge) > 0.0
                else zero
            )
            loss_stress_part = loss_stress_recon + float(lambda_stress_edge) * loss_stress_edge

            loss_flow_recon = (
                _object_mask_weighted_field_loss(
                    out["flow_field_pred"].float(),
                    flow_gt.float(),
                    object_gt.float(),
                    fg_weight=object_mask_fg_weight,
                    bg_weight=object_mask_bg_weight,
                    bg_black=object_mask_bg_black,
                )
                if use_flow_loss_epoch
                else zero
            )
            loss_flow_edge = (
                weighted_edge_l1_loss_bvcthw(
                    out["flow_field_pred"].float(),
                    flow_gt.float(),
                    object_gt.float(),
                    edge_boost=float(edge_boost),
                    bg_weight=float(object_mask_bg_weight),
                )
                if use_flow_loss_epoch and float(lambda_flow_edge) > 0.0
                else zero
            )
            loss_flow_part = loss_flow_recon + float(lambda_flow_edge) * loss_flow_edge

            loss_force_bce = zero
            loss_force_dice = zero
            if use_force_loss_epoch:
                if "force_logits" in out:
                    loss_force_bce = weighted_bce_with_logits_loss(
                        out["force_logits"].float(),
                        force_gt.float(),
                        object_gt.float(),
                        fg_weight=object_mask_fg_weight,
                        bg_weight=object_mask_bg_weight,
                        bg_black=object_mask_bg_black,
                    )
                    loss_force_dice = binary_dice_loss_with_logits(
                        out["force_logits"].float(),
                        force_gt.float(),
                        object_gt.float(),
                    )
                    loss_force_part = float(force_bce_weight) * loss_force_bce + float(force_dice_weight) * loss_force_dice
                else:
                    loss_force_part = _object_mask_weighted_field_loss(
                        out["force_pred"].float(),
                        force_gt.float(),
                        object_gt.float(),
                        fg_weight=object_mask_fg_weight,
                        bg_weight=object_mask_bg_weight,
                        bg_black=object_mask_bg_black,
                    )
            else:
                loss_force_part = zero

            if use_action_loss_epoch:
                action_ret = action_classification_loss(out["action_logits"].float(), action_label)
            else:
                action_ret = {"loss_action": zero, "action_acc": zero}
            if use_phys_loss_epoch:
                phys_ret = compute_physics_consistency_losses(
                    stress_pred=out["stress_field_pred"].float(),
                    flow_pred=out["flow_field_pred"].float(),
                    force_pred=out["force_pred"].float(),
                    force_gt=force_gt.float(),
                    cfg=phys_cfg,
                )
            else:
                phys_ret = {
                    "loss_stress_flow_consistency": zero,
                    "loss_force_stress_consistency": zero,
                    "loss_force_flow_consistency": zero,
                    "loss_phys_total": zero,
                }

            loss_total = _compose_total_loss(
                loss_reg_part=loss_reg_part,
                loss_stress_part=loss_stress_part,
                loss_flow_part=loss_flow_part,
                loss_force_part=loss_force_part,
                loss_action_part=action_ret["loss_action"],
                loss_phys_part=phys_ret["loss_phys_total"],
                lambda_stress=lambda_stress,
                lambda_flow=lambda_flow,
                lambda_force=lambda_force,
                lambda_action=lambda_action,
                lambda_phys=lambda_phys,
                use_reg_loss=use_reg_loss_epoch,
                use_stress_loss=use_stress_loss_epoch,
                use_flow_loss=use_flow_loss_epoch,
                use_force_loss=use_force_loss_epoch,
                use_action_loss=use_action_loss_epoch,
                use_phys_loss=use_phys_loss_epoch,
            )
            if use_cuda_event_timing and ev_fwd_e is not None:
                ev_fwd_e.record()
            forward_time_wall = time.perf_counter() - forward_wall_start

            opt.zero_grad()
            loss_total_detached = loss_total.detach()
            loss_total_is_finite = bool(torch.isfinite(loss_total_detached).item())
            loss_all_finite, loss_finite_count = _distributed_all_true(loss_total_is_finite, device)
            if not loss_all_finite:
                if rank == 0:
                    _tqdm_log(
                        f"[logic_train] WARN non-finite loss_total at epoch={epoch+1} "
                        f"stage={stage_name} bi={bi}; finite_ranks={loss_finite_count}/{world_size}; skip step"
                    )
                opt.zero_grad(set_to_none=True)
                last_end = time.perf_counter()
                continue
            backward_wall_start = time.perf_counter()
            if use_cuda_event_timing and ev_bwd_s is not None:
                ev_bwd_s.record()
            if use_amp:
                scaler.scale(loss_total).backward()
            else:
                loss_total.backward()
            if use_cuda_event_timing and ev_bwd_e is not None:
                ev_bwd_e.record()
            backward_time_wall = time.perf_counter() - backward_wall_start

            optim_wall_start = time.perf_counter()
            if use_cuda_event_timing and ev_opt_s is not None:
                ev_opt_s.record()
            grad_norm: Optional[torch.Tensor] = None
            if use_amp:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                grad_norm_is_finite = bool(torch.isfinite(grad_norm.detach()).item())
                grad_all_finite, grad_finite_count = _distributed_all_true(grad_norm_is_finite, device)
                if not grad_all_finite:
                    if rank == 0:
                        _tqdm_log(
                            f"[logic_train] WARN non-finite grad_norm at epoch={epoch+1} "
                            f"stage={stage_name} bi={bi}; finite_ranks={grad_finite_count}/{world_size}; skip step"
                        )
                    opt.zero_grad(set_to_none=True)
                    scaler.update()
                    last_end = time.perf_counter()
                    continue
                scaler.step(opt)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                grad_norm_is_finite = bool(torch.isfinite(grad_norm.detach()).item())
                grad_all_finite, grad_finite_count = _distributed_all_true(grad_norm_is_finite, device)
                if not grad_all_finite:
                    if rank == 0:
                        _tqdm_log(
                            f"[logic_train] WARN non-finite grad_norm at epoch={epoch+1} "
                            f"stage={stage_name} bi={bi}; finite_ranks={grad_finite_count}/{world_size}; skip step"
                        )
                    opt.zero_grad(set_to_none=True)
                    last_end = time.perf_counter()
                    continue
                opt.step()
            if use_cuda_event_timing and ev_opt_e is not None:
                ev_opt_e.record()
            optim_time_wall = time.perf_counter() - optim_wall_start

            step_wall_time = time.perf_counter() - step_wall_start
            t_iter_end = time.perf_counter()
            total_iter_time = t_iter_end - last_end
            last_end = t_iter_end

            h2d_time = h2d_time_wall
            preprocess_gpu_time = preprocess_gpu_time_wall
            forward_time = forward_time_wall
            backward_time = backward_time_wall
            optim_time = optim_time_wall
            timing_mode = "approx-wall"
            if use_cuda_event_timing:
                # 仅 sampled profiling 步读取 event，并在此处做必要同步。
                torch.cuda.synchronize(device)
                h2d_time = float(ev_h2d_s.elapsed_time(ev_h2d_e)) / 1000.0 if (ev_h2d_s is not None and ev_h2d_e is not None) else h2d_time
                preprocess_gpu_time = float(ev_prep_s.elapsed_time(ev_prep_e)) / 1000.0 if (ev_prep_s is not None and ev_prep_e is not None) else preprocess_gpu_time
                forward_time = float(ev_fwd_s.elapsed_time(ev_fwd_e)) / 1000.0 if (ev_fwd_s is not None and ev_fwd_e is not None) else forward_time
                backward_time = float(ev_bwd_s.elapsed_time(ev_bwd_e)) / 1000.0 if (ev_bwd_s is not None and ev_bwd_e is not None) else backward_time
                optim_time = float(ev_opt_s.elapsed_time(ev_opt_e)) / 1000.0 if (ev_opt_s is not None and ev_opt_e is not None) else optim_time
                timing_mode = "profiled(cuda-event)"

            t_metrics = {
                "data_wait": float(data_wait),
                "h2d_time": float(h2d_time),
                "preprocess_gpu_time": float(preprocess_gpu_time),
                "forward_time": float(forward_time),
                "backward_time": float(backward_time),
                "optim_time": float(optim_time),
                "step_wall_time": float(step_wall_time),
                "total_iter_time": float(total_iter_time),
            }
            timing_tracker.update(t_metrics, is_startup=(bi == 0))

            bsz = int(rgb.shape[0])
            if use_reg_loss_epoch:
                reg_mae_sum, reg_count = _per_target_abs_error_sums(out["param_pred"], gt_train_space, valid_mask)
            else:
                reg_mae_sum = torch.zeros(4, device=device, dtype=torch.float32)
                reg_count = torch.zeros(4, device=device, dtype=torch.float32)
            train_sums[0] += float(loss_total.item()) * bsz
            train_sums[1] += float(loss_reg_part.item()) * bsz
            train_sums[2] += float(loss_stress_part.item()) * bsz
            train_sums[3] += float(loss_flow_part.item()) * bsz
            train_sums[4] += float(loss_force_part.item()) * bsz
            train_sums[5] += float(action_ret["loss_action"].item()) * bsz
            train_sums[6] += float(phys_ret["loss_phys_total"].item()) * bsz
            train_sums[7] += float(phys_ret["loss_stress_flow_consistency"].item()) * bsz
            train_sums[8] += float(phys_ret["loss_force_stress_consistency"].item()) * bsz
            train_sums[9] += float(phys_ret["loss_force_flow_consistency"].item()) * bsz
            train_sums[10] += float(action_ret["action_acc"].item()) * bsz
            train_sums[11] += float(bsz)
            train_sums[_REG_EDGE_STRESS] += float(loss_stress_edge.item()) * bsz
            train_sums[_REG_EDGE_FLOW] += float(loss_flow_edge.item()) * bsz
            train_sums[_REG_STRESS_RECON] += float(loss_stress_recon.item()) * bsz
            train_sums[_REG_FLOW_RECON] += float(loss_flow_recon.item()) * bsz
            train_sums[_REG_FORCE_BCE] += float(loss_force_bce.item()) * bsz
            train_sums[_REG_FORCE_DICE] += float(loss_force_dice.item()) * bsz
            for i in range(4):
                train_sums[_REG_MAE_SUM_START + i] += float(reg_mae_sum[i].item())
                train_sums[_REG_COUNT_START + i] += float(reg_count[i].item())

            if rank == 0:
                _postfix_items = _stage_metric_parts(
                    stage_name=stage_name,
                    total_loss=float(loss_total.item()),
                    reg_loss=float(loss_reg_part.item()),
                    stress_loss=float(loss_stress_part.item()),
                    flow_loss=float(loss_flow_part.item()),
                    force_loss=float(loss_force_part.item()),
                    action_loss=float(action_ret["loss_action"].item()),
                    phys_loss=float(phys_ret["loss_phys_total"].item()),
                    action_acc=float(action_ret["action_acc"].item()),
                    stress_recon=float(loss_stress_recon.item()) if use_stress_loss_epoch else None,
                    stress_edge=float(loss_stress_edge.item())
                    if (use_stress_loss_epoch and float(lambda_stress_edge) > 0.0)
                    else None,
                    flow_recon=float(loss_flow_recon.item()) if use_flow_loss_epoch else None,
                    flow_edge=float(loss_flow_edge.item())
                    if (use_flow_loss_epoch and float(lambda_flow_edge) > 0.0)
                    else None,
                    force_bce=float(loss_force_bce.item())
                    if (use_force_loss_epoch and "force_logits" in out)
                    else None,
                    force_dice=float(loss_force_dice.item())
                    if (use_force_loss_epoch and "force_logits" in out)
                    else None,
                    lambda_stress_edge=float(lambda_stress_edge),
                    lambda_flow_edge=float(lambda_flow_edge),
                    force_bce_weight=float(force_bce_weight),
                    force_dice_weight=float(force_dice_weight),
                )
                train_pbar.set_postfix_str(" ".join(_postfix_items))

        if distributed:
            t_stat = torch.tensor(train_sums, device=device, dtype=torch.float64)
            dist.all_reduce(t_stat, op=dist.ReduceOp.SUM)
            train_sums = [float(t_stat[i].item()) for i in range(_REG_SUMS_LEN)]
        n = float(train_sums[11])
        denom = max(n, 1.0)
        avg_loss = float(train_sums[0] / denom)
        train_reg_maes = [_per_target_mae_from_sums(train_sums, i) for i in range(4)]
        if rank == 0:
            _fb = train_sums[_REG_FORCE_BCE] / denom
            _fd = train_sums[_REG_FORCE_DICE] / denom
            _epoch_parts = _stage_metric_parts(
                stage_name=stage_name,
                total_loss=avg_loss,
                reg_loss=train_sums[1] / denom,
                stress_loss=train_sums[2] / denom,
                flow_loss=train_sums[3] / denom,
                force_loss=train_sums[4] / denom,
                action_loss=train_sums[5] / denom,
                phys_loss=train_sums[6] / denom,
                action_acc=train_sums[10] / denom,
                prefix="avg_",
                stress_recon=(train_sums[_REG_STRESS_RECON] / denom) if use_stress_loss_epoch else None,
                stress_edge=(train_sums[_REG_EDGE_STRESS] / denom)
                if (use_stress_loss_epoch and float(lambda_stress_edge) > 0.0)
                else None,
                flow_recon=(train_sums[_REG_FLOW_RECON] / denom) if use_flow_loss_epoch else None,
                flow_edge=(train_sums[_REG_EDGE_FLOW] / denom)
                if (use_flow_loss_epoch and float(lambda_flow_edge) > 0.0)
                else None,
                force_bce=_fb if (_fb + _fd) > 1e-12 else None,
                force_dice=_fd if (_fb + _fd) > 1e-12 else None,
                lambda_stress_edge=float(lambda_stress_edge),
                lambda_flow_edge=float(lambda_flow_edge),
                force_bce_weight=float(force_bce_weight),
                force_dice_weight=float(force_dice_weight),
            )
            epoch_pbar.set_postfix_str(" ".join(_epoch_parts))
            _tqdm_log(
                f"[logic_train] epoch={epoch+1} stage={stage_name} " + " ".join(_epoch_parts)
            )
            if tb_writer is not None:
                ep = int(epoch + 1)
                tb_writer.add_scalar("train/avg_loss", avg_loss, ep)
                tb_writer.add_scalar("train/avg_loss_reg", train_sums[1] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_stress", train_sums[2] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_flow", train_sums[3] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_force", train_sums[4] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_action", train_sums[5] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_phys_total", train_sums[6] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_phys_sf", train_sums[7] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_phys_fs", train_sums[8] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_phys_ff", train_sums[9] / denom, ep)
                tb_writer.add_scalar("train/avg_action_acc", train_sums[10] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_stress_edge", train_sums[_REG_EDGE_STRESS] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_flow_edge", train_sums[_REG_EDGE_FLOW] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_stress_recon", train_sums[_REG_STRESS_RECON] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_flow_recon", train_sums[_REG_FLOW_RECON] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_force_bce", train_sums[_REG_FORCE_BCE] / denom, ep)
                tb_writer.add_scalar("train/avg_loss_force_dice", train_sums[_REG_FORCE_DICE] / denom, ep)
                tb_writer.add_scalar("train/avg_reg_mae_logE", train_reg_maes[0], ep)
                tb_writer.add_scalar("train/avg_reg_mae_nu", train_reg_maes[1], ep)
                tb_writer.add_scalar("train/avg_reg_mae_logDensity", train_reg_maes[2], ep)
                tb_writer.add_scalar("train/avg_reg_mae_logYield", train_reg_maes[3], ep)
                tb_writer.add_scalar(
                    "train/weighted_stress", lambda_stress * train_sums[2] / denom, ep
                )
                tb_writer.add_scalar("train/weighted_flow", lambda_flow * train_sums[3] / denom, ep)
                tb_writer.add_scalar("train/weighted_force", lambda_force * train_sums[4] / denom, ep)
                tb_writer.add_scalar("train/weighted_action", effective_lambda_action * train_sums[5] / denom, ep)
                tb_writer.add_scalar("train/weighted_phys", effective_lambda_phys * train_sums[6] / denom, ep)
                if timing_tb:
                    tb_writer.add_scalar("timing/startup_data_wait", float(timing_tracker.startup_data_wait or 0.0), ep)
                    tb_writer.add_scalar("timing/avg_data_wait", timing_tracker.avg("data_wait"), ep)
                    tb_writer.add_scalar("timing/avg_h2d", timing_tracker.avg("h2d_time"), ep)
                    tb_writer.add_scalar("timing/avg_preprocess_gpu", timing_tracker.avg("preprocess_gpu_time"), ep)
                    tb_writer.add_scalar("timing/avg_forward", timing_tracker.avg("forward_time"), ep)
                    tb_writer.add_scalar("timing/avg_backward", timing_tracker.avg("backward_time"), ep)
                    tb_writer.add_scalar("timing/avg_optim", timing_tracker.avg("optim_time"), ep)
                    tb_writer.add_scalar("timing/avg_step_wall", timing_tracker.avg("step_wall_time"), ep)
                    tb_writer.add_scalar("timing/avg_total_iter", timing_tracker.avg("total_iter_time"), ep)
                    tb_writer.add_scalar("timing/recent_step_wall", timing_tracker.recent_avg("step_wall_time"), ep)
                    tb_writer.add_scalar(
                        "timing/ema_step_wall",
                        float(timing_tracker.ema.get("step_wall_time", 0.0)),
                        ep,
                    )

        # checkpoint（全部在 logic_model/output/checkpoints）
        if ((epoch + 1) % save_every_epochs) == 0 and rank == 0:
            ckpt = _build_checkpoint_payload(
                epoch_1based=int(epoch + 1),
                avg_loss=avg_loss,
                model=model,
                optimizer=opt,
                scaler=scaler,
                use_amp=use_amp,
                action_to_id=ds.action_to_id,
                model_arch=model_arch,
                max_views=max_views,
                dec_h=dec_h,
                dec_w=dec_w,
                cfg_model=cfg_model,
                stage_name=stage_name,
                checkpoint_kind="periodic",
            )
            torch.save(ckpt, str(ckpt_dir / f"epoch_{epoch + 1:04d}.pt"))
            # 明确不维护 best/last checkpoint，避免误用历史 best.pt / last.pt。
            best_ckpt_path = ckpt_dir / "best.pt"
            if best_ckpt_path.exists():
                best_ckpt_path.unlink()
            last_ckpt_path = ckpt_dir / "last.pt"
            if last_ckpt_path.exists():
                last_ckpt_path.unlink()

        if distributed:
            _safe_barrier(local_rank if distributed else None)

        eval_loss_switches = _resolve_eval_loss_switches(
            stage_name=stage_name,
            use_reg_loss=use_reg_loss,
            use_stress_loss=use_stress_loss,
            use_flow_loss=use_flow_loss,
            use_force_loss=use_force_loss,
            use_action_loss=use_action_loss,
            use_phys_loss=use_phys_loss,
        )
        eval_eff_lambda_stress = float(lambda_stress) if eval_loss_switches["use_stress_loss"] else 0.0
        eval_eff_lambda_flow = float(lambda_flow) if eval_loss_switches["use_flow_loss"] else 0.0
        eval_eff_lambda_force = float(lambda_force) if eval_loss_switches["use_force_loss"] else 0.0
        eval_eff_lambda_action = float(lambda_action) if eval_loss_switches["use_action_loss"] else 0.0
        eval_eff_lambda_phys = float(lambda_phys) if eval_loss_switches["use_phys_loss"] else 0.0

        # eval：按 stage 使用不同指标与可视化方案，写入 eval/*.json 与附加产物
        if ((epoch + 1) % eval_every_epochs) == 0:
            if eval_sampler_eval is not None:
                eval_sampler_eval.set_epoch(epoch)

            if eval_sharded:
                model.eval()
                ev_m = _unwrap_model(model)
                with torch.no_grad():
                    sums_t = torch.zeros(_REG_SUMS_LEN, device=device, dtype=torch.float64)
                    local_param_eval_records: List[Dict[str, Any]] = []
                    batch_steps = 0
                    _eval_cap_sh = min(len(eval_loader), eval_max_batches)
                    eval_pbar = tqdm(
                        eval_loader,
                        total=_eval_cap_sh,
                        desc=f"eval(sharded) e{epoch + 1}",
                        disable=rank != 0,
                        leave=False,
                        dynamic_ncols=True,
                    )
                    for batch in eval_pbar:
                        loss_total, parts = _compute_eval_batch_losses(
                            model,
                            ev_m,
                            batch,
                            device,
                            loss_reg,
                            reg_target_mean,
                            reg_target_std,
                            phys_cfg,
                            lambda_stress=lambda_stress,
                            lambda_flow=lambda_flow,
                            lambda_force=lambda_force,
                            lambda_action=lambda_action,
                            lambda_phys=lambda_phys,
                            use_reg_loss=eval_loss_switches["use_reg_loss"],
                            use_stress_loss=eval_loss_switches["use_stress_loss"],
                            use_flow_loss=eval_loss_switches["use_flow_loss"],
                            use_force_loss=eval_loss_switches["use_force_loss"],
                            use_action_loss=eval_loss_switches["use_action_loss"],
                            use_phys_loss=eval_loss_switches["use_phys_loss"],
                            stage_name=stage_name,
                            object_mask_fg_weight=object_mask_fg_weight,
                            object_mask_bg_weight=object_mask_bg_weight,
                            object_mask_bg_black=object_mask_bg_black,
                            lambda_stress_edge=lambda_stress_edge,
                            lambda_flow_edge=lambda_flow_edge,
                            edge_boost=edge_boost,
                            force_bce_weight=force_bce_weight,
                            force_dice_weight=force_dice_weight,
                            use_amp=use_amp,
                            non_blocking=nb,
                        )
                        bsz = int(batch["rgb"].shape[0])
                        sums_t[0] += float(loss_total.item()) * bsz
                        sums_t[1] += float(parts["loss_reg"].item()) * bsz
                        sums_t[2] += float(parts["loss_stress"].item()) * bsz
                        sums_t[3] += float(parts["loss_flow"].item()) * bsz
                        sums_t[4] += float(parts["loss_force"].item()) * bsz
                        sums_t[5] += float(parts["loss_action"].item()) * bsz
                        sums_t[6] += float(parts["loss_phys_total"].item()) * bsz
                        sums_t[7] += float(parts["loss_phys_sf"].item()) * bsz
                        sums_t[8] += float(parts["loss_phys_fs"].item()) * bsz
                        sums_t[9] += float(parts["loss_phys_ff"].item()) * bsz
                        sums_t[10] += float(parts["action_acc"].item()) * bsz
                        sums_t[11] += float(bsz)
                        sums_t[_REG_EDGE_STRESS] += float(parts["loss_stress_edge"].item()) * bsz
                        sums_t[_REG_EDGE_FLOW] += float(parts["loss_flow_edge"].item()) * bsz
                        sums_t[_REG_STRESS_RECON] += float(parts["loss_stress_recon"].item()) * bsz
                        sums_t[_REG_FLOW_RECON] += float(parts["loss_flow_recon"].item()) * bsz
                        sums_t[_REG_FORCE_BCE] += float(parts["loss_force_bce"].item()) * bsz
                        sums_t[_REG_FORCE_DICE] += float(parts["loss_force_dice"].item()) * bsz
                        for i in range(4):
                            sums_t[_REG_MAE_SUM_START + i] += float(parts["reg_mae_sum"][i].item())
                            sums_t[_REG_COUNT_START + i] += float(parts["reg_count"][i].item())
                        if stage_name == "param" and eval_param_scatter:
                            pred_raw_cpu = parts["param_pred_raw"].detach().cpu()
                            gt_raw_cpu = batch["params"].detach().cpu()
                            sample_ids = [str(s) for s in batch.get("sample_id", [])]
                            params_meta = batch.get("params_dict", [])
                            action_names = batch.get("action_name", [])
                            for i in range(bsz):
                                meta = params_meta[i] if i < len(params_meta) and isinstance(params_meta[i], dict) else {}
                                action_name_i = (
                                    str(action_names[i]).strip()
                                    if i < len(action_names)
                                    else str(meta.get("action", "")).strip()
                                )
                                local_param_eval_records.append(
                                    {
                                        "sample_id": sample_ids[i] if i < len(sample_ids) else str(i),
                                        "object": str(meta.get("object", "")).strip() or "unknown",
                                        "material": str(meta.get("material", "")).strip() or "unknown",
                                        "action": action_name_i or "unknown",
                                        "param_gt_raw": gt_raw_cpu[i].tolist(),
                                        "param_pred_raw": pred_raw_cpu[i].tolist(),
                                    }
                                )
                        batch_steps += 1
                        if rank == 0:
                            _sn = float(sums_t[11].item())
                            _eval_postfix = _stage_metric_parts(
                                stage_name=stage_name,
                                total_loss=float(sums_t[0].item() / max(_sn, 1.0)),
                                reg_loss=float(sums_t[1].item() / max(_sn, 1.0)),
                                stress_loss=float(sums_t[2].item() / max(_sn, 1.0)),
                                flow_loss=float(sums_t[3].item() / max(_sn, 1.0)),
                                force_loss=float(sums_t[4].item() / max(_sn, 1.0)),
                                action_loss=float(sums_t[5].item() / max(_sn, 1.0)),
                                phys_loss=float(sums_t[6].item() / max(_sn, 1.0)),
                                action_acc=float(sums_t[10].item() / max(_sn, 1.0)),
                                include_field_metrics_in_param_stage=True,
                                **_eval_partial_sums_breakdown_kwargs(
                                    sums_t,
                                    eval_loss_switches=eval_loss_switches,
                                    lambda_stress_edge=lambda_stress_edge,
                                    lambda_flow_edge=lambda_flow_edge,
                                    force_bce_weight=force_bce_weight,
                                    force_dice_weight=force_dice_weight,
                                ),
                            )
                            eval_pbar.set_postfix_str(" ".join(_eval_postfix))
                        if batch_steps >= eval_max_batches:
                            break

                dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
                param_eval_records_rank0: List[Dict[str, Any]] = []
                if stage_name == "param" and eval_param_scatter:
                    gathered_param_eval_records: List[Optional[List[Dict[str, Any]]]] | None = (
                        [None] * world_size if rank == 0 else None
                    )
                    dist.gather_object(
                        local_param_eval_records,
                        object_gather_list=gathered_param_eval_records,
                        dst=0,
                    )
                    if rank == 0 and gathered_param_eval_records is not None:
                        for sub in gathered_param_eval_records:
                            if sub:
                                param_eval_records_rank0.extend(sub)
                sums_list = [float(sums_t[i].item()) for i in range(_REG_SUMS_LEN)]
                if rank == 0:
                    ep = int(epoch + 1)
                    metrics = _eval_sums_to_record(
                        sums_list,
                        lambda_stress=eval_eff_lambda_stress,
                        lambda_flow=eval_eff_lambda_flow,
                        lambda_force=eval_eff_lambda_force,
                        lambda_action=eval_eff_lambda_action,
                        lambda_phys=eval_eff_lambda_phys,
                    )
                    eval_obj: Dict[str, Any] = {
                        "epoch": ep,
                        "stage_name": str(stage_name),
                        "eval_mode": "distributed_sharded",
                        "eval_max_batches": int(eval_max_batches),
                        "eval_batches_run": int(min(batch_steps, eval_max_batches)),
                    }
                    eval_obj.update(metrics)
                    (eval_dir / f"epoch_{epoch + 1:04d}.json").write_text(
                        json.dumps(eval_obj, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    (eval_dir / "last.json").write_text(
                        json.dumps(eval_obj, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    if tb_writer is not None:
                        _tensorboard_log_eval_metrics(tb_writer, metrics, ep)
                    _tqdm_log(
                        f"[logic_train] eval epoch={ep} stage={stage_name} "
                        + " ".join(
                            _stage_metric_parts(
                                stage_name=stage_name,
                                total_loss=metrics["avg_loss"],
                                reg_loss=metrics["avg_loss_reg"],
                                stress_loss=metrics["avg_loss_stress"],
                                flow_loss=metrics["avg_loss_flow"],
                                force_loss=metrics["avg_loss_force"],
                                action_loss=metrics["avg_loss_action"],
                                phys_loss=metrics["avg_loss_phys_total"],
                                action_acc=metrics["avg_action_acc"],
                                prefix="avg_",
                                include_field_metrics_in_param_stage=True,
                                **_eval_loss_breakdown_kwargs(
                                    metrics,
                                    eval_loss_switches=eval_loss_switches,
                                    lambda_stress_edge=lambda_stress_edge,
                                    lambda_flow_edge=lambda_flow_edge,
                                    force_bce_weight=force_bce_weight,
                                    force_dice_weight=force_dice_weight,
                                ),
                            )
                        )
                    )
                    try:
                        visual_info = _save_eval_visual_artifacts_rank0(
                            model=model,
                            eval_ds=eval_ds,
                            train_vis_ds=train_source if len(train_source) > 0 else None,
                            device=device,
                            mcore=ev_m,
                            eval_dir=eval_dir,
                            epoch_1based=ep,
                            stage_name=stage_name,
                            eval_video_num_samples=eval_video_num_samples,
                            eval_train_video_num_samples=eval_train_video_num_samples,
                            param_eval_records=param_eval_records_rank0,
                            eval_param_scatter=eval_param_scatter,
                            eval_param_scatter_log=eval_param_scatter_log,
                            eval_scatter_pad_frac=eval_scatter_pad_frac,
                            eval_scatter_axis_percentiles=eval_scatter_axis_percentiles,
                        )
                        if visual_info:
                            _tqdm_log(
                                f"[logic_train] eval visuals epoch={ep} stage={stage_name} "
                                f"saved_to={visual_info.get('visual_dir', '')}"
                            )
                    except Exception as e:
                        _tqdm_log(
                            f"[logic_train] WARN eval visuals failed at epoch={ep} stage={stage_name}: {e}"
                        )
                    best_stage_key: Optional[str] = None
                    best_stage_ckpt_name: Optional[str] = None
                    if stage_name == "field":
                        best_stage_key = "field"
                        best_stage_ckpt_name = "best_stage1.pt"
                    elif stage_name == "flow_force":
                        best_stage_key = "flow_force"
                        best_stage_ckpt_name = "best_stage_flow_force.pt"
                    elif stage_name == "stress":
                        best_stage_key = "stress"
                        best_stage_ckpt_name = "best_stage_stress.pt"
                    elif stage_name == "joint":
                        best_stage_key = "joint"
                        best_stage_ckpt_name = "best_stage_joint.pt"
                    elif stage_name == "param":
                        best_stage_key = "param"
                        best_stage_ckpt_name = "best_stage2.pt"
                    if (
                        best_stage_key is not None
                        and best_stage_ckpt_name is not None
                        and math.isfinite(float(metrics["avg_loss"]))
                        and float(metrics["avg_loss"]) < float(best_eval_loss_by_stage[best_stage_key])
                    ):
                        best_eval_loss_by_stage[best_stage_key] = float(metrics["avg_loss"])
                        best_eval_epoch_by_stage[best_stage_key] = int(ep)
                        best_ckpt = _build_checkpoint_payload(
                            epoch_1based=ep,
                            avg_loss=float(metrics["avg_loss"]),
                            model=model,
                            optimizer=opt,
                            scaler=scaler,
                            use_amp=use_amp,
                            action_to_id=ds.action_to_id,
                            model_arch=model_arch,
                            max_views=max_views,
                            dec_h=dec_h,
                            dec_w=dec_w,
                            cfg_model=cfg_model,
                            eval_metrics=eval_obj,
                            stage_name=stage_name,
                            checkpoint_kind=f"best_eval_{best_stage_key}",
                        )
                        torch.save(best_ckpt, str(ckpt_dir / best_stage_ckpt_name))
                        _tqdm_log(
                            f"[logic_train] saved {best_stage_ckpt_name} at epoch={ep} "
                            f"stage={stage_name} eval_avg_loss={float(metrics['avg_loss']):.6f}"
                        )
                model.train()

            elif rank == 0:
                model.eval()
                ev_m = _unwrap_model(model)
                with torch.no_grad():
                    sums_list = [0.0] * _REG_SUMS_LEN
                    param_eval_records_rank0: List[Dict[str, Any]] = []
                    batch_steps = 0
                    _eval_cap = min(len(eval_loader), eval_max_batches)
                    eval_pbar = tqdm(
                        eval_loader,
                        total=_eval_cap,
                        desc=f"eval e{epoch + 1}",
                        leave=False,
                        dynamic_ncols=True,
                    )
                    for batch in eval_pbar:
                        loss_total, parts = _compute_eval_batch_losses(
                            model,
                            ev_m,
                            batch,
                            device,
                            loss_reg,
                            reg_target_mean,
                            reg_target_std,
                            phys_cfg,
                            lambda_stress=lambda_stress,
                            lambda_flow=lambda_flow,
                            lambda_force=lambda_force,
                            lambda_action=lambda_action,
                            lambda_phys=lambda_phys,
                            use_reg_loss=eval_loss_switches["use_reg_loss"],
                            use_stress_loss=eval_loss_switches["use_stress_loss"],
                            use_flow_loss=eval_loss_switches["use_flow_loss"],
                            use_force_loss=eval_loss_switches["use_force_loss"],
                            use_action_loss=eval_loss_switches["use_action_loss"],
                            use_phys_loss=eval_loss_switches["use_phys_loss"],
                            stage_name=stage_name,
                            object_mask_fg_weight=object_mask_fg_weight,
                            object_mask_bg_weight=object_mask_bg_weight,
                            object_mask_bg_black=object_mask_bg_black,
                            lambda_stress_edge=lambda_stress_edge,
                            lambda_flow_edge=lambda_flow_edge,
                            edge_boost=edge_boost,
                            force_bce_weight=force_bce_weight,
                            force_dice_weight=force_dice_weight,
                            use_amp=use_amp,
                            non_blocking=nb,
                        )
                        bsz = int(batch["rgb"].shape[0])
                        sums_list[0] += float(loss_total.item()) * bsz
                        sums_list[1] += float(parts["loss_reg"].item()) * bsz
                        sums_list[2] += float(parts["loss_stress"].item()) * bsz
                        sums_list[3] += float(parts["loss_flow"].item()) * bsz
                        sums_list[4] += float(parts["loss_force"].item()) * bsz
                        sums_list[5] += float(parts["loss_action"].item()) * bsz
                        sums_list[6] += float(parts["loss_phys_total"].item()) * bsz
                        sums_list[7] += float(parts["loss_phys_sf"].item()) * bsz
                        sums_list[8] += float(parts["loss_phys_fs"].item()) * bsz
                        sums_list[9] += float(parts["loss_phys_ff"].item()) * bsz
                        sums_list[10] += float(parts["action_acc"].item()) * bsz
                        sums_list[11] += float(bsz)
                        sums_list[_REG_EDGE_STRESS] += float(parts["loss_stress_edge"].item()) * bsz
                        sums_list[_REG_EDGE_FLOW] += float(parts["loss_flow_edge"].item()) * bsz
                        sums_list[_REG_STRESS_RECON] += float(parts["loss_stress_recon"].item()) * bsz
                        sums_list[_REG_FLOW_RECON] += float(parts["loss_flow_recon"].item()) * bsz
                        sums_list[_REG_FORCE_BCE] += float(parts["loss_force_bce"].item()) * bsz
                        sums_list[_REG_FORCE_DICE] += float(parts["loss_force_dice"].item()) * bsz
                        for i in range(4):
                            sums_list[_REG_MAE_SUM_START + i] += float(parts["reg_mae_sum"][i].item())
                            sums_list[_REG_COUNT_START + i] += float(parts["reg_count"][i].item())
                        if stage_name == "param" and eval_param_scatter:
                            pred_raw_cpu = parts["param_pred_raw"].detach().cpu()
                            gt_raw_cpu = batch["params"].detach().cpu()
                            sample_ids = [str(s) for s in batch.get("sample_id", [])]
                            params_meta = batch.get("params_dict", [])
                            action_names = batch.get("action_name", [])
                            for i in range(bsz):
                                meta = params_meta[i] if i < len(params_meta) and isinstance(params_meta[i], dict) else {}
                                action_name_i = (
                                    str(action_names[i]).strip()
                                    if i < len(action_names)
                                    else str(meta.get("action", "")).strip()
                                )
                                param_eval_records_rank0.append(
                                    {
                                        "sample_id": sample_ids[i] if i < len(sample_ids) else str(i),
                                        "object": str(meta.get("object", "")).strip() or "unknown",
                                        "material": str(meta.get("material", "")).strip() or "unknown",
                                        "action": action_name_i or "unknown",
                                        "param_gt_raw": gt_raw_cpu[i].tolist(),
                                        "param_pred_raw": pred_raw_cpu[i].tolist(),
                                    }
                                )
                        batch_steps += 1
                        _sn = sums_list[11]
                        _eval_postfix = _stage_metric_parts(
                            stage_name=stage_name,
                            total_loss=float(sums_list[0] / max(_sn, 1.0)),
                            reg_loss=float(sums_list[1] / max(_sn, 1.0)),
                            stress_loss=float(sums_list[2] / max(_sn, 1.0)),
                            flow_loss=float(sums_list[3] / max(_sn, 1.0)),
                            force_loss=float(sums_list[4] / max(_sn, 1.0)),
                            action_loss=float(sums_list[5] / max(_sn, 1.0)),
                            phys_loss=float(sums_list[6] / max(_sn, 1.0)),
                            action_acc=float(sums_list[10] / max(_sn, 1.0)),
                            include_field_metrics_in_param_stage=True,
                            **_eval_partial_list_breakdown_kwargs(
                                sums_list,
                                eval_loss_switches=eval_loss_switches,
                                lambda_stress_edge=lambda_stress_edge,
                                lambda_flow_edge=lambda_flow_edge,
                                force_bce_weight=force_bce_weight,
                                force_dice_weight=force_dice_weight,
                            ),
                        )
                        eval_pbar.set_postfix_str(" ".join(_eval_postfix))
                        if batch_steps >= eval_max_batches:
                            break

                ep = int(epoch + 1)
                metrics = _eval_sums_to_record(
                    sums_list,
                    lambda_stress=eval_eff_lambda_stress,
                    lambda_flow=eval_eff_lambda_flow,
                    lambda_force=eval_eff_lambda_force,
                    lambda_action=eval_eff_lambda_action,
                    lambda_phys=eval_eff_lambda_phys,
                )
                eval_obj = {
                    "epoch": ep,
                    "stage_name": str(stage_name),
                    "eval_mode": "rank0_only",
                    "eval_max_batches": int(eval_max_batches),
                    "eval_batches_run": int(min(batch_steps, eval_max_batches)),
                }
                eval_obj.update(metrics)
                (eval_dir / f"epoch_{epoch + 1:04d}.json").write_text(
                    json.dumps(eval_obj, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (eval_dir / "last.json").write_text(
                    json.dumps(eval_obj, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                if tb_writer is not None:
                    _tensorboard_log_eval_metrics(tb_writer, metrics, ep)
                _tqdm_log(
                    f"[logic_train] eval epoch={ep} stage={stage_name} "
                    + " ".join(
                        _stage_metric_parts(
                            stage_name=stage_name,
                            total_loss=metrics["avg_loss"],
                            reg_loss=metrics["avg_loss_reg"],
                            stress_loss=metrics["avg_loss_stress"],
                            flow_loss=metrics["avg_loss_flow"],
                            force_loss=metrics["avg_loss_force"],
                            action_loss=metrics["avg_loss_action"],
                            phys_loss=metrics["avg_loss_phys_total"],
                            action_acc=metrics["avg_action_acc"],
                            prefix="avg_",
                            include_field_metrics_in_param_stage=True,
                            **_eval_loss_breakdown_kwargs(
                                metrics,
                                eval_loss_switches=eval_loss_switches,
                                lambda_stress_edge=lambda_stress_edge,
                                lambda_flow_edge=lambda_flow_edge,
                                force_bce_weight=force_bce_weight,
                                force_dice_weight=force_dice_weight,
                            ),
                        )
                    )
                )
                try:
                    visual_info = _save_eval_visual_artifacts_rank0(
                        model=model,
                        eval_ds=eval_ds,
                        train_vis_ds=train_source if len(train_source) > 0 else None,
                        device=device,
                        mcore=ev_m,
                        eval_dir=eval_dir,
                        epoch_1based=ep,
                        stage_name=stage_name,
                        eval_video_num_samples=eval_video_num_samples,
                        eval_train_video_num_samples=eval_train_video_num_samples,
                        param_eval_records=param_eval_records_rank0,
                        eval_param_scatter=eval_param_scatter,
                        eval_param_scatter_log=eval_param_scatter_log,
                        eval_scatter_pad_frac=eval_scatter_pad_frac,
                        eval_scatter_axis_percentiles=eval_scatter_axis_percentiles,
                    )
                    if visual_info:
                        _tqdm_log(
                            f"[logic_train] eval visuals epoch={ep} stage={stage_name} "
                            f"saved_to={visual_info.get('visual_dir', '')}"
                        )
                except Exception as e:
                    _tqdm_log(
                        f"[logic_train] WARN eval visuals failed at epoch={ep} stage={stage_name}: {e}"
                    )
                best_stage_key: Optional[str] = None
                best_stage_ckpt_name: Optional[str] = None
                if stage_name == "field":
                    best_stage_key = "field"
                    best_stage_ckpt_name = "best_stage1.pt"
                elif stage_name == "flow_force":
                    best_stage_key = "flow_force"
                    best_stage_ckpt_name = "best_stage_flow_force.pt"
                elif stage_name == "stress":
                    best_stage_key = "stress"
                    best_stage_ckpt_name = "best_stage_stress.pt"
                elif stage_name == "joint":
                    best_stage_key = "joint"
                    best_stage_ckpt_name = "best_stage_joint.pt"
                elif stage_name == "param":
                    best_stage_key = "param"
                    best_stage_ckpt_name = "best_stage2.pt"
                if (
                    best_stage_key is not None
                    and best_stage_ckpt_name is not None
                    and math.isfinite(float(metrics["avg_loss"]))
                    and float(metrics["avg_loss"]) < float(best_eval_loss_by_stage[best_stage_key])
                ):
                    best_eval_loss_by_stage[best_stage_key] = float(metrics["avg_loss"])
                    best_eval_epoch_by_stage[best_stage_key] = int(ep)
                    best_ckpt = _build_checkpoint_payload(
                        epoch_1based=ep,
                        avg_loss=float(metrics["avg_loss"]),
                        model=model,
                        optimizer=opt,
                        scaler=scaler,
                        use_amp=use_amp,
                        action_to_id=ds.action_to_id,
                        model_arch=model_arch,
                        max_views=max_views,
                        dec_h=dec_h,
                        dec_w=dec_w,
                        cfg_model=cfg_model,
                        eval_metrics=eval_obj,
                        stage_name=stage_name,
                        checkpoint_kind=f"best_eval_{best_stage_key}",
                    )
                    torch.save(best_ckpt, str(ckpt_dir / best_stage_ckpt_name))
                    _tqdm_log(
                        f"[logic_train] saved {best_stage_ckpt_name} at epoch={ep} "
                        f"stage={stage_name} eval_avg_loss={float(metrics['avg_loss']):.6f}"
                    )
            model.train()

        if distributed:
            _safe_barrier(local_rank if distributed else None)

    if distributed:
        _safe_barrier(local_rank if distributed else None)
        dist.destroy_process_group()
    if rank == 0:
        if tb_writer is not None:
            tb_writer.close()
        _tqdm_log("[logic_train] training finished.")


if __name__ == "__main__":
    main()

