# -*- coding: utf-8 -*-
"""Arch4 主任务回归损失（SmoothL1 / 异方差）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def arch4_field_supervision_mse(
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    dec_h: int,
    dec_w: int,
) -> torch.Tensor:
    """
    时序场监督：
    - 预测 ``[B,V,C,T,dec_h,dec_w]``
    - 真值 ``[B,V,C,T,H,W]``
    仅做空间下采样到 ``(dec_h,dec_w)``，不再对时间维做均值压缩。
    """
    if pred_bvcthw.dim() != 6 or gt_bvcthw.dim() != 6:
        raise ValueError("pred/gt 均须为 [B,V,C,T,H,W]-style 6维")
    b, v, c, t, h_i, w_i = gt_bvcthw.shape
    if int(pred_bvcthw.shape[0]) != b or int(pred_bvcthw.shape[1]) != v or int(pred_bvcthw.shape[2]) != c or int(pred_bvcthw.shape[3]) != t:
        raise ValueError(f"pred/gt 维度不匹配: pred={tuple(pred_bvcthw.shape)} gt={tuple(gt_bvcthw.shape)}")
    x = gt_bvcthw.reshape(b * v * t, c, h_i, w_i)
    y = F.interpolate(x, size=(dec_h, dec_w), mode="bilinear", align_corners=False)
    tgt = y.view(b, v, t, c, dec_h, dec_w).permute(0, 1, 3, 2, 4, 5).contiguous()
    return F.mse_loss(pred_bvcthw, tgt)


@dataclass
class Arch4LossConfig:
    heteroscedastic: bool = False
    smoothl1_beta: float = 1.0
    target_weights: Optional[torch.Tensor] = None
    logvar_min: float = -10.0
    logvar_max: float = 10.0


def heteroscedastic_loss(
    pred: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor],
    logvar_min: float,
    logvar_max: float,
) -> torch.Tensor:
    lv = logvar.clamp(logvar_min, logvar_max)
    inv = torch.exp(-lv)
    diff2 = (pred - target) ** 2
    loss = 0.5 * (inv * diff2 + lv)
    if weights is not None:
        loss = loss * weights.view(1, -1).to(loss)
    if valid_mask is not None:
        m = valid_mask.to(loss)
        loss = loss * m
        denom = m.sum().clamp(min=1.0)
        return loss.sum() / denom
    return loss.mean()


class Arch4RegressionLoss(nn.Module):
    def __init__(self, cfg: Optional[Arch4LossConfig] = None):
        super().__init__()
        self.cfg = cfg or Arch4LossConfig()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        logvar: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w = self.cfg.target_weights
        if self.cfg.heteroscedastic:
            if logvar is None:
                raise ValueError("heteroscedastic 需要 logvar")
            return heteroscedastic_loss(
                pred,
                logvar,
                target,
                w,
                valid_mask,
                self.cfg.logvar_min,
                self.cfg.logvar_max,
            )
        loss = F.smooth_l1_loss(pred, target, beta=self.cfg.smoothl1_beta, reduction="none")
        if w is not None:
            loss = loss * w.view(1, -1).to(loss)
        if valid_mask is not None:
            m = valid_mask.to(loss)
            loss = loss * m
            denom = m.sum().clamp(min=1.0)
            return loss.sum() / denom
        return loss.mean()
