from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from my_model.losses import Arch4LossConfig, Arch4RegressionLoss, arch4_field_supervision_mse


def _as_bvthw(x: torch.Tensor) -> torch.Tensor:
    """
    支持输入:
    - [B,V,1,T,H,W]
    - [B,V,C,T,H,W]
    统一返回 [B,V,T,H,W]（通道取绝对值均值）。
    """
    if x.dim() != 6:
        raise ValueError(f"expect 6D tensor [B,V,C,T,H,W], got {tuple(x.shape)}")
    return x.abs().mean(dim=2)


def _safe_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = (1, 2, 3, 4)
    mu = x.mean(dim=dims, keepdim=True)
    var = (x - mu).pow(2).mean(dim=dims, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps)


def _align_bvthw(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    将 src [B,V,T,H,W] 对齐到 ref 的 [T,H,W]（时间线性、空间双线性）。
    """
    if src.shape == ref.shape:
        return src
    b, v, ts, hs, ws = src.shape
    tr, hr, wr = int(ref.shape[2]), int(ref.shape[3]), int(ref.shape[4])
    x = src.view(b * v, 1, ts, hs, ws)
    # 3D 插值: (D,T)=(T,H,W)
    x = F.interpolate(x, size=(tr, hr, wr), mode="trilinear", align_corners=False)
    return x.view(b, v, tr, hr, wr)


def _soft_weight_from_force(force: torch.Tensor, sharpness: float = 8.0) -> torch.Tensor:
    # force: [B,V,T,H,W], normalize to [0,1] then sigmoid sharpen
    fmin = force.amin(dim=(1, 2, 3, 4), keepdim=True)
    fmax = force.amax(dim=(1, 2, 3, 4), keepdim=True)
    fn = (force - fmin) / (fmax - fmin + 1e-6)
    return torch.sigmoid((fn - 0.5) * float(sharpness))


def _masked_contrast_loss(value: torch.Tensor, weight: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """
    软掩码区域对比：
    区域内均值应高于区域外均值至少 margin。
    """
    w_in = weight.clamp(0.0, 1.0)
    w_out = (1.0 - w_in).clamp(0.0, 1.0)
    in_mean = (w_in * value).sum(dim=(1, 2, 3, 4)) / (w_in.sum(dim=(1, 2, 3, 4)) + 1e-6)
    out_mean = (w_out * value).sum(dim=(1, 2, 3, 4)) / (w_out.sum(dim=(1, 2, 3, 4)) + 1e-6)
    # 希望 in_mean - out_mean >= margin
    return F.softplus(float(margin) - (in_mean - out_mean)).mean()


def loss_stress_flow_consistency(
    stress_bvcthw: torch.Tensor,
    flow_bvcthw: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    高 stress 区域倾向于高 flow：
    使用归一化后正相关约束（softplus-margin on correlation）。
    """
    s = _safe_zscore(_as_bvthw(stress_bvcthw))
    f = _safe_zscore(_as_bvthw(flow_bvcthw))
    f = _align_bvthw(f, s)
    corr = (s * f).mean(dim=(1, 2, 3, 4))
    return F.softplus(float(margin) - corr).mean()


def loss_force_stress_consistency(
    force_bvcthw: torch.Tensor,
    stress_bvcthw: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    力/接触邻域 stress 更高：mask 内外软对比。
    """
    force = _as_bvthw(force_bvcthw)
    stress = _as_bvthw(stress_bvcthw)
    force = _align_bvthw(force, stress)
    w = _soft_weight_from_force(force)
    return _masked_contrast_loss(stress, w, margin=float(margin))


def loss_force_flow_consistency(
    force_bvcthw: torch.Tensor,
    flow_bvcthw: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    力/接触邻域 flow 更高：mask 内外软对比。
    """
    force = _as_bvthw(force_bvcthw)
    flow = _as_bvthw(flow_bvcthw)
    force = _align_bvthw(force, flow)
    w = _soft_weight_from_force(force)
    return _masked_contrast_loss(flow, w, margin=float(margin))


@dataclass
class PhysicsConsistencyConfig:
    lambda_stress_flow_consistency: float = 1.0
    lambda_force_stress_consistency: float = 1.0
    lambda_force_flow_consistency: float = 1.0
    margin_stress_flow: float = 0.1
    margin_force_stress: float = 0.05
    margin_force_flow: float = 0.05
    use_pred_force_mask: bool = False


def compute_physics_consistency_losses(
    *,
    stress_pred: torch.Tensor,
    flow_pred: torch.Tensor,
    force_pred: torch.Tensor,
    force_gt: Optional[torch.Tensor],
    cfg: PhysicsConsistencyConfig,
) -> Dict[str, torch.Tensor]:
    force_ref = force_pred if cfg.use_pred_force_mask or (force_gt is None) else force_gt

    l_sf = loss_stress_flow_consistency(stress_pred, flow_pred, margin=cfg.margin_stress_flow)
    l_fs = loss_force_stress_consistency(force_ref, stress_pred, margin=cfg.margin_force_stress)
    l_ff = loss_force_flow_consistency(force_ref, flow_pred, margin=cfg.margin_force_flow)
    l_total = (
        float(cfg.lambda_stress_flow_consistency) * l_sf
        + float(cfg.lambda_force_stress_consistency) * l_fs
        + float(cfg.lambda_force_flow_consistency) * l_ff
    )
    return {
        "loss_stress_flow_consistency": l_sf,
        "loss_force_stress_consistency": l_fs,
        "loss_force_flow_consistency": l_ff,
        "loss_phys_total": l_total,
    }


def action_classification_loss(action_logits: torch.Tensor, action_label: torch.Tensor) -> Dict[str, torch.Tensor]:
    loss = F.cross_entropy(action_logits, action_label.long())
    pred = action_logits.argmax(dim=1)
    acc = (pred == action_label.long()).float().mean()
    return {"loss_action": loss, "action_acc": acc}


def _align_bvcthw_to_ref(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if src.shape == ref.shape:
        return src
    if src.dim() != 6 or ref.dim() != 6:
        raise ValueError(f"expect 6D [B,V,C,T,H,W], got src={tuple(src.shape)} ref={tuple(ref.shape)}")
    b, v, c, ts, hs, ws = src.shape
    tr, hr, wr = int(ref.shape[3]), int(ref.shape[4]), int(ref.shape[5])
    x = src.view(b * v, c, ts, hs, ws)
    x = F.interpolate(x, size=(tr, hr, wr), mode="trilinear", align_corners=False)
    return x.view(b, v, c, tr, hr, wr)


def mask_field_with_object_mask(
    field_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor,
) -> torch.Tensor:
    """
    将 ``object_mask`` 对齐到 ``field`` 的时空分辨率，通道取均值后按场通道展开相乘。
    用于前向中在送入 stress 融合 / 参数 field 编码器前削弱背景。
    """
    om = _align_bvcthw_to_ref(object_mask_bvcthw.float(), field_bvcthw)
    om = om.mean(dim=2, keepdim=True).clamp(0.0, 1.0).expand_as(field_bvcthw)
    return field_bvcthw * om


def weighted_bce_with_logits_loss(
    logits_bvcthw: torch.Tensor,
    target_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor | None = None,
    *,
    fg_weight: float = 10.0,
    bg_weight: float = 1.0,
    bg_black: bool = True,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    前景/背景加权 BCEWithLogits，与 train 中 _object_mask_weighted_field_loss 的 mask 语义对齐。
    bg_black=True 时背景目标为 0；否则背景仍监督 target。
    """
    logits = logits_bvcthw
    gt = _align_bvcthw_to_ref(target_bvcthw.float(), logits)
    if object_mask_bvcthw is None:
        obj = torch.ones_like(gt[:, :, :1, :, :, :])
    else:
        obj = _align_bvcthw_to_ref(object_mask_bvcthw.float(), logits)
    obj = obj.mean(dim=2, keepdim=True).clamp(0.0, 1.0)
    obj = obj.expand(-1, -1, int(logits.shape[2]), -1, -1, -1)

    if bg_black:
        target_bg = torch.zeros_like(gt)
    else:
        target_bg = gt
    target = obj * gt + (1.0 - obj) * target_bg

    bce = F.binary_cross_entropy_with_logits(
        logits.float(),
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    w_fg = float(max(0.0, fg_weight))
    w_bg = float(max(0.0, bg_weight))
    num = (w_fg * obj * bce + w_bg * (1.0 - obj) * bce).sum()
    den = w_fg * obj.sum() + w_bg * (1.0 - obj).sum()
    if den <= 1e-12:
        return logits.sum() * 0.0
    return num / den


def binary_dice_loss_with_logits(
    logits_bvcthw: torch.Tensor,
    target_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits_bvcthw.float())
    tgt = _align_bvcthw_to_ref(target_bvcthw.float(), prob).clamp(0.0, 1.0)
    if object_mask_bvcthw is not None:
        m = _align_bvcthw_to_ref(object_mask_bvcthw.float(), prob).mean(dim=2, keepdim=True).clamp(0.0, 1.0)
        m = m.expand_as(prob)
    else:
        m = torch.ones_like(prob)

    p = (prob * m).flatten(1)
    t = (tgt * m).flatten(1)
    inter = (p * t).sum(dim=1)
    sp = p.sum(dim=1)
    st = t.sum(dim=1)
    dice = (2.0 * inter + eps) / (sp + st + eps)
    return (1.0 - dice).mean()


def _spatial_grad_xy_bvcthw(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """空间差分梯度，返回 dx, dy，空间维比输入各短 1。"""
    dx = x[..., :, :, :, :, 1:] - x[..., :, :, :, :, :-1]
    dy = x[..., :, :, :, 1:, :] - x[..., :, :, :, :-1, :]
    return dx, dy


def edge_l1_loss_bvcthw(
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    p = pred_bvcthw.float()
    g = _align_bvcthw_to_ref(gt_bvcthw.float(), p)
    dx_p, dy_p = _spatial_grad_xy_bvcthw(p)
    dx_g, dy_g = _spatial_grad_xy_bvcthw(g)
    return (dx_p - dx_g).abs().mean() + (dy_p - dy_g).abs().mean()


def weighted_edge_l1_loss_bvcthw(
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor | None = None,
    *,
    edge_boost: float = 4.0,
    eps: float = 1e-6,
    bg_weight: float = 1.0,
) -> torch.Tensor:
    """
    基于 GT 梯度幅值（分方向）归一化后的边界权重，对 pred/gt 的空间梯度差做 L1。

    - ``bg_weight <= 0``：仅在 **两邻域像素均在 object 内** 的边上累计（背景不参与；边界权重
      ``_norm01`` 只用前景上的 GT 梯度，避免背景纹理影响权重）。
    - 分母与分子均在 **逐通道** 上与 ``om`` 对齐，避免 C 通道重复计数导致的尺度偏差。
    """
    p = pred_bvcthw.float()
    g = _align_bvcthw_to_ref(gt_bvcthw.float(), p)
    dx_p, dy_p = _spatial_grad_xy_bvcthw(p)
    dx_g, dy_g = _spatial_grad_xy_bvcthw(g)

    def _norm01(t: torch.Tensor) -> torch.Tensor:
        tmin = t.amin(dim=(1, 2, 3, 4, 5), keepdim=True)
        tmax = t.amax(dim=(1, 2, 3, 4, 5), keepdim=True)
        return (t - tmin) / (tmax - tmin + eps)

    fg_strict = float(bg_weight) <= 0.0
    if object_mask_bvcthw is not None:
        om = _align_bvcthw_to_ref(object_mask_bvcthw.float(), p).mean(dim=2, keepdim=True).clamp(0.0, 1.0)
        if fg_strict:
            om_dx = torch.minimum(om[..., :, :, :, :, :-1], om[..., :, :, :, :, 1:])
            om_dy = torch.minimum(om[..., :, :, :, :, :-1, :], om[..., :, :, :, :, 1:, :])
        else:
            om_dx = 0.5 * (om[..., :, :, :, :, :-1] + om[..., :, :, :, :, 1:])
            om_dy = 0.5 * (om[..., :, :, :, :, :-1, :] + om[..., :, :, :, :, 1:, :])
    else:
        om_dx = torch.ones_like(dx_p[:, :, :1, :, :, :]).expand_as(dx_p)
        om_dy = torch.ones_like(dy_p[:, :, :1, :, :, :]).expand_as(dy_p)

    adx = dx_g.abs()
    ady = dy_g.abs()
    if fg_strict:
        adx_w = adx * om_dx
        ady_w = ady * om_dy
        w_dx = 1.0 + float(edge_boost) * _norm01(adx_w)
        w_dy = 1.0 + float(edge_boost) * _norm01(ady_w)
    else:
        w_dx = 1.0 + float(edge_boost) * _norm01(adx)
        w_dy = 1.0 + float(edge_boost) * _norm01(ady)

    om_dx_f = om_dx.expand_as(dx_p)
    om_dy_f = om_dy.expand_as(dy_p)
    den_x = om_dx_f.sum()
    den_y = om_dy_f.sum()
    num_x = (om_dx_f * w_dx * (dx_p - dx_g).abs()).sum()
    num_y = (om_dy_f * w_dy * (dy_p - dy_g).abs()).sum()
    loss_dx = torch.where(den_x > eps, num_x / den_x, num_x.new_zeros(()))
    loss_dy = torch.where(den_y > eps, num_y / den_y, num_y.new_zeros(()))
    return loss_dx + loss_dy


__all__ = [
    "Arch4LossConfig",
    "Arch4RegressionLoss",
    "arch4_field_supervision_mse",
    "PhysicsConsistencyConfig",
    "compute_physics_consistency_losses",
    "action_classification_loss",
    "mask_field_with_object_mask",
    "weighted_bce_with_logits_loss",
    "binary_dice_loss_with_logits",
    "edge_l1_loss_bvcthw",
    "weighted_edge_l1_loss_bvcthw",
]

