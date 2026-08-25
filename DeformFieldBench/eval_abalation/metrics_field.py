from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


FIELD_KEYS: Tuple[str, ...] = ("stress", "flow", "force_mask")
FLOW_CHANGE_KEEP_RATIO = 0.15
FLOW_VECTOR_COVERAGE_EPS = 2.0 / 255.0
STRESS_HIGH_KEEP_RATIO = 0.10
FORCE_THRESHOLD = 0.5
STRESS_PALETTE_ANCHORS: Tuple[Tuple[float, float, float], ...] = (
    (0.02, 0.05, 0.50),
    (0.00, 0.20, 1.00),
    (0.00, 0.90, 1.00),
    (0.10, 1.00, 0.25),
    (0.50, 1.00, 0.00),
    (1.00, 0.95, 0.00),
    (1.00, 0.45, 0.00),
    (1.00, 0.00, 0.00),
    (0.65, 0.00, 0.15),
)
FIELD_METRIC_KEYS: Tuple[str, ...] = (
    "mse",
    "gradient_l1",
    "ssim",
    "region_iou",
    "region_recall",
    "soft_region_iou",
    "region_dice",
    "coverage_l1",
    "gt_weighted_recall",
    "flow_coverage_iou",
    "flow_coverage_recall",
    "flow_velocity_weighted_overlap",
    "flow_velocity_similarity_overlap",
    "flow_velocity_weighted_epe",
    "flow_motion_quality",
    "flow_change_recall_main",
    "flow_change_soft_overlap",
    "flow_temporal_consistency",
    "force_main_recall",
    "force_centroid_dist",
    "force_area_ratio_err",
    "stress_hotspot_recall",
    "stress_hotspot_soft_overlap",
    "stress_weighted_mae",
    "stress_rank_corr",
)


def align_field_to_prediction(gt_bvcthw: torch.Tensor, pred_bvcthw: torch.Tensor) -> torch.Tensor:
    if gt_bvcthw.shape == pred_bvcthw.shape:
        return gt_bvcthw
    if gt_bvcthw.dim() != 6 or pred_bvcthw.dim() != 6:
        raise ValueError(f"expect 6D tensors, got gt={tuple(gt_bvcthw.shape)} pred={tuple(pred_bvcthw.shape)}")
    b, v, c, t, h, w = gt_bvcthw.shape
    tp, hp, wp = int(pred_bvcthw.shape[3]), int(pred_bvcthw.shape[4]), int(pred_bvcthw.shape[5])
    x = gt_bvcthw.reshape(b * v, c, t, h, w)
    y = F.interpolate(x.float(), size=(tp, hp, wp), mode="trilinear", align_corners=False)
    return y.view(b, v, c, tp, hp, wp).contiguous()


def _align_object_mask_to_prediction(
    object_mask_bvcthw: torch.Tensor | None,
    ref_bvcthw: torch.Tensor,
) -> torch.Tensor | None:
    if object_mask_bvcthw is None:
        return None
    om = align_field_to_prediction(object_mask_bvcthw.float(), ref_bvcthw.float())
    om = om.mean(dim=2, keepdim=True).clamp(0.0, 1.0)
    return om.expand(-1, -1, int(ref_bvcthw.shape[2]), -1, -1, -1)


def _scalar_object_mask(
    object_mask_bvcthw: torch.Tensor | None,
    ref_bvcthw: torch.Tensor,
) -> torch.Tensor | None:
    om = _align_object_mask_to_prediction(object_mask_bvcthw, ref_bvcthw)
    if om is None:
        return None
    return om[:, :, 0, :, :, :]


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        return value.mean()
    den = mask.sum()
    if float(den) <= eps:
        return value.new_zeros(())
    return (value * mask).sum() / den


def _masked_weighted_mean(value: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    eff = weight.float()
    if mask is not None:
        eff = eff * mask.float()
    den = eff.sum()
    if float(den) <= 1e-6:
        return value.new_zeros(())
    return (value.float() * eff).sum() / den


def gradient_l1(
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor | None = None,
) -> torch.Tensor:
    dx_p = pred_bvcthw[..., 1:] - pred_bvcthw[..., :-1]
    dx_g = gt_bvcthw[..., 1:] - gt_bvcthw[..., :-1]
    dy_p = pred_bvcthw[..., 1:, :] - pred_bvcthw[..., :-1, :]
    dy_g = gt_bvcthw[..., 1:, :] - gt_bvcthw[..., :-1, :]
    om = _align_object_mask_to_prediction(object_mask_bvcthw, pred_bvcthw)
    if om is None:
        mask_dx = None
        mask_dy = None
    else:
        mask_dx = torch.minimum(om[..., 1:], om[..., :-1])
        mask_dy = torch.minimum(om[..., 1:, :], om[..., :-1, :])
    return _masked_mean((dx_p - dx_g).abs(), mask_dx) + _masked_mean((dy_p - dy_g).abs(), mask_dy)


def _normalize_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        lo = torch.minimum(x.amin(dim=(-2, -1), keepdim=True), y.amin(dim=(-2, -1), keepdim=True))
        hi = torch.maximum(x.amax(dim=(-2, -1), keepdim=True), y.amax(dim=(-2, -1), keepdim=True))
    else:
        inf = torch.full_like(x, float("inf"))
        ninf = torch.full_like(x, float("-inf"))
        x_min = torch.where(mask > 0, x, inf).amin(dim=(-2, -1), keepdim=True)
        y_min = torch.where(mask > 0, y, inf).amin(dim=(-2, -1), keepdim=True)
        x_max = torch.where(mask > 0, x, ninf).amax(dim=(-2, -1), keepdim=True)
        y_max = torch.where(mask > 0, y, ninf).amax(dim=(-2, -1), keepdim=True)
        lo = torch.minimum(x_min, y_min)
        hi = torch.maximum(x_max, y_max)
        invalid = ~torch.isfinite(lo) | ~torch.isfinite(hi)
        lo = torch.where(invalid, torch.zeros_like(lo), lo)
        hi = torch.where(invalid, torch.ones_like(hi), hi)
    den = (hi - lo).clamp(min=1e-6)
    return (x - lo) / den, (y - lo) / den


def _normalize_by_gt(
    gt_score: torch.Tensor,
    pred_score: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reduce_dims = (-3, -2, -1)
    if mask is None:
        lo = gt_score.amin(dim=reduce_dims, keepdim=True)
        hi = gt_score.amax(dim=reduce_dims, keepdim=True)
    else:
        inf = torch.full_like(gt_score, float("inf"))
        ninf = torch.full_like(gt_score, float("-inf"))
        lo = torch.where(mask > 0, gt_score, inf).amin(dim=reduce_dims, keepdim=True)
        hi = torch.where(mask > 0, gt_score, ninf).amax(dim=reduce_dims, keepdim=True)
        invalid = ~torch.isfinite(lo) | ~torch.isfinite(hi)
        lo = torch.where(invalid, torch.zeros_like(lo), lo)
        hi = torch.where(invalid, torch.ones_like(hi), hi)
    den = (hi - lo).clamp(min=1e-6)
    gt_norm = ((gt_score - lo) / den).clamp(0.0, 1.0)
    pred_norm = ((pred_score - lo) / den).clamp(0.0, 1.0)
    return gt_norm, pred_norm


def ssim_mean(
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor | None = None,
) -> torch.Tensor:
    x = pred_bvcthw.float().reshape(-1, pred_bvcthw.shape[-2], pred_bvcthw.shape[-1])
    y = gt_bvcthw.float().reshape(-1, gt_bvcthw.shape[-2], gt_bvcthw.shape[-1])
    om = _align_object_mask_to_prediction(object_mask_bvcthw, pred_bvcthw)
    mask = None
    if om is not None:
        mask = om.reshape(-1, pred_bvcthw.shape[-2], pred_bvcthw.shape[-1])
    x, y = _normalize_pair(x, y, mask)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    if mask is None:
        mu_x = x.mean(dim=(-2, -1), keepdim=True)
        mu_y = y.mean(dim=(-2, -1), keepdim=True)
        sigma_x = ((x - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
        sigma_y = ((y - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(-2, -1), keepdim=True)
        ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
        ).clamp(min=1e-12)
        return ssim_map.mean()
    den = mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    mu_x = (x * mask).sum(dim=(-2, -1), keepdim=True) / den
    mu_y = (y * mask).sum(dim=(-2, -1), keepdim=True) / den
    sigma_x = (((x - mu_x) ** 2) * mask).sum(dim=(-2, -1), keepdim=True) / den
    sigma_y = (((y - mu_y) ** 2) * mask).sum(dim=(-2, -1), keepdim=True) / den
    sigma_xy = (((x - mu_x) * (y - mu_y)) * mask).sum(dim=(-2, -1), keepdim=True) / den
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    return (num / den.clamp(min=1e-12)).mean()


def _to_scalar_field(field_bvcthw: torch.Tensor) -> torch.Tensor:
    if field_bvcthw.dim() != 6:
        raise ValueError(f"expect [B,V,C,T,H,W], got {tuple(field_bvcthw.shape)}")
    return field_bvcthw.float().mean(dim=2)


def _rgb_field_to_hsv(field_bvcthw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(field_bvcthw.shape[2]) < 3:
        v = _to_scalar_field(field_bvcthw).clamp(0.0, 1.0)
        z = torch.zeros_like(v)
        return z, z, v
    rgb = field_bvcthw[:, :, :3, :, :, :].float().clamp(0.0, 1.0).permute(0, 1, 3, 4, 5, 2)
    r, g, b = rgb.unbind(dim=-1)
    maxc = rgb.amax(dim=-1)
    minc = rgb.amin(dim=-1)
    delta = maxc - minc
    sat = torch.where(maxc > 1e-6, delta / maxc.clamp(min=1e-6), torch.zeros_like(maxc))
    hue = torch.zeros_like(maxc)
    nonzero = delta > 1e-6
    rc = (g - b) / delta.clamp(min=1e-6)
    gc = (b - r) / delta.clamp(min=1e-6) + 2.0
    bc = (r - g) / delta.clamp(min=1e-6) + 4.0
    hue = torch.where(nonzero & (maxc == r), rc, hue)
    hue = torch.where(nonzero & (maxc == g), gc, hue)
    hue = torch.where(nonzero & (maxc == b), bc, hue)
    hue = torch.remainder(hue / 6.0, 1.0)
    return hue, sat.clamp(0.0, 1.0), maxc.clamp(0.0, 1.0)


def _stress_palette(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(STRESS_PALETTE_ANCHORS, device=device, dtype=dtype)


def _decode_stress_heat(field_bvcthw: torch.Tensor) -> torch.Tensor:
    if int(field_bvcthw.shape[2]) == 1:
        return _to_scalar_field(field_bvcthw).clamp(0.0, 1.0)
    rgb = field_bvcthw[:, :, :3, :, :, :].float().clamp(0.0, 1.0).permute(0, 1, 3, 4, 5, 2)
    gray = rgb.mean(dim=-1)
    sat = rgb.amax(dim=-1) - rgb.amin(dim=-1)
    palette = _stress_palette(rgb.device, rgb.dtype)
    dists = (rgb.unsqueeze(-2) - palette.view(*((1,) * (rgb.dim() - 1)), palette.shape[0], 3)).pow(2).sum(dim=-1)
    idx = dists.argmin(dim=-1).float()
    pal = idx / max(int(palette.shape[0]) - 1, 1)
    return torch.where(sat > 0.05, pal, gray).clamp(0.0, 1.0)


def _force_soft_mask(field_bvcthw: torch.Tensor) -> torch.Tensor:
    scalar = _to_scalar_field(field_bvcthw).clamp(0.0, 1.0)
    if int(field_bvcthw.shape[2]) < 3:
        return scalar
    rgb = field_bvcthw[:, :, :3, :, :, :].float().clamp(0.0, 1.0)
    r = rgb[:, :, 0, :, :, :]
    g = rgb[:, :, 1, :, :, :]
    b = rgb[:, :, 2, :, :, :]
    sat = rgb.amax(dim=2) - rgb.amin(dim=2)
    red_dom = torch.clamp(r - torch.maximum(g, b), min=0.0, max=1.0)
    return torch.where(sat > 0.05, red_dom, scalar).clamp(0.0, 1.0)


def _spatiotemporal_gradient_energy(score_bvthw: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(score_bvthw)
    dy = torch.zeros_like(score_bvthw)
    dt = torch.zeros_like(score_bvthw)
    dx[..., 1:] = (score_bvthw[..., 1:] - score_bvthw[..., :-1]).abs()
    dy[..., 1:, :] = (score_bvthw[..., 1:, :] - score_bvthw[..., :-1, :]).abs()
    dt[:, :, 1:, :, :] = (score_bvthw[:, :, 1:, :, :] - score_bvthw[:, :, :-1, :, :]).abs()
    return dx + dy + dt


def _flow_change_score(field_bvcthw: torch.Tensor) -> torch.Tensor:
    _, _, val = _rgb_field_to_hsv(field_bvcthw)
    return _spatiotemporal_gradient_energy(val)


def _decode_flow_uv(field_bvcthw: torch.Tensor) -> torch.Tensor:
    hue, sat, val = _rgb_field_to_hsv(field_bvcthw)
    angle = hue * (2.0 * math.pi) - math.pi
    ux = val * torch.cos(angle)
    uy = val * torch.sin(angle)
    # Low-saturation inputs appear in scalar-grid fallbacks; preserve magnitude there.
    fallback = sat <= 0.05
    ux = torch.where(fallback, val, ux)
    uy = torch.where(fallback, torch.zeros_like(val), uy)
    return torch.stack([ux, uy], dim=-1)


def _flow_speed_from_uv(flow_uv: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(flow_uv.float(), dim=-1)


def _flow_vector_similarity(pred_uv: torch.Tensor, gt_uv: torch.Tensor) -> torch.Tensor:
    pred = pred_uv.float()
    gt = gt_uv.float()
    pred_mag = _flow_speed_from_uv(pred)
    gt_mag = _flow_speed_from_uv(gt)
    diff = torch.linalg.norm(pred - gt, dim=-1)
    denom = (pred_mag + gt_mag).clamp(min=1e-6)
    sim = 1.0 - diff / denom
    both_zero = (pred_mag <= 1e-8) & (gt_mag <= 1e-8)
    return torch.where(both_zero, torch.ones_like(sim), sim.clamp(0.0, 1.0))


def _flow_weighted_overlap(
    pred_uv: torch.Tensor,
    gt_uv: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    pred_mag = _flow_speed_from_uv(pred_uv)
    gt_mag = _flow_speed_from_uv(gt_uv)
    sim = _flow_vector_similarity(pred_uv, gt_uv)
    inter = torch.minimum(pred_mag, gt_mag) * sim
    union = torch.maximum(pred_mag, gt_mag)
    if valid_mask is not None:
        valid = valid_mask.float()
        inter = inter * valid
        union = union * valid
    union_sum = float(union.sum().item())
    if union_sum <= 1e-8:
        return 1.0
    return float((inter.sum() / union.sum().clamp(min=1e-8)).item())


def _flow_similarity_on_overlap(
    pred_uv: torch.Tensor,
    gt_uv: torch.Tensor,
    pred_region: torch.Tensor,
    gt_region: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    overlap = pred_region.bool() & gt_region.bool()
    if valid_mask is not None:
        overlap = overlap & (valid_mask > 0)
    if not bool(overlap.any().item()):
        pred_has = pred_region.bool()
        gt_has = gt_region.bool()
        if valid_mask is not None:
            valid = valid_mask > 0
            pred_has = pred_has & valid
            gt_has = gt_has & valid
        return 1.0 if not bool(pred_has.any().item() or gt_has.any().item()) else 0.0
    sim = _flow_vector_similarity(pred_uv, gt_uv)
    return float(sim[overlap].mean().item())


def _masked_threshold_from_gt(gt_score: torch.Tensor, mask: torch.Tensor | None, keep_ratio: float) -> torch.Tensor:
    gt_r = gt_score.reshape(-1, *gt_score.shape[-3:])
    if mask is None:
        mask_r = torch.ones_like(gt_r)
    else:
        mask_r = mask.reshape(-1, *mask.shape[-3:])
    out = gt_r.new_zeros((gt_r.shape[0], 1, 1, 1))
    q = float(max(0.0, min(1.0, 1.0 - keep_ratio)))
    for i in range(int(gt_r.shape[0])):
        valid = mask_r[i] > 0
        vals = gt_r[i][valid]
        if vals.numel() == 0:
            out[i, 0, 0, 0] = 1.0
        else:
            out[i, 0, 0, 0] = torch.quantile(vals, q=q)
    return out.view(*gt_score.shape[:-3], 1, 1, 1)


def _binary_region_metrics(
    pred_region: torch.Tensor,
    gt_region: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    pred = pred_region.bool()
    gt = gt_region.bool()
    if valid_mask is not None:
        valid = valid_mask > 0
        pred = pred & valid
        gt = gt & valid
    pred_f = pred.float()
    gt_f = gt.float()
    inter = float((pred_f * gt_f).sum().item())
    pred_sum = float(pred_f.sum().item())
    gt_sum = float(gt_f.sum().item())
    union = pred_sum + gt_sum - inter
    iou = 1.0 if union <= 1e-8 else inter / union
    if gt_sum <= 1e-8:
        recall = 1.0 if pred_sum <= 1e-8 else 0.0
        coverage = 0.0 if pred_sum <= 1e-8 else 1.0
    else:
        recall = inter / gt_sum
        coverage = abs(pred_sum - gt_sum) / gt_sum
    dice_den = pred_sum + gt_sum
    dice = 1.0 if dice_den <= 1e-8 else (2.0 * inter / dice_den)
    return {
        "region_iou": float(iou),
        "region_recall": float(recall),
        "region_dice": float(dice),
        "coverage_l1": float(coverage),
    }


def _soft_region_metrics(
    pred_score: torch.Tensor,
    gt_score: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    pred = pred_score.float()
    gt = gt_score.float()
    if valid_mask is not None:
        valid = valid_mask.float()
        pred = pred * valid
        gt = gt * valid
    inter = torch.minimum(pred, gt).sum()
    union = torch.maximum(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    soft_iou = 1.0 if float(union) <= 1e-8 else float((inter / union).item())
    if float(gt_sum) <= 1e-8:
        weighted_recall = 1.0 if float(pred_sum) <= 1e-8 else 0.0
        coverage = 0.0 if float(pred_sum) <= 1e-8 else 1.0
    else:
        weighted_recall = float(((gt * pred).sum() / gt_sum).item())
        coverage = float((pred_sum - gt_sum).abs().div(gt_sum).item())
    return {
        "soft_region_iou": float(soft_iou),
        "coverage_l1": float(coverage),
        "gt_weighted_recall": float(weighted_recall),
    }


def _empty_region_metrics() -> Dict[str, float | None]:
    return {
        "region_iou": None,
        "region_recall": None,
        "soft_region_iou": None,
        "region_dice": None,
        "coverage_l1": None,
        "gt_weighted_recall": None,
    }


def _empty_rgb_semantic_metrics() -> Dict[str, float | None]:
    return {
        "flow_coverage_iou": None,
        "flow_coverage_recall": None,
        "flow_velocity_weighted_overlap": None,
        "flow_velocity_similarity_overlap": None,
        "flow_velocity_weighted_epe": None,
        "flow_motion_quality": None,
        "flow_change_recall_main": None,
        "flow_change_soft_overlap": None,
        "flow_temporal_consistency": None,
        "force_main_recall": None,
        "force_centroid_dist": None,
        "force_area_ratio_err": None,
        "stress_hotspot_recall": None,
        "stress_hotspot_soft_overlap": None,
        "stress_weighted_mae": None,
        "stress_rank_corr": None,
    }


def _temporal_distribution_similarity(
    pred_score: torch.Tensor,
    gt_score: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> float:
    pred = pred_score.float()
    gt = gt_score.float()
    if valid_mask is not None:
        valid = valid_mask.float()
        pred = pred * valid
        gt = gt * valid
        den = valid.sum(dim=(-2, -1)).clamp(min=1e-6)
        pred_curve = pred.sum(dim=(-2, -1)) / den
        gt_curve = gt.sum(dim=(-2, -1)) / den
    else:
        pred_curve = pred.mean(dim=(-2, -1))
        gt_curve = gt.mean(dim=(-2, -1))
    pred_flat = pred_curve.reshape(-1, pred_curve.shape[-1])
    gt_flat = gt_curve.reshape(-1, gt_curve.shape[-1])
    sims: List[float] = []
    for i in range(int(pred_flat.shape[0])):
        p = pred_flat[i]
        g = gt_flat[i]
        sp = float(p.sum().item())
        sg = float(g.sum().item())
        if sp <= 1e-8 and sg <= 1e-8:
            sims.append(1.0)
            continue
        p = p / max(sp, 1e-8)
        g = g / max(sg, 1e-8)
        sims.append(float((1.0 - 0.5 * (p - g).abs().sum()).item()))
    return float(sum(sims) / max(len(sims), 1))


def _project_force_region(score_bvthw: torch.Tensor) -> torch.Tensor:
    return score_bvthw.amax(dim=(1, 2))


def _centroid_distance(
    pred_hw: torch.Tensor,
    gt_hw: torch.Tensor,
    valid_mask_hw: torch.Tensor | None,
) -> float:
    b, h, w = pred_hw.shape
    ys = torch.linspace(0.0, 1.0, h, device=pred_hw.device, dtype=pred_hw.dtype).view(1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, device=pred_hw.device, dtype=pred_hw.dtype).view(1, 1, w)
    dists: List[float] = []
    for i in range(b):
        p = pred_hw[i]
        g = gt_hw[i]
        if valid_mask_hw is not None:
            vm = valid_mask_hw[i].float()
            p = p * vm
            g = g * vm
        ps = float(p.sum().item())
        gs = float(g.sum().item())
        if ps <= 1e-8 and gs <= 1e-8:
            dists.append(0.0)
            continue
        if ps <= 1e-8 or gs <= 1e-8:
            dists.append(1.0)
            continue
        py = float((p * ys[0]).sum().item() / ps)
        px = float((p * xs[0]).sum().item() / ps)
        gy = float((g * ys[0]).sum().item() / gs)
        gx = float((g * xs[0]).sum().item() / gs)
        dist = ((py - gy) ** 2 + (px - gx) ** 2) ** 0.5 / (2.0 ** 0.5)
        dists.append(float(dist))
    return float(sum(dists) / max(len(dists), 1))


def _area_ratio_error(
    pred_hw: torch.Tensor,
    gt_hw: torch.Tensor,
    valid_mask_hw: torch.Tensor | None,
) -> float:
    errs: List[float] = []
    for i in range(int(pred_hw.shape[0])):
        p = pred_hw[i]
        g = gt_hw[i]
        if valid_mask_hw is not None:
            vm = valid_mask_hw[i].float()
            p = p * vm
            g = g * vm
        ps = float(p.sum().item())
        gs = float(g.sum().item())
        if gs <= 1e-8:
            errs.append(0.0 if ps <= 1e-8 else 1.0)
        else:
            errs.append(abs(ps - gs) / gs)
    return float(sum(errs) / max(len(errs), 1))


def _spearman_rank_corr(
    pred_score: torch.Tensor,
    gt_score: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> float:
    pred_flat = pred_score.reshape(-1)
    gt_flat = gt_score.reshape(-1)
    if valid_mask is None:
        valid = torch.ones_like(pred_flat, dtype=torch.bool)
    else:
        valid = valid_mask.reshape(-1) > 0
    pred_flat = pred_flat[valid]
    gt_flat = gt_flat[valid]
    if int(pred_flat.numel()) < 2:
        return 1.0
    pred_order = torch.argsort(torch.argsort(pred_flat))
    gt_order = torch.argsort(torch.argsort(gt_flat))
    pr = pred_order.float()
    gr = gt_order.float()
    pr = pr - pr.mean()
    gr = gr - gr.mean()
    den = pr.norm() * gr.norm()
    if float(den) <= 1e-8:
        return 1.0
    return float((pr * gr).sum().div(den).item())


def _field_specific_region_metrics(
    *,
    field_name: str,
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor | None,
) -> Dict[str, float | None]:
    valid_mask = _scalar_object_mask(object_mask_bvcthw, pred_bvcthw)
    if field_name == "flow":
        gt_uv = _decode_flow_uv(gt_bvcthw)
        pred_uv = _decode_flow_uv(pred_bvcthw)
        gt_score = _flow_speed_from_uv(gt_uv)
        pred_score = _flow_speed_from_uv(pred_uv)
        gt_region = gt_score > FLOW_VECTOR_COVERAGE_EPS
        pred_region = pred_score > FLOW_VECTOR_COVERAGE_EPS
    elif field_name == "stress":
        gt_score_raw = _to_scalar_field(gt_bvcthw)
        pred_score_raw = _to_scalar_field(pred_bvcthw)
        gt_score, pred_score = _normalize_by_gt(gt_score_raw, pred_score_raw, valid_mask)
        thr = _masked_threshold_from_gt(gt_score, valid_mask, keep_ratio=STRESS_HIGH_KEEP_RATIO)
        gt_region = gt_score >= thr
        pred_region = pred_score >= thr
    elif field_name == "force_mask":
        gt_score = _to_scalar_field(gt_bvcthw).clamp(0.0, 1.0)
        pred_score = _to_scalar_field(pred_bvcthw).clamp(0.0, 1.0)
        gt_region = gt_score >= FORCE_THRESHOLD
        pred_region = pred_score >= FORCE_THRESHOLD
    else:
        return _empty_region_metrics()
    out: Dict[str, float | None] = _empty_region_metrics()
    out.update(_binary_region_metrics(pred_region, gt_region, valid_mask))
    out.update(_soft_region_metrics(pred_score, gt_score, valid_mask))
    return out


def _flow_rgb_semantic_metrics(
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> Dict[str, float | None]:
    gt_uv = _decode_flow_uv(gt_bvcthw)
    pred_uv = _decode_flow_uv(pred_bvcthw)
    gt_speed = _flow_speed_from_uv(gt_uv)
    pred_speed = _flow_speed_from_uv(pred_uv)
    gt_region = gt_speed > FLOW_VECTOR_COVERAGE_EPS
    pred_region = pred_speed > FLOW_VECTOR_COVERAGE_EPS
    hard = _binary_region_metrics(pred_region, gt_region, valid_mask)
    weighted_overlap = _flow_weighted_overlap(pred_uv, gt_uv, valid_mask)
    similarity_overlap = _flow_similarity_on_overlap(pred_uv, gt_uv, pred_region, gt_region, valid_mask)
    weighted_epe = _masked_weighted_mean(
        torch.linalg.norm(pred_uv.float() - gt_uv.float(), dim=-1),
        torch.maximum(pred_speed, gt_speed),
        valid_mask,
    )
    motion_quality = (max(float(hard["region_iou"]), 0.0) * max(float(weighted_overlap), 0.0)) ** 0.5
    return {
        "flow_coverage_iou": hard["region_iou"],
        "flow_coverage_recall": hard["region_recall"],
        "flow_velocity_weighted_overlap": weighted_overlap,
        "flow_velocity_similarity_overlap": similarity_overlap,
        "flow_velocity_weighted_epe": float(weighted_epe.item()),
        "flow_motion_quality": float(motion_quality),
        # Keep legacy keys populated so existing readers do not break immediately.
        "flow_change_recall_main": hard["region_recall"],
        "flow_change_soft_overlap": weighted_overlap,
        "flow_temporal_consistency": _temporal_distribution_similarity(pred_speed, gt_speed, valid_mask),
    }


def _force_main_region_metrics(
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> Dict[str, float | None]:
    pred_soft = _force_soft_mask(pred_bvcthw)
    gt_soft = _force_soft_mask(gt_bvcthw)
    pred_proj = _project_force_region(pred_soft)
    gt_proj = _project_force_region(gt_soft)
    valid_proj = None if valid_mask is None else _project_force_region(valid_mask)
    hard = _binary_region_metrics(pred_proj >= FORCE_THRESHOLD, gt_proj >= FORCE_THRESHOLD, valid_proj)
    return {
        "force_main_recall": hard["region_recall"],
        "force_centroid_dist": _centroid_distance(pred_proj, gt_proj, valid_proj),
        "force_area_ratio_err": _area_ratio_error(pred_proj, gt_proj, valid_proj),
    }


def _stress_rgb_semantic_metrics(
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> Dict[str, float | None]:
    gt_heat = _decode_stress_heat(gt_bvcthw)
    pred_heat = _decode_stress_heat(pred_bvcthw)
    thr = _masked_threshold_from_gt(gt_heat, valid_mask, keep_ratio=STRESS_HIGH_KEEP_RATIO)
    gt_region = gt_heat >= thr
    pred_region = pred_heat >= thr
    hard = _binary_region_metrics(pred_region, gt_region, valid_mask)
    soft = _soft_region_metrics(pred_heat, gt_heat, valid_mask)
    weights = 0.10 + 0.90 * gt_heat.pow(2)
    weighted_mae = _masked_weighted_mean((pred_heat - gt_heat).abs(), weights, valid_mask)
    return {
        "stress_hotspot_recall": hard["region_recall"],
        "stress_hotspot_soft_overlap": soft["soft_region_iou"],
        "stress_weighted_mae": float(weighted_mae.item()),
        "stress_rank_corr": _spearman_rank_corr(pred_heat, gt_heat, valid_mask),
    }


def _rgb_semantic_metrics(
    *,
    field_name: str,
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor | None,
) -> Dict[str, float | None]:
    valid_mask = _scalar_object_mask(object_mask_bvcthw, pred_bvcthw)
    out = _empty_rgb_semantic_metrics()
    if field_name == "flow":
        out.update(_flow_rgb_semantic_metrics(pred_bvcthw, gt_bvcthw, valid_mask))
    elif field_name == "force_mask":
        out.update(_force_main_region_metrics(pred_bvcthw, gt_bvcthw, valid_mask))
    elif field_name == "stress":
        out.update(_stress_rgb_semantic_metrics(pred_bvcthw, gt_bvcthw, valid_mask))
    return out


def build_field_sample_record(
    *,
    sample_id: str,
    action: str,
    material: str,
    object_name: str,
    field_name: str,
    pred_bvcthw: torch.Tensor,
    gt_bvcthw: torch.Tensor,
    object_mask_bvcthw: torch.Tensor | None = None,
) -> Dict[str, object]:
    gt_aligned = align_field_to_prediction(gt_bvcthw, pred_bvcthw)
    pred = pred_bvcthw.float()
    gt = gt_aligned.float()
    om = _align_object_mask_to_prediction(object_mask_bvcthw, pred)
    mse = _masked_mean((pred - gt) ** 2, om).item()
    grad = gradient_l1(pred, gt, object_mask_bvcthw=object_mask_bvcthw).item()
    ssim = ssim_mean(pred, gt, object_mask_bvcthw=object_mask_bvcthw).item()
    region_metrics = _field_specific_region_metrics(
        field_name=field_name,
        pred_bvcthw=pred,
        gt_bvcthw=gt,
        object_mask_bvcthw=object_mask_bvcthw,
    )
    rgb_metrics = _rgb_semantic_metrics(
        field_name=field_name,
        pred_bvcthw=pred,
        gt_bvcthw=gt,
        object_mask_bvcthw=object_mask_bvcthw,
    )
    return {
        "sample_id": sample_id,
        "action": action,
        "material": material,
        "object_name": object_name,
        "field_name": field_name,
        "mse": float(mse),
        "gradient_l1": float(grad),
        "ssim": float(ssim),
        "region_iou": region_metrics["region_iou"],
        "region_recall": region_metrics["region_recall"],
        "soft_region_iou": region_metrics["soft_region_iou"],
        "region_dice": region_metrics["region_dice"],
        "coverage_l1": region_metrics["coverage_l1"],
        "gt_weighted_recall": region_metrics["gt_weighted_recall"],
        "flow_coverage_iou": rgb_metrics["flow_coverage_iou"],
        "flow_coverage_recall": rgb_metrics["flow_coverage_recall"],
        "flow_velocity_weighted_overlap": rgb_metrics["flow_velocity_weighted_overlap"],
        "flow_velocity_similarity_overlap": rgb_metrics["flow_velocity_similarity_overlap"],
        "flow_velocity_weighted_epe": rgb_metrics["flow_velocity_weighted_epe"],
        "flow_motion_quality": rgb_metrics["flow_motion_quality"],
        "flow_change_recall_main": rgb_metrics["flow_change_recall_main"],
        "flow_change_soft_overlap": rgb_metrics["flow_change_soft_overlap"],
        "flow_temporal_consistency": rgb_metrics["flow_temporal_consistency"],
        "force_main_recall": rgb_metrics["force_main_recall"],
        "force_centroid_dist": rgb_metrics["force_centroid_dist"],
        "force_area_ratio_err": rgb_metrics["force_area_ratio_err"],
        "stress_hotspot_recall": rgb_metrics["stress_hotspot_recall"],
        "stress_hotspot_soft_overlap": rgb_metrics["stress_hotspot_soft_overlap"],
        "stress_weighted_mae": rgb_metrics["stress_weighted_mae"],
        "stress_rank_corr": rgb_metrics["stress_rank_corr"],
    }


def _group_key(row: Dict[str, object], group_fields: Sequence[str]) -> Tuple[object, ...]:
    return tuple(row.get(k, "") for k in group_fields)


def _mean_metric(rows: Sequence[Dict[str, object]], key: str) -> float | None:
    vals: List[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        fv = float(value)
        if fv == fv:
            vals.append(fv)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def aggregate_field_records(
    records: Iterable[Dict[str, object]],
    *,
    group_fields: Sequence[str],
) -> List[Dict[str, object]]:
    rec_list = list(records)
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rec_list:
        grouped.setdefault(_group_key(row, group_fields + ("field_name",)), []).append(row)
    out: List[Dict[str, object]] = []
    for key_vals, rows in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        count = len(rows)
        out_row: Dict[str, object] = {"count": count}
        for metric_key in FIELD_METRIC_KEYS:
            out_row[metric_key] = _mean_metric(rows, metric_key)
        for field, val in zip(tuple(group_fields) + ("field_name",), key_vals):
            out_row[field] = val
        out.append(out_row)
    return out
