from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from eval_abalation.metrics_field import (
    _decode_flow_uv,
    _decode_stress_heat,
    _flow_change_score,
    _flow_speed_from_uv,
    _force_soft_mask,
    align_field_to_prediction,
)


def build_field_export_payload(
    *,
    sample_id: str,
    stress_pred: torch.Tensor,
    stress_gt: torch.Tensor,
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    force_pred: torch.Tensor,
    force_gt: torch.Tensor,
    object_mask: torch.Tensor | None,
    meta: Dict[str, object],
) -> Dict[str, object]:
    stress_pred_cpu = stress_pred.detach().cpu().float().contiguous()
    flow_pred_cpu = flow_pred.detach().cpu().float().contiguous()
    force_pred_cpu = force_pred.detach().cpu().float().contiguous()
    stress_gt_cpu = align_field_to_prediction(stress_gt.detach().cpu().float(), stress_pred_cpu)
    flow_gt_cpu = align_field_to_prediction(flow_gt.detach().cpu().float(), flow_pred_cpu)
    force_gt_cpu = align_field_to_prediction(force_gt.detach().cpu().float(), force_pred_cpu)
    object_mask_cpu = None if object_mask is None else object_mask.detach().cpu().float().contiguous()
    meta_out = dict(meta)
    meta_out.setdefault("field_metric_version", "rgb_semantic_v3_flow_uv_quality")
    meta_out.setdefault("object_mask_policy", "all_ones_when_missing")
    meta_out.setdefault("flow_visual_encoding", "rgb_to_normalized_screen_uv")
    meta_out.setdefault("stress_visual_encoding", "jet_like_palette_nearest_or_gray_fallback")
    meta_out.setdefault("force_visual_encoding", "red_dominant_or_gray_soft_mask")
    meta_out.setdefault(
        "video_render_note",
        "mp4 display uses per-frame visualization; numeric metrics use mask-aware decoding of normalized screen-space flow vectors",
    )
    flow_uv_pred = _decode_flow_uv(flow_pred_cpu)
    flow_uv_gt = _decode_flow_uv(flow_gt_cpu)
    meta_out.setdefault(
        "stress_palette_anchor_rgb",
        [
            [0.02, 0.05, 0.50],
            [0.00, 0.20, 1.00],
            [0.00, 0.90, 1.00],
            [0.10, 1.00, 0.25],
            [0.50, 1.00, 0.00],
            [1.00, 0.95, 0.00],
            [1.00, 0.45, 0.00],
            [1.00, 0.00, 0.00],
            [0.65, 0.00, 0.15],
        ],
    )
    meta_out["object_mask_included"] = object_mask_cpu is not None
    return {
        "sample_id": str(sample_id),
        "stress_pred": stress_pred_cpu,
        "stress_gt": stress_gt_cpu,
        "flow_pred": flow_pred_cpu,
        "flow_gt": flow_gt_cpu,
        "force_pred": force_pred_cpu,
        "force_gt": force_gt_cpu,
        "stress_heat_pred": _decode_stress_heat(stress_pred_cpu),
        "stress_heat_gt": _decode_stress_heat(stress_gt_cpu),
        "flow_uv_pred": flow_uv_pred,
        "flow_uv_gt": flow_uv_gt,
        "flow_speed_pred": _flow_speed_from_uv(flow_uv_pred),
        "flow_speed_gt": _flow_speed_from_uv(flow_uv_gt),
        "flow_change_pred": _flow_change_score(flow_pred_cpu),
        "flow_change_gt": _flow_change_score(flow_gt_cpu),
        "force_soft_pred": _force_soft_mask(force_pred_cpu),
        "force_soft_gt": _force_soft_mask(force_gt_cpu),
        "object_mask": object_mask_cpu,
        "meta": meta_out,
    }


def export_field_tensors_pt(
    *,
    root_dir: Path,
    sample_id: str,
    payload: Dict[str, object],
) -> Path:
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    out_path = root_dir / f"{sample_id}.pt"
    torch.save(payload, out_path)
    return out_path


def _safe_sample_id_for_filename(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_")[:200]


def _to_color_map(img_2d: torch.Tensor) -> "object":
    import cv2
    import numpy as np

    a = img_2d.detach().cpu().float().numpy()
    amin = float(a.min())
    amax = float(a.max())
    denom = max(1e-8, amax - amin)
    u8 = (((a - amin) / denom) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _field_chw_to_bgr_uint8(chw: torch.Tensor) -> "object":
    import cv2

    x = chw.detach().cpu().float()
    if x.dim() != 3:
        raise ValueError(f"expect [C,H,W], got {tuple(x.shape)}")
    c = int(x.shape[0])
    if c == 1:
        g = x[0]
        lo, hi = g.min(), g.max()
        g01 = (g - lo) / (hi - lo + 1e-8)
        rgb_hwc = (g01.unsqueeze(-1).expand(-1, -1, 3) * 255.0).clamp(0, 255).byte().numpy()
    elif c == 3:
        out = torch.zeros_like(x)
        for ci in range(3):
            ch = x[ci]
            lo, hi = ch.min(), ch.max()
            out[ci] = (ch - lo) / (hi - lo + 1e-8)
        rgb_hwc = (out * 255.0).clamp(0, 255).permute(1, 2, 0).byte().numpy()
    else:
        raise ValueError(f"expect C=1 or 3, got C={c}")
    return cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)


def _save_field_pred_vs_gt_mp4(
    pred_bvcthw: torch.Tensor,
    tgt_bvcthw: torch.Tensor,
    out_path: Path,
    *,
    view_idx: int = 0,
    fps: int = 8,
    color_mode: str = "auto",
) -> None:
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if int(pred_bvcthw.shape[0]) <= 0:
        return
    v = int(pred_bvcthw.shape[1])
    c = int(pred_bvcthw.shape[2])
    t = int(pred_bvcthw.shape[3])
    h = int(pred_bvcthw.shape[4])
    w = int(pred_bvcthw.shape[5])
    if t <= 0:
        return
    view_idx = max(0, min(int(view_idx), v - 1))
    cm = str(color_mode).strip().lower()
    if cm == "auto":
        cm = "rgb" if c == 3 else "jet"
    if cm not in ("rgb", "jet"):
        raise ValueError(f"unknown color_mode: {color_mode}")

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, int(fps))),
        (w * 2, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {out_path}")
    try:
        for ti in range(t):
            if cm == "rgb":
                pred_frame = _field_chw_to_bgr_uint8(pred_bvcthw[0, view_idx, :, ti, :, :])
                tgt_frame = _field_chw_to_bgr_uint8(tgt_bvcthw[0, view_idx, :, ti, :, :])
            else:
                pred_frame = _to_color_map(pred_bvcthw[0, view_idx, 0, ti, :, :])
                tgt_frame = _to_color_map(tgt_bvcthw[0, view_idx, 0, ti, :, :])
            writer.write(cv2.hconcat([pred_frame, tgt_frame]))
    finally:
        writer.release()


def export_field_videos(
    *,
    root_dir: Path,
    sample_id: str,
    stress_pred: torch.Tensor,
    stress_gt: torch.Tensor,
    flow_pred: torch.Tensor,
    flow_gt: torch.Tensor,
    force_pred: torch.Tensor,
    force_gt: torch.Tensor,
    view_idx: int = 0,
    fps: int = 8,
    color_mode: str = "auto",
) -> Dict[str, str]:
    root_dir = Path(root_dir)
    sample_dir = root_dir / _safe_sample_id_for_filename(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    stress_pred_cpu = stress_pred.detach().cpu().float().contiguous()
    flow_pred_cpu = flow_pred.detach().cpu().float().contiguous()
    force_pred_cpu = force_pred.detach().cpu().float().contiguous()
    stress_gt_cpu = align_field_to_prediction(stress_gt.detach().cpu().float(), stress_pred_cpu)
    flow_gt_cpu = align_field_to_prediction(flow_gt.detach().cpu().float(), flow_pred_cpu)
    force_gt_cpu = align_field_to_prediction(force_gt.detach().cpu().float(), force_pred_cpu)

    stress_path = sample_dir / "stress_pred_vs_gt.mp4"
    flow_path = sample_dir / "flow_pred_vs_gt.mp4"
    force_path = sample_dir / "force_pred_vs_gt.mp4"
    _save_field_pred_vs_gt_mp4(
        stress_pred_cpu, stress_gt_cpu, stress_path, view_idx=view_idx, fps=fps, color_mode=color_mode
    )
    _save_field_pred_vs_gt_mp4(
        flow_pred_cpu, flow_gt_cpu, flow_path, view_idx=view_idx, fps=fps, color_mode=color_mode
    )
    _save_field_pred_vs_gt_mp4(
        force_pred_cpu, force_gt_cpu, force_path, view_idx=view_idx, fps=fps, color_mode=color_mode
    )
    return {
        "sample_dir": str(sample_dir),
        "stress_video": str(stress_path),
        "flow_video": str(flow_path),
        "force_video": str(force_path),
    }
