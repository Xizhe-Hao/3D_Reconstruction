from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_THIS_FILE = Path(__file__).resolve()
_PHYS_ROOT = _THIS_FILE.parents[1]
if str(_PHYS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHYS_ROOT))

try:
    from vlm_benchmark.run_vlm_benchmark import (
        _bootstrap_output_root,
        _collect_all_runs,
        _collect_split_samples,
        _load_prompt,
        _parse_tag_chain,
        _prepare_sample_video,
        _should_rotate_model,
    )
    from vlm_benchmark.vlm_model_registry import VLM_REGISTRY, create_vlm_client
except ModuleNotFoundError:
    from run_vlm_benchmark import (
        _bootstrap_output_root,
        _collect_all_runs,
        _collect_split_samples,
        _load_prompt,
        _parse_tag_chain,
        _prepare_sample_video,
        _should_rotate_model,
    )
    from vlm_model_registry import VLM_REGISTRY, create_vlm_client

from eval_abalation.field_export import (
    _field_chw_to_bgr_uint8,
    _to_color_map,
    build_field_export_payload,
    export_field_tensors_pt,
)
from eval_abalation.metrics_field import aggregate_field_records, build_field_sample_record
from eval_abalation.report_utils import write_csv, write_json, write_jsonl
from eval_abalation.dataset_meta import load_sample_meta
from logic_model.dataset import _coerce_force_mask_layout
from my_utils.sample_pack import read_sample_pack_arrays


def _list_completed_sample_ids(sample_metrics_dir: Path) -> set[str]:
    if not sample_metrics_dir.is_dir():
        return set()
    return {p.stem for p in sample_metrics_dir.glob("*.json") if p.is_file()}


def _read_all_sample_metric_rows(sample_metrics_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not sample_metrics_dir.is_dir():
        return rows
    for path in sorted(sample_metrics_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        field_records = raw.get("field_records")
        if isinstance(field_records, list):
            for row in field_records:
                if isinstance(row, dict):
                    rows.append(dict(row))
    return rows


def _normalize_scalar_grid(
    value: Any,
    *,
    field_name: str,
    grid_size: int,
    num_frames: int,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"{field_name}.frames 维度错误，期望 3D，实际 {arr.shape}")
    x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    if tuple(arr.shape) != (num_frames, grid_size, grid_size):
        x = F.interpolate(
            x,
            size=(int(num_frames), int(grid_size), int(grid_size)),
            mode="trilinear",
            align_corners=False,
        )
    out = x.squeeze(0).squeeze(0).clamp(0.0, 9.0)
    out = torch.round(out) / 9.0
    return out.cpu().numpy().astype(np.float32, copy=False)


def _extract_field_grids(
    prediction: Dict[str, Any],
    *,
    grid_size: int,
    num_frames: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for field_name in ("force_mask", "flow", "stress"):
        block = prediction.get(field_name)
        if not isinstance(block, dict):
            raise ValueError(f"模型输出缺少对象字段: {field_name}")
        out[field_name] = _normalize_scalar_grid(
            block.get("frames"),
            field_name=field_name,
            grid_size=grid_size,
            num_frames=num_frames,
        )
    return out


def _resize_bvcthw(
    x: torch.Tensor,
    *,
    target_t: int,
    target_h: int,
    target_w: int,
    mode: str,
) -> torch.Tensor:
    b, v, c, t, h, w = x.shape
    if (int(t), int(h), int(w)) == (int(target_t), int(target_h), int(target_w)):
        return x.contiguous()
    y = F.interpolate(
        x.view(b * v, c, t, h, w).float(),
        size=(int(target_t), int(target_h), int(target_w)),
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )
    return y.view(b, v, c, int(target_t), int(target_h), int(target_w)).contiguous()


def _scalar_grid_to_bvcthw_rgb(
    arr_txy: np.ndarray,
    *,
    target_t: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(arr_txy, dtype=np.float32))
    x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 3, 1, 1, 1)
    return _resize_bvcthw(x, target_t=target_t, target_h=target_h, target_w=target_w, mode="trilinear").clamp(0.0, 1.0)


def _scalar_grid_to_bvcthw_mask(
    arr_txy: np.ndarray,
    *,
    target_t: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(arr_txy, dtype=np.float32)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return _resize_bvcthw(x, target_t=target_t, target_h=target_h, target_w=target_w, mode="trilinear").clamp(0.0, 1.0)


def _load_gt_fields(
    sample_dir: Path,
    *,
    num_frames: int,
    target_h: int,
    target_w: int,
) -> Dict[str, torch.Tensor]:
    arrays = read_sample_pack_arrays(sample_dir, num_frames=num_frames, max_views=1)
    object_mask_np = np.asarray(arrays["object_mask"][:1], dtype=np.float32)
    force_np = _coerce_force_mask_layout(
        np.asarray(arrays["force_mask"][:1], dtype=np.float32),
        object_mask=object_mask_np,
        single_channel_binary=True,
        binary_threshold=0.5,
    ).astype(np.float32, copy=False)
    out = {
        "stress": torch.from_numpy(np.asarray(arrays["stress"][:1], dtype=np.float32)).unsqueeze(0).contiguous(),
        "flow": torch.from_numpy(np.asarray(arrays["flow"][:1], dtype=np.float32)).unsqueeze(0).contiguous(),
        "force_mask": torch.from_numpy(force_np).unsqueeze(0).contiguous(),
        "object_mask": torch.from_numpy(object_mask_np).unsqueeze(0).contiguous(),
    }
    out["stress"] = _resize_bvcthw(out["stress"], target_t=num_frames, target_h=target_h, target_w=target_w, mode="trilinear")
    out["flow"] = _resize_bvcthw(out["flow"], target_t=num_frames, target_h=target_h, target_w=target_w, mode="trilinear")
    out["force_mask"] = _resize_bvcthw(out["force_mask"], target_t=num_frames, target_h=target_h, target_w=target_w, mode="nearest").clamp(0.0, 1.0)
    out["object_mask"] = _resize_bvcthw(out["object_mask"], target_t=num_frames, target_h=target_h, target_w=target_w, mode="nearest").clamp(0.0, 1.0)
    return out


def _format_user_prompt(
    template: str,
    *,
    sample_id: str,
    view_name: str,
    grid_size: int,
    num_frames: int,
) -> str:
    return template.format(
        sample_id=str(sample_id),
        view_name=str(view_name),
        grid_size=int(grid_size),
        num_frames=int(num_frames),
    )


def _sample_prediction_row(
    *,
    sample_id: str,
    view_name: str,
    action: str,
    material: str,
    object_name: str,
    source_sample_dir: Path,
    video_rel_path: str,
    used_vlm_tag: str,
    used_vlm_model: str,
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "sample_id": str(sample_id),
        "view_name": str(view_name),
        "action": str(action),
        "material": str(material),
        "object_name": str(object_name),
        "source_sample_dir": source_sample_dir.resolve().as_posix(),
        "video_rel_path": str(video_rel_path),
        "used_vlm_tag": str(used_vlm_tag),
        "used_vlm_model": str(used_vlm_model),
        "prediction": prediction,
    }


def _score_from_overall_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in rows:
        field_name = str(row.get("field_name", "")).strip()
        if not field_name:
            continue
        out[field_name] = {
            key: row.get(key)
            for key in (
                "count",
                "mse",
                "gradient_l1",
                "ssim",
                "region_recall",
                "soft_region_iou",
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
        }
    return out


def _pred_frame_to_bgr(pred_bvcthw: torch.Tensor, *, view_idx: int, frame_idx: int, color_mode: str) -> np.ndarray:
    c = int(pred_bvcthw.shape[2])
    cm = str(color_mode).strip().lower()
    if cm == "auto":
        cm = "rgb" if c == 3 else "jet"
    if cm == "rgb":
        return _field_chw_to_bgr_uint8(pred_bvcthw[0, view_idx, :, frame_idx, :, :])
    return _to_color_map(pred_bvcthw[0, view_idx, 0, frame_idx, :, :])


def _save_pred_video(
    pred_bvcthw: torch.Tensor,
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
    t = int(pred_bvcthw.shape[3])
    h = int(pred_bvcthw.shape[4])
    w = int(pred_bvcthw.shape[5])
    if t <= 0:
        return
    view_idx = max(0, min(int(view_idx), v - 1))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, int(fps))),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {out_path}")
    try:
        for ti in range(t):
            writer.write(_pred_frame_to_bgr(pred_bvcthw, view_idx=view_idx, frame_idx=ti, color_mode=color_mode))
    finally:
        writer.release()


def _save_pred_contact_sheet(
    pred_bvcthw: torch.Tensor,
    out_path: Path,
    *,
    view_idx: int = 0,
    color_mode: str = "auto",
    cols: int = 4,
) -> None:
    import cv2
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if int(pred_bvcthw.shape[0]) <= 0:
        return
    v = int(pred_bvcthw.shape[1])
    t = int(pred_bvcthw.shape[3])
    h = int(pred_bvcthw.shape[4])
    w = int(pred_bvcthw.shape[5])
    if t <= 0:
        return
    view_idx = max(0, min(int(view_idx), v - 1))
    frames = [_pred_frame_to_bgr(pred_bvcthw, view_idx=view_idx, frame_idx=ti, color_mode=color_mode) for ti in range(t)]
    cols = max(1, min(int(cols), len(frames)))
    rows = []
    for start in range(0, len(frames), cols):
        row = frames[start : start + cols]
        if len(row) < cols:
            row.extend([np.zeros((h, w, 3), dtype=np.uint8) for _ in range(cols - len(row))])
        rows.append(cv2.hconcat(row))
    sheet = rows[0] if len(rows) == 1 else cv2.vconcat(rows)
    cv2.imwrite(str(out_path), sheet)


def _export_vlm_prediction_visuals(
    *,
    root_dir: Path,
    sample_id: str,
    stress_pred: torch.Tensor,
    flow_pred: torch.Tensor,
    force_pred: torch.Tensor,
    view_idx: int = 0,
    fps: int = 8,
) -> Dict[str, str]:
    sample_dir = Path(root_dir) / str(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    stress_video = sample_dir / "stress_pred.mp4"
    flow_video = sample_dir / "flow_pred.mp4"
    force_video = sample_dir / "force_pred.mp4"
    stress_sheet = sample_dir / "stress_pred_sheet.png"
    flow_sheet = sample_dir / "flow_pred_sheet.png"
    force_sheet = sample_dir / "force_pred_sheet.png"

    _save_pred_video(stress_pred.detach().cpu().float().contiguous(), stress_video, view_idx=view_idx, fps=fps, color_mode="rgb")
    _save_pred_video(flow_pred.detach().cpu().float().contiguous(), flow_video, view_idx=view_idx, fps=fps, color_mode="rgb")
    _save_pred_video(force_pred.detach().cpu().float().contiguous(), force_video, view_idx=view_idx, fps=fps, color_mode="jet")
    _save_pred_contact_sheet(stress_pred.detach().cpu().float().contiguous(), stress_sheet, view_idx=view_idx, color_mode="rgb")
    _save_pred_contact_sheet(flow_pred.detach().cpu().float().contiguous(), flow_sheet, view_idx=view_idx, color_mode="rgb")
    _save_pred_contact_sheet(force_pred.detach().cpu().float().contiguous(), force_sheet, view_idx=view_idx, color_mode="jet")
    return {
        "sample_dir": str(sample_dir),
        "stress_pred_video": str(stress_video),
        "flow_pred_video": str(flow_video),
        "force_pred_video": str(force_video),
        "stress_pred_sheet": str(stress_sheet),
        "flow_pred_sheet": str(flow_sheet),
        "force_pred_sheet": str(force_sheet),
    }


def run_field_benchmark(
    *,
    physgaussian_root: Path,
    output_root: Path,
    vlm_tag: str,
    fallback_vlm_tags: Optional[List[str]],
    max_videos: Optional[int],
    random_sample: bool,
    split_json: Optional[Path],
    eval_split: str,
    seed: int,
    grid_size: int,
    num_frames: int,
    target_num_frames: int,
    target_h: int,
    target_w: int,
    resume_from_output_root: Optional[Path],
) -> Dict[str, Any]:
    script_dir = physgaussian_root / "vlm_benchmark"
    system_prompt = _load_prompt(script_dir / "prompts" / "system_prompt_field_grid.txt")
    user_prompt_template = _load_prompt(script_dir / "prompts" / "user_prompt_video_field_grid.txt")

    tag_chain = _parse_tag_chain(vlm_tag, fallback_vlm_tags)
    active_tag_idx = 0
    active_tag = tag_chain[active_tag_idx]
    client = create_vlm_client(active_tag)

    output_root.mkdir(parents=True, exist_ok=True)
    _bootstrap_output_root(output_root, resume_from_output_root)

    reasoning_dir = output_root / "reasoning"
    predictions_dir = output_root / "predictions"
    sample_metrics_dir = output_root / "sample_metrics"
    field_export_dir = output_root / "field_exports"
    field_metrics_dir = output_root / "field_metrics"
    pred_visual_dir = output_root / "pred_visuals"
    for d in (reasoning_dir, predictions_dir, sample_metrics_dir, field_export_dir, field_metrics_dir, pred_visual_dir):
        d.mkdir(parents=True, exist_ok=True)

    completed_sample_ids = _list_completed_sample_ids(sample_metrics_dir)
    existing_completed = len(completed_sample_ids)

    if split_json is not None:
        all_runs = _collect_split_samples(
            physgaussian_root,
            split_json=split_json,
            eval_split=eval_split,
            max_samples=None,
            random_sample=random_sample,
            seed=seed,
        )
    else:
        all_runs = _collect_all_runs(physgaussian_root / "auto_output")
        if random_sample:
            rng = np.random.default_rng(int(seed))
            idx = np.arange(len(all_runs))
            rng.shuffle(idx)
            all_runs = [all_runs[int(i)] for i in idx]

    chosen_sample_ids: List[str] = sorted(completed_sample_ids)
    used_vlm_tags: List[str] = []
    num_processed = 0

    for action, run_dir, _params in all_runs:
        if max_videos is not None and int(max_videos) > 0 and (existing_completed + num_processed) >= int(max_videos):
            break
        sample_id = str(run_dir.name)
        if sample_id in completed_sample_ids:
            continue

        sample_meta = load_sample_meta(run_dir)
        video_path, view_name = _prepare_sample_video(
            run_dir,
            output_root=output_root,
            sample_id=sample_id,
            max_views=1,
        )
        video_rel_path = video_path.relative_to(physgaussian_root).as_posix()
        user_prompt = _format_user_prompt(
            user_prompt_template,
            sample_id=sample_id,
            view_name=view_name,
            grid_size=grid_size,
            num_frames=num_frames,
        )

        prediction: Optional[Dict[str, Any]] = None
        used_tag = active_tag
        while True:
            try:
                prediction = client.predict_video(
                    str(video_path),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    video_id=sample_id,
                )
                used_tag = active_tag
                break
            except Exception as exc:
                if _should_rotate_model(exc) and active_tag_idx + 1 < len(tag_chain):
                    failed_tag = active_tag
                    active_tag_idx += 1
                    active_tag = tag_chain[active_tag_idx]
                    print(
                        f"[WARN] VLM 模型 {failed_tag} 可能已到额度/限流，"
                        f"切换到 {active_tag} 并重试样本 {sample_id}: {exc}"
                    )
                    client = create_vlm_client(active_tag)
                    continue
                (reasoning_dir / f"{sample_id}.txt").write_text(
                    f"[ERROR][{active_tag}] {exc}\n",
                    encoding="utf-8",
                )
                print(f"[WARN] VLM 调用失败，跳过样本 {sample_id}: {exc}")
                prediction = None
                break
        if prediction is None:
            continue

        raw_text = str(prediction.get("__raw_text__", "") or "")
        (reasoning_dir / f"{sample_id}.txt").write_text(raw_text, encoding="utf-8")

        try:
            pred_grids = _extract_field_grids(
                prediction,
                grid_size=grid_size,
                num_frames=num_frames,
            )
            gt = _load_gt_fields(
                run_dir,
                num_frames=target_num_frames,
                target_h=target_h,
                target_w=target_w,
            )
            stress_pred = _scalar_grid_to_bvcthw_rgb(
                pred_grids["stress"],
                target_t=target_num_frames,
                target_h=target_h,
                target_w=target_w,
            )
            flow_pred = _scalar_grid_to_bvcthw_rgb(
                pred_grids["flow"],
                target_t=target_num_frames,
                target_h=target_h,
                target_w=target_w,
            )
            force_pred = _scalar_grid_to_bvcthw_mask(
                pred_grids["force_mask"],
                target_t=target_num_frames,
                target_h=target_h,
                target_w=target_w,
            )
            object_mask = gt["object_mask"]

            field_records = [
                build_field_sample_record(
                    sample_id=sample_id,
                    action=sample_meta.action or action,
                    material=sample_meta.material_group,
                    object_name=sample_meta.object_name,
                    field_name="stress",
                    pred_bvcthw=stress_pred,
                    gt_bvcthw=gt["stress"],
                    object_mask_bvcthw=object_mask,
                ),
                build_field_sample_record(
                    sample_id=sample_id,
                    action=sample_meta.action or action,
                    material=sample_meta.material_group,
                    object_name=sample_meta.object_name,
                    field_name="flow",
                    pred_bvcthw=flow_pred,
                    gt_bvcthw=gt["flow"],
                    object_mask_bvcthw=object_mask,
                ),
                build_field_sample_record(
                    sample_id=sample_id,
                    action=sample_meta.action or action,
                    material=sample_meta.material_group,
                    object_name=sample_meta.object_name,
                    field_name="force_mask",
                    pred_bvcthw=force_pred,
                    gt_bvcthw=gt["force_mask"],
                    object_mask_bvcthw=object_mask,
                ),
            ]
        except Exception as exc:
            print(f"[WARN] 解析或评估失败，跳过样本 {sample_id}: {exc}")
            (reasoning_dir / f"{sample_id}.txt").write_text(
                raw_text + f"\n\n[PARSE_OR_EVAL_ERROR] {exc}\n",
                encoding="utf-8",
            )
            continue

        used_vlm_model = str(VLM_REGISTRY[used_tag].model)
        pred_visuals = _export_vlm_prediction_visuals(
            root_dir=pred_visual_dir,
            sample_id=sample_id,
            stress_pred=stress_pred,
            flow_pred=flow_pred,
            force_pred=force_pred,
            view_idx=0,
            fps=8,
        )
        pred_row = _sample_prediction_row(
                sample_id=sample_id,
                view_name=view_name,
                action=sample_meta.action or action,
                material=sample_meta.material_group,
                object_name=sample_meta.object_name,
                source_sample_dir=run_dir,
                video_rel_path=video_rel_path,
                used_vlm_tag=used_tag,
                used_vlm_model=used_vlm_model,
                prediction=prediction,
            )
        pred_row["pred_visuals"] = pred_visuals
        write_json(predictions_dir / f"{sample_id}.json", pred_row)
        write_json(
            sample_metrics_dir / f"{sample_id}.json",
            {
                "sample_id": sample_id,
                "view_name": view_name,
                "action": sample_meta.action or action,
                "material": sample_meta.material_group,
                "object_name": sample_meta.object_name,
                "video_rel_path": video_rel_path,
                "source_sample_dir": run_dir.resolve().as_posix(),
                "used_vlm_tag": used_tag,
                "used_vlm_model": used_vlm_model,
                "pred_visuals": pred_visuals,
                "field_records": field_records,
            },
        )
        export_field_tensors_pt(
            root_dir=field_export_dir,
            sample_id=sample_id,
            payload=build_field_export_payload(
                sample_id=sample_id,
                stress_pred=stress_pred,
                stress_gt=gt["stress"],
                flow_pred=flow_pred,
                flow_gt=gt["flow"],
                force_pred=force_pred,
                force_gt=gt["force_mask"],
                object_mask=object_mask,
                meta={
                    "predictor": "vlm_field_grid",
                    "used_vlm_tag": used_tag,
                    "used_vlm_model": used_vlm_model,
                    "grid_size": int(grid_size),
                    "grid_num_frames": int(num_frames),
                    "target_num_frames": int(target_num_frames),
                    "target_h": int(target_h),
                    "target_w": int(target_w),
                    "view_name": str(view_name),
                    "video_rel_path": str(video_rel_path),
                },
            ),
        )
        completed_sample_ids.add(sample_id)
        chosen_sample_ids.append(sample_id)
        used_vlm_tags.append(used_tag)
        num_processed += 1
        print(f"[OK] 已处理样本 {num_processed}: {sample_id} [{used_tag}]")

    sample_records = _read_all_sample_metric_rows(sample_metrics_dir)
    overall_rows = aggregate_field_records(sample_records, group_fields=())
    by_action_rows = aggregate_field_records(sample_records, group_fields=("action",))
    by_material_rows = aggregate_field_records(sample_records, group_fields=("material",))

    write_jsonl(field_metrics_dir / "sample_records.jsonl", sample_records)
    write_csv(field_metrics_dir / "overall.csv", overall_rows)
    write_csv(field_metrics_dir / "by_action.csv", by_action_rows)
    write_csv(field_metrics_dir / "by_material.csv", by_material_rows)
    write_json(output_root / "score.json", _score_from_overall_rows(overall_rows))

    meta = {
        "benchmark_type": "vlm_field_grid",
        "physgaussian_root": str(physgaussian_root),
        "vlm_tag": str(vlm_tag),
        "fallback_vlm_tags": list(fallback_vlm_tags or []),
        "vlm_tag_chain": tag_chain,
        "split_json": str(split_json.resolve()) if split_json is not None else "",
        "eval_split": str(eval_split),
        "max_videos": int(max_videos or 0),
        "random_sample": bool(random_sample),
        "seed": int(seed),
        "grid_size": int(grid_size),
        "grid_num_frames": int(num_frames),
        "target_num_frames": int(target_num_frames),
        "target_h": int(target_h),
        "target_w": int(target_w),
        "field_metric_version": "rgb_semantic_v3_flow_uv_quality",
        "field_metrics_masked_by_object": True,
        "existing_completed": int(existing_completed),
        "newly_completed": int(num_processed),
        "num_samples_used": int(len(completed_sample_ids)),
        "chosen_sample_ids": chosen_sample_ids,
        "used_vlm_tags": used_vlm_tags,
        "predictions_dir": str(predictions_dir),
        "sample_metrics_dir": str(sample_metrics_dir),
        "field_metrics_dir": str(field_metrics_dir),
        "field_export_dir": str(field_export_dir),
        "pred_visual_dir": str(pred_visual_dir),
    }
    write_json(output_root / "benchmark_meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM dense field benchmark for force/flow/stress.")
    parser.add_argument(
        "--physgaussian_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--vlm_tag",
        type=str,
        default="qwen3.5-plus",
        choices=list(VLM_REGISTRY.keys()),
    )
    parser.add_argument("--output_tag", type=str, default="qwen_field_100")
    parser.add_argument("--resume_from_output_tag", type=str, default="")
    parser.add_argument("--fallback_vlm_tags", type=str, default="")
    parser.add_argument("--max_videos", type=int, default=100)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--split_json", type=str, default="")
    parser.add_argument("--eval_split", type=str, default="test", choices=("train", "test"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--target_num_frames", type=int, default=64)
    parser.add_argument("--target_h", type=int, default=112)
    parser.add_argument("--target_w", type=int, default=112)
    args = parser.parse_args()

    phys_root = Path(args.physgaussian_root).resolve()
    output_tag = str(args.output_tag).strip() or "qwen_field_100"
    output_root = phys_root / "vlm_benchmark" / "output" / output_tag
    split_json = (
        Path(args.split_json).expanduser().resolve()
        if str(args.split_json).strip()
        else (phys_root / "auto_output" / "dataset_5000" / "train_test_split.cleaned.json")
    )
    run_field_benchmark(
        physgaussian_root=phys_root,
        output_root=output_root,
        vlm_tag=args.vlm_tag,
        fallback_vlm_tags=[x.strip() for x in str(args.fallback_vlm_tags).split(",") if x.strip()],
        max_videos=args.max_videos,
        random_sample=args.random_sample,
        split_json=split_json,
        eval_split=args.eval_split,
        seed=args.seed,
        grid_size=args.grid_size,
        num_frames=args.num_frames,
        target_num_frames=args.target_num_frames,
        target_h=args.target_h,
        target_w=args.target_w,
        resume_from_output_root=(
            phys_root / "vlm_benchmark" / "output" / str(args.resume_from_output_tag).strip()
            if str(args.resume_from_output_tag).strip()
            else None
        ),
    )


if __name__ == "__main__":
    main()
