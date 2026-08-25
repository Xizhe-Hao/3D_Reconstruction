from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

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
    from vlm_benchmark.run_vlm_field_benchmark import (
        _list_completed_sample_ids,
        _load_gt_fields,
        _read_all_sample_metric_rows,
        _resize_bvcthw,
        _save_pred_contact_sheet,
        _save_pred_video,
        _score_from_overall_rows,
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
    from run_vlm_field_benchmark import (
        _list_completed_sample_ids,
        _load_gt_fields,
        _read_all_sample_metric_rows,
        _resize_bvcthw,
        _save_pred_contact_sheet,
        _save_pred_video,
        _score_from_overall_rows,
    )
    from vlm_model_registry import VLM_REGISTRY, create_vlm_client

from eval_abalation.field_export import export_field_tensors_pt
from eval_abalation.metrics_field import _decode_stress_heat, aggregate_field_records, align_field_to_prediction, build_field_sample_record
from eval_abalation.report_utils import write_csv, write_json, write_jsonl
from eval_abalation.dataset_meta import load_sample_meta


def _format_user_prompt(
    template: str,
    *,
    sample_id: str,
    view_name: str,
    num_frames: int,
    frame_h: int,
    frame_w: int,
) -> str:
    return template.format(
        sample_id=str(sample_id),
        view_name=str(view_name),
        num_frames=int(num_frames),
        frame_h=int(frame_h),
        frame_w=int(frame_w),
        sheet_h=int(frame_h),
        sheet_w=int(frame_w) * int(num_frames),
    )


def _image_sheet_to_stress_bvcthw(
    image_path: Path,
    *,
    num_frames: int,
    target_num_frames: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    import cv2

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取生成图片: {image_path}")

    sheet_w = int(target_w) * int(num_frames)
    sheet_h = int(target_h)
    if int(bgr.shape[1]) != sheet_w or int(bgr.shape[0]) != sheet_h:
        bgr = cv2.resize(bgr, (sheet_w, sheet_h), interpolation=cv2.INTER_AREA)

    panels = []
    for ti in range(int(num_frames)):
        x0 = int(round(ti * sheet_w / float(num_frames)))
        x1 = int(round((ti + 1) * sheet_w / float(num_frames)))
        panel = bgr[:, x0:x1, :]
        if panel.shape[1] != int(target_w) or panel.shape[0] != int(target_h):
            panel = cv2.resize(panel, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        panels.append(rgb)

    arr = np.stack(panels, axis=0)
    x = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).unsqueeze(0).contiguous()
    return _resize_bvcthw(
        x,
        target_t=int(target_num_frames),
        target_h=int(target_h),
        target_w=int(target_w),
        mode="trilinear",
    ).clamp(0.0, 1.0)


def _export_stress_prediction_visuals(
    *,
    root_dir: Path,
    sample_id: str,
    stress_pred: torch.Tensor,
    view_idx: int = 0,
    fps: int = 8,
) -> Dict[str, str]:
    sample_dir = Path(root_dir) / str(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    stress_video = sample_dir / "stress_pred.mp4"
    stress_sheet = sample_dir / "stress_pred_sheet.png"
    stress_cpu = stress_pred.detach().cpu().float().contiguous()
    _save_pred_video(stress_cpu, stress_video, view_idx=view_idx, fps=fps, color_mode="rgb")
    _save_pred_contact_sheet(stress_cpu, stress_sheet, view_idx=view_idx, color_mode="rgb")
    return {
        "sample_dir": str(sample_dir),
        "stress_pred_video": str(stress_video),
        "stress_pred_sheet": str(stress_sheet),
    }


def _build_stress_export_payload(
    *,
    sample_id: str,
    stress_pred: torch.Tensor,
    stress_gt: torch.Tensor,
    object_mask: torch.Tensor | None,
    meta: Dict[str, object],
) -> Dict[str, object]:
    stress_pred_cpu = stress_pred.detach().cpu().float().contiguous()
    stress_gt_cpu = align_field_to_prediction(stress_gt.detach().cpu().float(), stress_pred_cpu)
    object_mask_cpu = None if object_mask is None else object_mask.detach().cpu().float().contiguous()
    meta_out = dict(meta)
    meta_out.setdefault("field_metric_version", "rgb_semantic_v2_flow_uv")
    meta_out.setdefault("object_mask_policy", "all_ones_when_missing")
    meta_out.setdefault("stress_visual_encoding", "jet_like_palette_nearest_or_gray_fallback")
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
        "stress_heat_pred": _decode_stress_heat(stress_pred_cpu),
        "stress_heat_gt": _decode_stress_heat(stress_gt_cpu),
        "object_mask": object_mask_cpu,
        "meta": meta_out,
    }


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


def run_stress_image_benchmark(
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
    image_num_frames: int,
    target_num_frames: int,
    target_h: int,
    target_w: int,
    request_interval_s: float,
    resume_from_output_root: Optional[Path],
) -> Dict[str, Any]:
    script_dir = physgaussian_root / "vlm_benchmark"
    system_prompt = _load_prompt(script_dir / "prompts" / "system_prompt_stress_image.txt")
    user_prompt_template = _load_prompt(script_dir / "prompts" / "user_prompt_video_stress_image.txt")

    tag_chain = _parse_tag_chain(vlm_tag, fallback_vlm_tags)
    active_tag_idx = 0
    active_tag = tag_chain[active_tag_idx]
    client = create_vlm_client(active_tag)

    output_root.mkdir(parents=True, exist_ok=True)
    _bootstrap_output_root(output_root, resume_from_output_root)

    reasoning_dir = output_root / "reasoning"
    predictions_dir = output_root / "predictions"
    generated_image_dir = output_root / "generated_images"
    sample_metrics_dir = output_root / "sample_metrics"
    field_export_dir = output_root / "field_exports"
    field_metrics_dir = output_root / "field_metrics"
    pred_visual_dir = output_root / "pred_visuals"
    for d in (reasoning_dir, predictions_dir, generated_image_dir, sample_metrics_dir, field_export_dir, field_metrics_dir, pred_visual_dir):
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
    num_attempted = 0

    for action, run_dir, _params in all_runs:
        if max_videos is not None and int(max_videos) > 0 and (existing_completed + num_attempted) >= int(max_videos):
            break
        sample_id = str(run_dir.name)
        if sample_id in completed_sample_ids:
            continue
        num_attempted += 1

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
            num_frames=image_num_frames,
            frame_h=target_h,
            frame_w=target_w,
        )

        prediction: Optional[Dict[str, Any]] = None
        used_tag = active_tag
        generated_image_path = generated_image_dir / f"{sample_id}.png"
        if request_interval_s > 0 and num_attempted > 1:
            time.sleep(float(request_interval_s))
        while True:
            try:
                prediction = client.generate_stress_image(
                    str(video_path),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_path=str(generated_image_path),
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
                print(f"[WARN] Qwen 图片生成失败，跳过样本 {sample_id}: {exc}")
                prediction = None
                break
        if prediction is None:
            continue

        (reasoning_dir / f"{sample_id}.txt").write_text(
            json.dumps(prediction, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        try:
            gt = _load_gt_fields(run_dir, num_frames=target_num_frames, target_h=target_h, target_w=target_w)
            stress_pred = _image_sheet_to_stress_bvcthw(
                generated_image_path,
                num_frames=image_num_frames,
                target_num_frames=target_num_frames,
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
                )
            ]
        except Exception as exc:
            print(f"[WARN] 解析生成图片或评估失败，跳过样本 {sample_id}: {exc}")
            (reasoning_dir / f"{sample_id}.txt").write_text(
                json.dumps(prediction, ensure_ascii=False, indent=2) + f"\n\n[PARSE_OR_EVAL_ERROR] {exc}\n",
                encoding="utf-8",
            )
            continue

        used_vlm_model = str(VLM_REGISTRY[used_tag].model)
        pred_visuals = _export_stress_prediction_visuals(
            root_dir=pred_visual_dir,
            sample_id=sample_id,
            stress_pred=stress_pred,
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
        pred_row["generated_image_path"] = str(generated_image_path)
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
                "generated_image_path": str(generated_image_path),
                "pred_visuals": pred_visuals,
                "field_records": field_records,
            },
        )
        export_field_tensors_pt(
            root_dir=field_export_dir,
            sample_id=sample_id,
            payload=_build_stress_export_payload(
                sample_id=sample_id,
                stress_pred=stress_pred,
                stress_gt=gt["stress"],
                object_mask=object_mask,
                meta={
                    "predictor": "vlm_stress_image",
                    "used_vlm_tag": used_tag,
                    "used_vlm_model": used_vlm_model,
                    "image_num_frames": int(image_num_frames),
                    "target_num_frames": int(target_num_frames),
                    "target_h": int(target_h),
                    "target_w": int(target_w),
                    "view_name": str(view_name),
                    "video_rel_path": str(video_rel_path),
                    "generated_image_path": str(generated_image_path),
                },
            ),
        )
        completed_sample_ids.add(sample_id)
        chosen_sample_ids.append(sample_id)
        used_vlm_tags.append(used_tag)
        num_processed += 1
        print(f"[OK] 已处理 stress image 样本 {num_processed}: {sample_id} [{used_tag}]")

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
        "benchmark_type": "vlm_stress_image",
        "physgaussian_root": str(physgaussian_root),
        "vlm_tag": str(vlm_tag),
        "fallback_vlm_tags": list(fallback_vlm_tags or []),
        "vlm_tag_chain": tag_chain,
        "split_json": str(split_json.resolve()) if split_json is not None else "",
        "eval_split": str(eval_split),
        "max_videos": int(max_videos or 0),
        "random_sample": bool(random_sample),
        "seed": int(seed),
        "image_num_frames": int(image_num_frames),
        "target_num_frames": int(target_num_frames),
        "target_h": int(target_h),
        "target_w": int(target_w),
        "field_metric_version": "rgb_semantic_v2_flow_uv",
        "field_metrics_masked_by_object": True,
        "existing_completed": int(existing_completed),
        "newly_attempted": int(num_attempted),
        "newly_completed": int(num_processed),
        "num_samples_used": int(len(completed_sample_ids)),
        "chosen_sample_ids": chosen_sample_ids,
        "used_vlm_tags": used_vlm_tags,
        "predictions_dir": str(predictions_dir),
        "generated_image_dir": str(generated_image_dir),
        "sample_metrics_dir": str(sample_metrics_dir),
        "field_metrics_dir": str(field_metrics_dir),
        "field_export_dir": str(field_export_dir),
        "pred_visual_dir": str(pred_visual_dir),
    }
    write_json(output_root / "benchmark_meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen image-generation stress field benchmark.")
    parser.add_argument(
        "--physgaussian_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--vlm_tag",
        type=str,
        default="qwen-image-2.0-pro",
        choices=list(VLM_REGISTRY.keys()),
    )
    parser.add_argument("--output_tag", type=str, default="qwen_stress_image_100")
    parser.add_argument("--resume_from_output_tag", type=str, default="")
    parser.add_argument("--fallback_vlm_tags", type=str, default="")
    parser.add_argument("--max_videos", type=int, default=100)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--split_json", type=str, default="")
    parser.add_argument("--eval_split", type=str, default="test", choices=("train", "test"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_num_frames", type=int, default=64)
    parser.add_argument("--target_num_frames", type=int, default=64)
    parser.add_argument("--target_h", type=int, default=112)
    parser.add_argument("--target_w", type=int, default=112)
    parser.add_argument("--request_interval_s", type=float, default=5.0)
    args = parser.parse_args()

    phys_root = Path(args.physgaussian_root).resolve()
    output_tag = str(args.output_tag).strip() or "qwen_stress_image_100"
    output_root = phys_root / "vlm_benchmark" / "output" / output_tag
    split_json = (
        Path(args.split_json).expanduser().resolve()
        if str(args.split_json).strip()
        else (phys_root / "auto_output" / "dataset_5000" / "train_test_split.cleaned.json")
    )
    run_stress_image_benchmark(
        physgaussian_root=phys_root,
        output_root=output_root,
        vlm_tag=args.vlm_tag,
        fallback_vlm_tags=[x.strip() for x in str(args.fallback_vlm_tags).split(",") if x.strip()],
        max_videos=args.max_videos,
        random_sample=args.random_sample,
        split_json=split_json,
        eval_split=args.eval_split,
        seed=args.seed,
        image_num_frames=args.image_num_frames,
        target_num_frames=args.target_num_frames,
        target_h=args.target_h,
        target_w=args.target_w,
        request_interval_s=args.request_interval_s,
        resume_from_output_root=(
            phys_root / "vlm_benchmark" / "output" / str(args.resume_from_output_tag).strip()
            if str(args.resume_from_output_tag).strip()
            else None
        ),
    )


if __name__ == "__main__":
    main()
