import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    # 作为包运行: python -m vlm_benchmark.run_vlm_benchmark
    from vlm_benchmark.vlm_model_registry import create_vlm_client, VLM_REGISTRY  # type: ignore
except ModuleNotFoundError:
    # 作为脚本运行: python vlm_benchmark/run_vlm_benchmark.py
    from vlm_model_registry import create_vlm_client, VLM_REGISTRY  # type: ignore


ACTION_NAMES = ("bend", "drop", "press", "shear", "stretch")
MATERIAL_CATEGORIES = ("jelly", "metal", "plasticine")
VLM_INPUT_NUM_FRAMES = 64
RGB_VIEW_MP4_RE = re.compile(r"^az\d+_el\d+\.mp4$", re.IGNORECASE)
ROTATE_ERROR_HINTS = (
    "quota",
    "free quota",
    "insufficient_quota",
    "rate limit",
    "ratelimit",
    "429",
    "bill",
    "balance",
    "credit",
    "resource exhausted",
    "too many requests",
    "thrott",
    "限流",
    "额度",
    "配额",
    "余额",
    "欠费",
    "invalid model",
    "model not found",
    "unsupported model",
    "service unavailable",
    "模型不存在",
    "模型不可用",
)


def _ensure_phys_root_importable(physgaussian_root: Path) -> None:
    root = str(physgaussian_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_tag_chain(vlm_tag: str, fallback_vlm_tags: Optional[List[str]] = None) -> List[str]:
    chain: List[str] = []
    seen = set()
    for tag in [str(vlm_tag)] + list(fallback_vlm_tags or []):
        key = str(tag).strip()
        if not key or key in seen:
            continue
        if key not in VLM_REGISTRY:
            raise KeyError(f"未知的 VLM 标签: {key}，可选值: {list(VLM_REGISTRY.keys())}")
        chain.append(key)
        seen.add(key)
    if not chain:
        raise ValueError("VLM 标签链为空")
    return chain


def _should_rotate_model(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in ROTATE_ERROR_HINTS)


def _collect_all_runs(auto_output_root: Path) -> List[Tuple[str, Path, Dict[str, str]]]:
    """与 `my_model.dataset.iter_auto_output_runs` 一致：支持旧 `__` 目录名与数字目录 + gt_parameters.json。"""
    root = auto_output_root.resolve()
    pg_root = root.parent
    _ensure_phys_root_importable(pg_root)
    from my_model.dataset import iter_auto_output_runs

    return list(iter_auto_output_runs(auto_output_root))


def _select_sample_ids(
    sample_ids: List[str],
    *,
    max_samples: Optional[int],
    random_sample: bool,
    seed: int,
) -> List[str]:
    chosen = list(sample_ids)
    if random_sample:
        rng = random.Random(int(seed))
        rng.shuffle(chosen)
    if max_samples is not None and int(max_samples) > 0:
        chosen = chosen[: int(max_samples)]
    return chosen


def _collect_split_samples(
    physgaussian_root: Path,
    *,
    split_json: Path,
    eval_split: str,
    max_samples: Optional[int],
    random_sample: bool,
    seed: int,
) -> List[Tuple[str, Path, Dict[str, str]]]:
    _ensure_phys_root_importable(physgaussian_root)
    from eval_abalation.split_utils import load_split_info, pick_eval_ids

    split_info = load_split_info(split_json)
    all_ids = pick_eval_ids(split_info, eval_split)
    chosen_ids = _select_sample_ids(
        [str(x) for x in all_ids],
        max_samples=max_samples,
        random_sample=random_sample,
        seed=seed,
    )
    rows: List[Tuple[str, Path, Dict[str, str]]] = []
    for sid in chosen_ids:
        sample_dir = split_info.split_root / str(sid)
        gt_path = sample_dir / "gt.json"
        if not sample_dir.is_dir() or not gt_path.is_file():
            continue
        with gt_path.open("r", encoding="utf-8") as f:
            gt = json.load(f)
        action = str(gt.get("action", "")).strip().lower()
        if action not in ACTION_NAMES:
            continue
        params_obj = gt.get("regression") or gt.get("params") or {}
        params: Dict[str, str] = {}
        if isinstance(params_obj, dict):
            params = {str(k): str(v) for k, v in params_obj.items()}
        material_obj = gt.get("params") or {}
        if isinstance(material_obj, dict) and "material" in material_obj and "material" not in params:
            params["material"] = str(material_obj.get("material"))
        rows.append((action, sample_dir, params))
    return rows


def _object_name_for_sample(run_dir: Path) -> str:
    gt_path = run_dir / "gt_parameters.json"
    if gt_path.is_file():
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt = json.load(f)
            return str(gt.get("ply_stem", run_dir.parent.name))
        except (OSError, json.JSONDecodeError):
            pass
    name = run_dir.name
    if "__" in name:
        return name.split("__")[0]
    return run_dir.parent.name


def _find_video_in_run_dir(run_dir: Path) -> Optional[Path]:
    """
    在单个仿真目录下查找视频文件。
    """
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    for root, _, files in os.walk(run_dir):
        for fname in sorted(files):
            if any(fname.lower().endswith(ext) for ext in exts):
                return Path(root) / fname
    return None


def _safe_float(params: Dict[str, str], key: str) -> float:
    v = params.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _normalize_material_label(params: Dict[str, str]) -> str:
    raw = params.get("material", "").lower()
    if raw in MATERIAL_CATEGORIES:
        return raw
    if "jelly" in raw:
        return "jelly"
    if "metal" in raw:
        return "metal"
    if "plastic" in raw:
        return "plasticine"
    return "plasticine"


def _load_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _dedupe_rows_by_sample_id(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped: Dict[str, Dict[str, str]] = {}
    ordered_ids: List[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if sample_id not in deduped:
            ordered_ids.append(sample_id)
        deduped[sample_id] = dict(row)
    return [deduped[sample_id] for sample_id in ordered_ids]


def _normalize_existing_pred_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item.setdefault("used_vlm_tag", "")
        item.setdefault("used_vlm_model", "")
        out.append(item)
    return out


def _bootstrap_output_root(output_root: Path, resume_from_output_root: Optional[Path]) -> None:
    if resume_from_output_root is None:
        return
    src_root = resume_from_output_root.resolve()
    dst_root = output_root.resolve()
    if src_root == dst_root or not src_root.is_dir():
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    for fname in ("vlm_benchmark_gt.csv", "vlm_benchmark_pred.csv", "benchmark_meta.json"):
        src = src_root / fname
        dst = dst_root / fname
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)

    src_reasoning = src_root / "reasoning"
    dst_reasoning = dst_root / "reasoning"
    if src_reasoning.is_dir():
        dst_reasoning.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(src_reasoning.glob("*.txt")):
            dst_file = dst_reasoning / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)


def _list_view_dirs(root: Path) -> List[str]:
    if not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _sorted_image_files(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name)


def _list_arch4_video_groups(videos_dir: Path) -> List[Tuple[str, Path, Path, Path, Path]]:
    if not videos_dir.is_dir():
        return []
    groups: Dict[str, Dict[str, Path]] = {}
    for p in videos_dir.glob("*.mp4"):
        stem = p.stem
        if stem.endswith("_stress_gaussian"):
            base = stem[: -len("_stress_gaussian")]
            groups.setdefault(base, {})["stress"] = p
        elif stem.endswith("_flow_gaussian"):
            base = stem[: -len("_flow_gaussian")]
            groups.setdefault(base, {})["flow"] = p
        elif stem.endswith("_force_mask"):
            base = stem[: -len("_force_mask")]
            groups.setdefault(base, {})["force"] = p
        else:
            groups.setdefault(stem, {})["rgb"] = p
    out: List[Tuple[str, Path, Path, Path, Path]] = []
    for base in sorted(groups.keys()):
        g = groups[base]
        if "rgb" in g and "stress" in g and "flow" in g and "force" in g:
            out.append((base, g["rgb"], g["stress"], g["flow"], g["force"]))
    return out


def _list_tensor_stems(tensor_root: Path) -> List[str]:
    if not tensor_root.is_dir():
        return []
    return sorted({p.stem for p in tensor_root.glob("*.pt") if p.is_file()})


def _sample_pack_rgb_views(sample_dir: Path, *, max_views: int) -> List[Tuple[str, np.ndarray]]:
    from my_utils.sample_pack import _load_sample_pack_npz, _sample_pack_path

    pack_path = _sample_pack_path(sample_dir)
    if not pack_path.is_file():
        return []
    meta, arrays = _load_sample_pack_npz(pack_path)
    rgb = np.asarray(arrays.get("rgb"), dtype=np.uint8)
    views = [str(v) for v in (meta.get("views") or [])]
    out: List[Tuple[str, np.ndarray]] = []
    for idx in range(min(int(rgb.shape[0]), int(max_views))):
        view_name = views[idx] if idx < len(views) else f"view{idx:02d}"
        frames_tchw = np.ascontiguousarray(rgb[idx])
        if frames_tchw.ndim != 4 or int(frames_tchw.shape[1]) != 3:
            continue
        frames_thwc = np.transpose(frames_tchw, (0, 2, 3, 1))
        out.append((view_name, np.ascontiguousarray(frames_thwc)))
    return out


def _lmdb_rgb_views(sample_dir: Path, *, max_views: int) -> List[Tuple[str, np.ndarray]]:
    from my_utils.sample_pack import _read_lmdb_uint8_arrays

    env_path = sample_dir / "arch4_data.lmdb"
    if not env_path.is_dir() or not (env_path / "data.mdb").is_file():
        return []
    arrays, meta = _read_lmdb_uint8_arrays(env_path, max_views=max_views)
    rgb = np.asarray(arrays.get("rgb"), dtype=np.uint8)
    views = [str(v) for v in (meta.get("views") or [])]
    out: List[Tuple[str, np.ndarray]] = []
    for idx in range(min(int(rgb.shape[0]), int(max_views))):
        view_name = views[idx] if idx < len(views) else f"view{idx:02d}"
        frames_tchw = np.ascontiguousarray(rgb[idx])
        if frames_tchw.ndim != 4 or int(frames_tchw.shape[1]) != 3:
            continue
        frames_thwc = np.transpose(frames_tchw, (0, 2, 3, 1))
        out.append((view_name, np.ascontiguousarray(frames_thwc)))
    return out


def _png_rgb_views(sample_dir: Path, *, max_views: int) -> List[Tuple[str, np.ndarray]]:
    img_root = sample_dir / "images"
    if not img_root.is_dir():
        return []
    out: List[Tuple[str, np.ndarray]] = []
    for view_name in _list_view_dirs(img_root)[: int(max_views)]:
        files = _sorted_image_files(img_root / view_name)
        if not files:
            continue
        frames: List[np.ndarray] = []
        for p in files:
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if frames:
            out.append((view_name, np.stack(frames, axis=0)))
    return out


def _tensor_rgb_views(sample_dir: Path, *, max_views: int) -> List[Tuple[str, np.ndarray]]:
    import torch
    tensor_root = sample_dir / "arch4_tensors"
    if not tensor_root.is_dir():
        return []
    out: List[Tuple[str, np.ndarray]] = []
    for stem in _list_tensor_stems(tensor_root)[: int(max_views)]:
        pt_path = tensor_root / f"{stem}.pt"
        if not pt_path.is_file():
            continue
        try:
            obj = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(pt_path), map_location="cpu")
        if not isinstance(obj, dict) or "rgb" not in obj:
            continue
        rgb = obj["rgb"] if isinstance(obj["rgb"], torch.Tensor) else torch.as_tensor(obj["rgb"])
        if rgb.dim() != 4 or int(rgb.shape[0]) != 3:
            continue
        arr = rgb.detach().cpu().numpy()
        arr = np.transpose(arr, (1, 2, 3, 0))
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        out.append((stem, np.ascontiguousarray(arr)))
    return out


def _existing_rgb_video(sample_dir: Path) -> Optional[Tuple[str, Path]]:
    video_root = sample_dir / "videos"
    if not video_root.is_dir():
        return None
    candidates = [
        p for p in sorted(video_root.glob("*.mp4"))
        if p.is_file() and RGB_VIEW_MP4_RE.match(p.name)
    ]
    if not candidates:
        return None
    rgb_path = candidates[0]
    return rgb_path.stem, rgb_path


def _resample_frames_thwc(frames_thwc_u8: np.ndarray, target_frames: int) -> np.ndarray:
    frames = np.asarray(frames_thwc_u8, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] <= 0:
        raise ValueError(f"invalid rgb frames for resampling: {frames.shape}")
    target = max(1, int(target_frames))
    if int(frames.shape[0]) == target:
        return np.ascontiguousarray(frames)
    indices = np.linspace(0, int(frames.shape[0]) - 1, target)
    indices = np.rint(indices).astype(np.int64)
    return np.ascontiguousarray(frames[indices])


def _read_video_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"无法从视频读取帧: {video_path}")
    return np.stack(frames, axis=0)


def _video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def _write_rgb_video(
    frames_thwc_u8: np.ndarray,
    out_path: Path,
    *,
    fps: float = 12.0,
    target_frames: int = VLM_INPUT_NUM_FRAMES,
) -> Path:
    frames = np.asarray(frames_thwc_u8, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] <= 0:
        raise ValueError(f"invalid rgb frames for video writing: {frames.shape}")
    frames = _resample_frames_thwc(frames, target_frames)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {out_path}")
    try:
        for frame_rgb in frames:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return out_path


def _prepare_sample_video(
    sample_dir: Path,
    *,
    output_root: Path,
    sample_id: str,
    max_views: int = 1,
    target_frames: int = VLM_INPUT_NUM_FRAMES,
) -> Tuple[Path, str]:
    existing = _existing_rgb_video(sample_dir)
    if existing is not None:
        view_name, video_path = existing
        if _video_frame_count(video_path) == int(target_frames):
            return video_path, view_name
        temp_dir = output_root / "temp_videos"
        fixed_path = temp_dir / f"{sample_id}__{view_name}__{int(target_frames)}f.mp4"
        if not fixed_path.is_file():
            _write_rgb_video(
                _read_video_frames(video_path),
                fixed_path,
                target_frames=target_frames,
            )
        return fixed_path, view_name

    rgb_views: List[Tuple[str, np.ndarray]] = []
    for loader in (_sample_pack_rgb_views, _lmdb_rgb_views, _png_rgb_views, _tensor_rgb_views):
        rgb_views = loader(sample_dir, max_views=max_views)
        if rgb_views:
            break
    if not rgb_views:
        raise FileNotFoundError(f"样本缺少可用于 VLM 的 RGB 视频/帧缓存: {sample_dir}")

    view_name, rgb_frames = rgb_views[0]
    temp_dir = output_root / "temp_videos"
    video_path = temp_dir / f"{sample_id}__{view_name}__{int(target_frames)}f.mp4"
    if not video_path.is_file():
        _write_rgb_video(rgb_frames, video_path, target_frames=target_frames)
    return video_path, view_name


def _ensure_vlm_input_video(
    video_path: Path,
    *,
    output_root: Path,
    sample_id: str,
    view_name: str = "video",
    target_frames: int = VLM_INPUT_NUM_FRAMES,
) -> Path:
    if _video_frame_count(video_path) == int(target_frames):
        return video_path
    temp_dir = output_root / "temp_videos"
    fixed_path = temp_dir / f"{sample_id}__{view_name}__{int(target_frames)}f.mp4"
    if not fixed_path.is_file():
        _write_rgb_video(
            _read_video_frames(video_path),
            fixed_path,
            target_frames=target_frames,
        )
    return fixed_path


def run_benchmark(
    physgaussian_root: Path,
    output_root: Path,
    max_videos: Optional[int] = None,
    random_sample: bool = False,
    debug_print: bool = True,
    vlm_tag: str = "qwen2.5-vl",
    fallback_vlm_tags: Optional[List[str]] = None,
    resume_from_output_root: Optional[Path] = None,
    split_json: Optional[Path] = None,
    eval_split: str = "test",
    seed: int = 0,
) -> Dict[str, Any]:
    script_dir = physgaussian_root / "vlm_benchmark"
    auto_output_root = physgaussian_root / "auto_output"

    if not auto_output_root.is_dir():
        raise FileNotFoundError(f"未找到 auto_output 目录: {auto_output_root}")

    system_prompt = _load_prompt(script_dir / "prompts" / "system_prompt.txt")
    user_prompt = _load_prompt(script_dir / "prompts" / "user_prompt_video_regression.txt")

    tag_chain = _parse_tag_chain(vlm_tag, fallback_vlm_tags)
    active_tag_idx = 0
    active_tag = tag_chain[active_tag_idx]
    client = create_vlm_client(active_tag)

    # 当前模型的输出目录：vlm_benchmark/output/<vlm_tag>/
    output_root.mkdir(parents=True, exist_ok=True)
    _bootstrap_output_root(output_root, resume_from_output_root)
    gt_csv = output_root / "vlm_benchmark_gt.csv"
    pred_csv = output_root / "vlm_benchmark_pred.csv"
    meta_json = output_root / "benchmark_meta.json"
    reasoning_dir = output_root / "reasoning"
    reasoning_dir.mkdir(parents=True, exist_ok=True)

    existing_gt_rows = _dedupe_rows_by_sample_id(_read_csv_rows(gt_csv))
    existing_pred_rows = _normalize_existing_pred_rows(_dedupe_rows_by_sample_id(_read_csv_rows(pred_csv)))
    completed_sample_ids = {
        str(row.get("sample_id", "")).strip()
        for row in existing_pred_rows
        if str(row.get("sample_id", "")).strip()
    }
    existing_completed = len(completed_sample_ids)

    # random_sample 模式下：将抽到的视频打包保存，便于后续对比/复现
    sampled_videos_dir = output_root / "sampled_videos" if random_sample else None
    sampled_manifest_path = output_root / "sampled_videos_manifest.csv" if random_sample else None
    sampled_manifest_f = None
    if sampled_videos_dir is not None and sampled_manifest_path is not None:
        sampled_videos_dir.mkdir(parents=True, exist_ok=True)
        sampled_manifest_f = open(sampled_manifest_path, "a", encoding="utf-8")
        if sampled_manifest_path.stat().st_size == 0:
            sampled_manifest_f.write(
                "sample_id,action,src_rel_video_path,dst_rel_video_path\n"
            )

    gt_f = open(gt_csv, "w", encoding="utf-8", newline="")
    pred_f = open(pred_csv, "w", encoding="utf-8", newline="")

    # sample_id 作为第一列，方便对齐和跨模型比较
    gt_fieldnames = ["sample_id", "action", "material_gt", "E_gt", "nu_gt", "density_gt", "yield_stress_gt"]
    pred_fieldnames = [
        "sample_id",
        "action",
        "material_gt",
        "E_gt",
        "nu_gt",
        "density_gt",
        "yield_stress_gt",
        "E_pred",
        "nu_pred",
        "density_pred",
        "yield_stress_pred",
        "material_pred",
        "motion_pred",
        "view_name",
        "video_rel_path",
        "source_sample_dir",
        "used_vlm_tag",
        "used_vlm_model",
    ]
    gt_writer = csv.DictWriter(gt_f, fieldnames=gt_fieldnames)
    pred_writer = csv.DictWriter(pred_f, fieldnames=pred_fieldnames)
    gt_writer.writeheader()
    pred_writer.writeheader()
    for row in existing_gt_rows:
        gt_writer.writerow({key: row.get(key, "") for key in gt_fieldnames})
    for row in existing_pred_rows:
        pred_writer.writerow({key: row.get(key, "") for key in pred_fieldnames})

    # 收集所有候选样本
    if split_json is not None:
        all_runs = _collect_split_samples(
            physgaussian_root,
            split_json=split_json,
            eval_split=eval_split,
            max_samples=max_videos,
            random_sample=random_sample,
            seed=seed,
        )
    else:
        all_runs = _collect_all_runs(auto_output_root)
    if not all_runs:
        print("未在 auto_output 下找到任何仿真目录。")
        gt_f.close()
        pred_f.close()
        return {"num_samples_used": 0, "chosen_sample_ids": []}

    # 随机采样模式：打乱顺序
    if random_sample and split_json is None:
        rng = random.Random(42)
        rng.shuffle(all_runs)

    num_processed = 0
    chosen_sample_ids: List[str] = [sid for sid in sorted(completed_sample_ids)]
    used_vlm_tags: List[str] = [
        str(row.get("used_vlm_tag", "")).strip()
        for row in existing_pred_rows
        if str(row.get("sample_id", "")).strip()
    ]
    try:
        for action, run_dir, params in all_runs:
            if max_videos is not None and int(max_videos) > 0 and (existing_completed + num_processed) >= max_videos:
                break

            gt_path = run_dir / "gt.json"
            sample_id = run_dir.name if gt_path.is_file() else ""
            if gt_path.is_file():
                sample_id = str(run_dir.name)
                with gt_path.open("r", encoding="utf-8") as f:
                    gt = json.load(f)
                rel_params = gt.get("regression") or {}
                params = {str(k): str(v) for k, v in params.items()}
                if isinstance(rel_params, dict):
                    for k in ("E", "nu", "density", "yield_stress"):
                        if k in rel_params:
                            params[k] = str(rel_params[k])
                material_obj = gt.get("params") or {}
                if isinstance(material_obj, dict) and "material" in material_obj:
                    params["material"] = str(material_obj.get("material"))
                object_name = str(gt.get("object", run_dir.parent.name))
            else:
                object_name = _object_name_for_sample(run_dir)
                sample_id = f"{object_name}_{num_processed + 1:04d}"

            if sample_id in completed_sample_ids:
                print(f"[SKIP] 已有结果，跳过样本 {sample_id}")
                continue

            if gt_path.is_file():
                video_path, view_name = _prepare_sample_video(
                    run_dir,
                    output_root=output_root,
                    sample_id=sample_id,
                    max_views=1,
                )
            else:
                video_path = _find_video_in_run_dir(run_dir)
                view_name = ""
                if video_path is None:
                    continue
                video_path = _ensure_vlm_input_video(
                    video_path,
                    output_root=output_root,
                    sample_id=sample_id,
                    view_name="existing_video",
                )

            rel_video_path = video_path.relative_to(physgaussian_root).as_posix()

            material_gt = _normalize_material_label(params)
            E_gt = _safe_float(params, "E")
            nu_gt = _safe_float(params, "nu")
            density_gt = _safe_float(params, "density")
            yield_stress_gt = _safe_float(params, "yield_stress")

            # random_sample 模式：复制本次采样到的视频到 output/<vlm_tag>/sampled_videos/
            if sampled_videos_dir is not None and sampled_manifest_f is not None:
                ext = video_path.suffix.lower() if video_path.suffix else ".mp4"
                dst_name = f"{sample_id}{ext}"
                dst_path = sampled_videos_dir / dst_name
                try:
                    shutil.copy2(video_path, dst_path)
                except Exception as exc:
                    print(f"[WARN] 复制采样视频失败: {video_path} -> {dst_path} ({exc})")
                dst_rel = dst_path.relative_to(output_root).as_posix()
                sampled_manifest_f.write(
                    f"{sample_id},{action},{rel_video_path},{dst_rel}\n"
                )

            gt_writer.writerow(
                {
                    "sample_id": sample_id,
                    "action": action,
                    "material_gt": material_gt,
                    "E_gt": E_gt,
                    "nu_gt": nu_gt,
                    "density_gt": density_gt,
                    "yield_stress_gt": yield_stress_gt,
                }
            )

            # 调用 VLM：若命中额度/限流错误，则切到下一个模型继续同一样本
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
                    print(f"[WARN] VLM 调用失败，跳过样本 {sample_id}: {exc}")
                    with open(reasoning_dir / f"{sample_id}.txt", "w", encoding="utf-8") as rf:
                        rf.write(f"[ERROR][{active_tag}] {exc}\n")
                    prediction = None
                    break
            if prediction is None:
                continue

            # 将 raw model text 保存到 reasoning/<sample_id>.txt
            raw_text = ""
            try:
                raw_text = str(prediction.get("__raw_text__", "") or "")
            except Exception:
                raw_text = ""
            reasoning_path = reasoning_dir / f"{sample_id}.txt"
            with open(reasoning_path, "w", encoding="utf-8") as rf:
                rf.write(raw_text)

            E_pred = float(prediction.get("E", 0.0))
            nu_pred = float(prediction.get("nu", 0.0))
            density_pred = float(prediction.get("density", 0.0))
            yield_stress_pred = float(prediction.get("yield_stress", 0.0))
            material_pred = str(prediction.get("material_type", "")).lower()
            motion_pred = str(prediction.get("motion_type", "")).lower()
            used_vlm_model = str(VLM_REGISTRY[used_tag].model)

            if debug_print:
                print("\n================ VLM 调用结果 ================")
                print(f"样本编号: {num_processed + 1}")
                print(f"vlm_tag: {vlm_tag}")
                print(f"used_vlm_tag: {used_tag}")
                print(f"used_vlm_model: {used_vlm_model}")
                print(f"sample_id: {sample_id}")
                print(f"视频路径: {rel_video_path}")
                print(f"动作类型 (GT): {action}")
                print(f"材质类型 (GT): {material_gt}")
                print(
                    f"GT 数值: E={E_gt:.3g}, nu={nu_gt:.3g}, "
                    f"density={density_gt:.3g}, yield_stress={yield_stress_gt:.3g}"
                )
                print("原始 JSON 输出:")
                try:
                    print(json.dumps(prediction, ensure_ascii=False, indent=2))
                except TypeError:
                    print(prediction)
                raw_text = prediction.get("__raw_text__")
                if raw_text:
                    print("\n[RAW MODEL TEXT]")
                    print(str(raw_text))
                print(
                    "解析后数值: "
                    f"E_pred={E_pred:.3g}, nu_pred={nu_pred:.3g}, "
                    f"density_pred={density_pred:.3g}, "
                    f"yield_stress_pred={yield_stress_pred:.3g}, "
                    f"material_pred={material_pred}, motion_pred={motion_pred}"
                )
                print("=============================================\n")

            pred_writer.writerow(
                {
                    "sample_id": sample_id,
                    "action": action,
                    "material_gt": material_gt,
                    "E_gt": E_gt,
                    "nu_gt": nu_gt,
                    "density_gt": density_gt,
                    "yield_stress_gt": yield_stress_gt,
                    "E_pred": E_pred,
                    "nu_pred": nu_pred,
                    "density_pred": density_pred,
                    "yield_stress_pred": yield_stress_pred,
                    "material_pred": material_pred,
                    "motion_pred": motion_pred,
                    "view_name": view_name,
                    "video_rel_path": rel_video_path,
                    "source_sample_dir": run_dir.resolve().as_posix(),
                    "used_vlm_tag": used_tag,
                    "used_vlm_model": used_vlm_model,
                }
            )
            gt_f.flush()
            pred_f.flush()

            completed_sample_ids.add(sample_id)
            chosen_sample_ids.append(sample_id)
            used_vlm_tags.append(used_tag)
            num_processed += 1
            print(f"[OK] 已处理视频 {num_processed}: {rel_video_path} [{used_tag}]")
    finally:
        gt_f.close()
        pred_f.close()
        if sampled_manifest_f is not None:
            sampled_manifest_f.close()

    meta = {
        "vlm_tag": vlm_tag,
        "fallback_vlm_tags": list(fallback_vlm_tags or []),
        "resume_from_output_root": str(resume_from_output_root.resolve()) if resume_from_output_root is not None else "",
        "vlm_tag_chain": tag_chain,
        "physgaussian_root": str(physgaussian_root),
        "source_mode": "split_json" if split_json is not None else "auto_output_walk",
        "split_json": str(split_json.resolve()) if split_json is not None else "",
        "eval_split": str(eval_split),
        "max_videos": int(max_videos or 0),
        "random_sample": bool(random_sample),
        "seed": int(seed),
        "existing_completed": int(existing_completed),
        "newly_completed": int(num_processed),
        "num_samples_used": int(len(completed_sample_ids)),
        "chosen_sample_ids": chosen_sample_ids,
        "used_vlm_tags": used_vlm_tags,
        "gt_csv": str(gt_csv),
        "pred_csv": str(pred_csv),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"完成。共写入 {num_processed} 条样本。")
    print(f"GT 文件:   {gt_csv}")
    print(f"预测文件: {pred_csv}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通用 VLM benchmark：对 PhysGaussian/auto_output 中的视频进行评测。"
    )
    parser.add_argument(
        "--physgaussian_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="PhysGaussian 根目录（包含 auto_output 和 vlm_benchmark）",
    )
    parser.add_argument(
        "--vlm_tag",
        type=str,
        default="qwen2.5-vl",
        choices=list(VLM_REGISTRY.keys()),
        help="选择要使用的 VLM 标签（在 vlm_model_registry.py 中维护具体配置）",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default="",
        help="统一输出目录标签；为空时默认使用 --vlm_tag。",
    )
    parser.add_argument(
        "--resume_from_output_tag",
        type=str,
        default="",
        help="若提供且输出目录为空，可从 vlm_benchmark/output/<该标签> 继承已有结果后续跑。",
    )
    parser.add_argument(
        "--fallback_vlm_tags",
        type=str,
        default="",
        help="逗号分隔的回退模型标签；当当前模型命中额度/限流错误时按顺序切换。",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="最多评测多少个视频（默认全部）",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="是否以随机顺序从 bend/drop/press/shear/stretch 采样",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default="",
        help="若提供，则按 split json 里的 train/test id 选择样本，而不是遍历整个 auto_output。",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=("train", "test"),
        help="配合 --split_json 使用，指定评测 split。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机抽样时使用的随机种子。",
    )

    args = parser.parse_args()
    phys_root = Path(args.physgaussian_root).resolve()

    # 统一输出目录：vlm_benchmark/output/<output_tag or vlm_tag>/
    output_tag = str(args.output_tag).strip() or str(args.vlm_tag)
    output_root = phys_root / "vlm_benchmark" / "output" / output_tag

    run_benchmark(
        physgaussian_root=phys_root,
        output_root=output_root,
        max_videos=args.max_videos,
        random_sample=args.random_sample,
        debug_print=True,
        vlm_tag=args.vlm_tag,
        fallback_vlm_tags=[x.strip() for x in str(args.fallback_vlm_tags).split(",") if x.strip()],
        resume_from_output_root=(
            phys_root / "vlm_benchmark" / "output" / str(args.resume_from_output_tag).strip()
            if str(args.resume_from_output_tag).strip()
            else None
        ),
        split_json=(Path(args.split_json).expanduser().resolve() if str(args.split_json).strip() else None),
        eval_split=args.eval_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()