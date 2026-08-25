"""
Particle filling 结果缓存：写入 <model_root>/filled/...，同一 PLY + 相同预处理/填充参数时直接加载，跳过 fill_particles。

失效条件：filling_meta.json 中的 fingerprint 与当前配置不一致，或 PLY 文件大小/mtime 变化。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

import torch


def safe_dir_name(name: str, max_len: int = 120) -> str:
    s = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return s[:max_len] if len(s) > max_len else s


def cache_dir_for_ply_model(model_root: str, ply_stem: str) -> str:
    """PLY 位于 model_root/*.ply 时，缓存目录为 model_root/filled/<stem>/"""
    return os.path.join(os.path.abspath(model_root), "filled", safe_dir_name(ply_stem))


def cache_dir_for_checkpoint_model(model_path: str, iteration: int) -> str:
    """checkpoint 模式：model_path/filled/checkpoint_iter_<iteration>/"""
    return os.path.join(
        os.path.abspath(model_path),
        "filled",
        f"checkpoint_iter_{int(iteration)}",
    )


def _json_normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            str(k): _json_normalize(v)
            for k, v in sorted(obj.items(), key=lambda x: str(x[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_json_normalize(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


def ply_file_stat_fingerprint(ply_path: str) -> str:
    st = os.stat(ply_path)
    ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
    return f"{st.st_size}_{ns}"


def build_filling_fingerprint(
    *,
    ply_path: str,
    preprocessing_params: dict,
    material_params: dict,
    filling_params: dict,
    ply_extra_x90: bool,
) -> str:
    """
    与 fill_particles 输入相关的配置摘要（含源 PLY 文件 stat）。
    ply_extra_x90：modified_simulation 对独立 PLY 在预处理前绕 x 轴转 90°，必须与 checkpoint 区分。
    """
    fp_dict = {
        "version": 2,
        "ply_stat": ply_file_stat_fingerprint(ply_path),
        "ply_extra_x90": bool(ply_extra_x90),
        "opacity_threshold": float(preprocessing_params["opacity_threshold"]),
        "rotation_degree": _json_normalize(preprocessing_params["rotation_degree"]),
        "rotation_axis": _json_normalize(preprocessing_params["rotation_axis"]),
        "sim_area": _json_normalize(preprocessing_params["sim_area"]),
        "scale": float(preprocessing_params["scale"]),
        "material_grid_lim": float(material_params["grid_lim"]),
        "material_n_grid": int(material_params["n_grid"]),
        "filling": _json_normalize(dict(filling_params)),
    }
    raw = json.dumps(fp_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def try_load_filled_positions(
    cache_dir: str,
    fingerprint: str,
    device: str,
    expected_gs_num: int,
) -> Optional[torch.Tensor]:
    meta_path = os.path.join(cache_dir, "filling_meta.json")
    pt_path = os.path.join(cache_dir, "mpm_init_pos.pt")
    if not (os.path.isfile(meta_path) and os.path.isfile(pt_path)):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("fingerprint") != fingerprint:
        return None
    if int(meta.get("gs_num_surface", -1)) != int(expected_gs_num):
        return None
    if int(meta.get("cache_version", 0)) != 2:
        return None
    try:
        data = torch.load(pt_path, map_location=device, weights_only=False)
    except TypeError:
        data = torch.load(pt_path, map_location=device)
    pos = data["mpm_init_pos"]
    if not isinstance(pos, torch.Tensor):
        pos = torch.as_tensor(pos)
    pos = pos.to(device=device, dtype=torch.float32)
    return pos


def save_filled_positions(
    cache_dir: str,
    fingerprint: str,
    mpm_init_pos: torch.Tensor,
    gs_num_surface: int,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "fingerprint": fingerprint,
        "gs_num_surface": int(gs_num_surface),
        "n_total": int(mpm_init_pos.shape[0]),
        "cache_version": 2,
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(cache_dir, "filling_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    torch.save(
        {"mpm_init_pos": mpm_init_pos.detach().cpu()},
        os.path.join(cache_dir, "mpm_init_pos.pt"),
    )
