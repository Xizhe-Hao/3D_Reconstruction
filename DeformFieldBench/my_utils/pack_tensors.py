from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch


def _parse_tensor_storage_dtype(name: str) -> torch.dtype:
    """打包到 .pt 时的张量 dtype；训练侧应 ``.float()`` 再送 GPU。"""
    k = str(name).strip().lower()
    if k in ("float16", "fp16", "half"):
        return torch.float16
    if k in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def _sorted_pngs(d: Path) -> List[Path]:
    files = sorted([p for p in d.glob("*.png") if p.is_file()])
    return files


def compress_png_directory(dir_path: Union[str, Path], compression: int = 6) -> int:
    """
    对目录下 ``*.png`` 原地重写，仅提高 zlib 压缩级别（PNG 仍无损，像素不变）。
    返回成功重写的文件数。
    """
    d = Path(dir_path)
    if not d.is_dir():
        return 0
    level = max(0, min(9, int(compression)))
    flags = [int(cv2.IMWRITE_PNG_COMPRESSION), level]
    n = 0
    for p in _sorted_pngs(d):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if cv2.imwrite(str(p), img, flags):
            n += 1
    return n


def compress_multiview_render_png_dirs(
    view_dirs: Sequence[Optional[str]],
    stress_gaussian_dirs: Sequence[Optional[str]],
    flow_gaussian_dirs: Sequence[Optional[str]],
    force_mask_dirs: Sequence[Optional[str]],
    *,
    compression: int = 6,
) -> int:
    """对 RGB / stress_gaussian / flow_gaussian / force_mask 各视角子目录中的 PNG 做 zlib 压缩。"""
    total = 0
    for group in (
        view_dirs,
        stress_gaussian_dirs,
        flow_gaussian_dirs,
        force_mask_dirs,
    ):
        for d in group:
            if not d:
                continue
            total += compress_png_directory(d, compression)
    return total


def downscale_png_directory(dir_path: Union[str, Path], target_w: int, target_h: int) -> int:
    """
    将目录下各 ``*.png`` 缩放到 ``(target_w, target_h)``（``INTER_AREA``），原地覆盖。
    已与目标尺寸一致的文件跳过。返回处理的文件数。
    """
    d = Path(dir_path)
    tw, th = max(2, int(target_w)), max(2, int(target_h))
    if not d.is_dir():
        return 0
    n = 0
    for p in _sorted_pngs(d):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        if (w0, h0) == (tw, th):
            n += 1
            continue
        resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        if cv2.imwrite(str(p), resized):
            n += 1
    return n


def downscale_multiview_render_png_dirs(
    view_dirs: Sequence[Optional[str]],
    stress_gaussian_dirs: Sequence[Optional[str]],
    flow_gaussian_dirs: Sequence[Optional[str]],
    force_mask_dirs: Sequence[Optional[str]],
    *,
    target_w: int,
    target_h: int,
) -> int:
    """与 ``compress_multiview_render_png_dirs`` 相同的四类视角目录，统一缩放到同一分辨率。"""
    total = 0
    for group in (
        view_dirs,
        stress_gaussian_dirs,
        flow_gaussian_dirs,
        force_mask_dirs,
    ):
        for d in group:
            if not d:
                continue
            total += downscale_png_directory(d, target_w, target_h)
    return total


def compute_export_resolution(
    native_w: int,
    native_h: int,
    *,
    max_side: int = 0,
    scale: float = 1.0,
) -> Tuple[int, int]:
    """
    根据原生光栅宽高计算导出 PNG 目标分辨率（宽、高均为偶数，便于视频编码）。
    ``max_side>0`` 时优先：最长边缩放到不超过 ``max_side``；否则若 ``0<scale<1`` 则整体乘以 scale。
    """
    w, h = max(2, int(native_w)), max(2, int(native_h))
    ms = int(max_side)
    if ms > 0:
        m = max(w, h)
        if m > ms:
            s = ms / float(m)
            w = max(2, int(round(w * s)) // 2 * 2)
            h = max(2, int(round(h * s)) // 2 * 2)
        return w, h
    sc = float(scale)
    if 0.0 < sc < 1.0:
        w = max(2, int(round(w * sc)) // 2 * 2)
        h = max(2, int(round(h * sc)) // 2 * 2)
    return w, h


def _read_png_rgb_float(path: Path, img_size: Optional[int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"无法读取 PNG: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img_size is not None and (rgb.shape[0] != img_size or rgb.shape[1] != img_size):
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0


def _stack_tchw(
    files: Sequence[Path],
    num_frames: Optional[int],
    img_size: Optional[int],
) -> np.ndarray:
    if not files:
        raise RuntimeError("空文件列表，无法堆叠")
    arrs = [_read_png_rgb_float(p, img_size) for p in files]
    # HWC -> CHW
    chw = [np.transpose(a, (2, 0, 1)) for a in arrs]
    x = np.stack(chw, axis=0)  # [T,C,H,W]
    if num_frames is not None and num_frames > 0:
        t = int(num_frames)
        if x.shape[0] >= t:
            x = x[:t]
        else:
            pad = np.repeat(x[-1:], t - x.shape[0], axis=0)
            x = np.concatenate([x, pad], axis=0)
    return np.ascontiguousarray(x.astype(np.float32))


def pack_sample_arch4_tensors(
    sample_dir: str | Path,
    *,
    out_subdir: str = "arch4_tensors",
    force_mask_subdir: str = "force_mask",
    num_frames: Optional[int] = None,
    img_size: Optional[int] = None,
    overwrite: bool = True,
    tensor_dtype: str = "float32",
) -> Dict[str, int]:
    """
    将单个样本目录中的 images/stress_gaussian/flow_gaussian/force_mask 按视角打包为 .pt。
    ``force_mask_subdir`` 可指定 force_mask 来源目录（如 object_mask）。

    输出字段与现有 pipeline 兼容：
      version=3, rgb/stress/flow/force_mask [3,T,H,W], x (2,T,H,W), num_frames, img_size, view,
      storage_dtype（字符串，如 float32/float16/bfloat16）便于下游选择是否转回 float32。

    ``tensor_dtype``：磁盘存储 dtype。``float16``/``bfloat16`` 约减半体积；读取训练时常 ``.float()``。
    """
    sample = Path(sample_dir).resolve()
    img_root = sample / "images"
    stress_root = sample / "stress_gaussian"
    flow_root = sample / "flow_gaussian"
    force_mask_root = sample / str(force_mask_subdir).strip().strip("/\\")
    out_root = sample / out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    if not img_root.is_dir() or not stress_root.is_dir() or not flow_root.is_dir():
        return {"written": 0, "skipped": 0}

    views = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
    written = 0
    skipped = 0
    for v in views:
        d_img = img_root / v
        d_st = stress_root / v
        d_fl = flow_root / v
        d_fm = force_mask_root / v
        if not d_st.is_dir() or not d_fl.is_dir():
            skipped += 1
            continue
        outp = out_root / f"{v}.pt"
        if outp.exists() and not overwrite:
            skipped += 1
            continue

        f_img = _sorted_pngs(d_img)
        f_st = _sorted_pngs(d_st)
        f_fl = _sorted_pngs(d_fl)
        if not f_img or not f_st or not f_fl:
            skipped += 1
            continue

        rgb = _stack_tchw(f_img, num_frames=num_frames, img_size=img_size)
        stress = _stack_tchw(f_st, num_frames=num_frames, img_size=img_size)
        flow = _stack_tchw(f_fl, num_frames=num_frames, img_size=img_size)
        if d_fm.is_dir():
            f_fm = _sorted_pngs(d_fm)
            if f_fm:
                force_mask = _stack_tchw(f_fm, num_frames=num_frames, img_size=img_size)
            else:
                force_mask = np.zeros_like(rgb, dtype=np.float32)
        else:
            force_mask = np.zeros_like(rgb, dtype=np.float32)
        t_eff = int(rgb.shape[0])
        h_eff = int(rgb.shape[2])

        # x: [2,T,H,W]，用于兼容旧训练读取逻辑
        x0 = stress.mean(axis=1)  # [T,H,W]
        x1 = flow.mean(axis=1)  # [T,H,W]
        x = np.stack([x0, x1], axis=0).astype(np.float32)

        dt = _parse_tensor_storage_dtype(tensor_dtype)
        blob = {
            "version": 3,
            "rgb": torch.from_numpy(rgb).to(dtype=dt),
            "stress": torch.from_numpy(stress).to(dtype=dt),
            "flow": torch.from_numpy(flow).to(dtype=dt),
            "force_mask": torch.from_numpy(force_mask).to(dtype=dt),
            "x": torch.from_numpy(x).to(dtype=dt),
            "num_frames": t_eff,
            "img_size": h_eff,
            "view": str(v),
            "storage_dtype": (
                "bfloat16"
                if dt == torch.bfloat16
                else ("float16" if dt == torch.float16 else "float32")
            ),
        }
        torch.save(blob, outp)
        written += 1

    return {"written": written, "skipped": skipped}

