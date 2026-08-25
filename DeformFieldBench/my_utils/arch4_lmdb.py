# -*- coding: utf-8 -*-
"""
将多视角 ``images`` / ``stress_gaussian`` / ``flow_gaussian`` / ``force_mask`` 序列
读入后 **resize 到固定边长**，以 **uint8 [T,3,H,W]** 写入 LMDB，便于训练时顺序读、少文件句柄。

键 ``__meta__``：JSON（utf-8），含 ``views``、``num_frames``、``img_size`` 等。
数据键：``{view}/rgb``、``{view}/stress``、``{view}/flow``、``{view}/force_mask``（utf-8 字节键）。
值：魔数 ``PG4\x02`` + ``struct pack T,C,H,W (uint32 LE)`` + 原始 ``C`` 连续 ``uint8`` 负载。
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_LM_MAGIC = b"PG4\x02"
_KEY_META = b"__meta__"


def _pack_uint8_thwc(arr: np.ndarray) -> bytes:
    """``arr`` shape ``[T,3,H,W]`` uint8，C 连续。"""
    if arr.dtype != np.uint8:
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
    else:
        arr = np.ascontiguousarray(arr)
    t, c, h, w = map(int, arr.shape)
    if c != 3:
        raise ValueError(f"期望 C=3，得到 {arr.shape}")
    head = _LM_MAGIC + struct.pack("<IIII", t, c, h, w)
    return head + arr.tobytes(order="C")


def unpack_uint8_thwc(blob: bytes) -> np.ndarray:
    if len(blob) < 4 + 16:
        raise ValueError("LMDB 值过短")
    if blob[:4] != _LM_MAGIC:
        raise ValueError(f"未知魔数: {blob[:4]!r}")
    t, c, h, w = struct.unpack("<IIII", blob[4:20])
    need = t * c * h * w
    raw = blob[20:]
    if len(raw) < need:
        raise ValueError(f"负载长度不足: need={need} got={len(raw)}")
    x = np.frombuffer(raw[:need], dtype=np.uint8).reshape(t, c, h, w)
    return np.ascontiguousarray(x)


def _read_view_pngs_to_thwc_uint8(
    png_paths: Sequence[Path],
    resize: int,
    num_frames: Optional[int],
) -> np.ndarray:
    import cv2

    if not png_paths:
        raise RuntimeError("空 PNG 列表")
    R = max(8, int(resize))
    frames: List[np.ndarray] = []
    for p in png_paths:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"无法读取: {p}")
        if raw.ndim == 2:
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif raw.ndim == 3 and int(raw.shape[2]) == 1:
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif raw.ndim == 3 and int(raw.shape[2]) >= 3:
            bgr = raw[:, :, :3]
        else:
            raise RuntimeError(f"不支持的 PNG 形状: {p} -> {getattr(raw, 'shape', None)}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != R or rgb.shape[1] != R:
            rgb = cv2.resize(rgb, (R, R), interpolation=cv2.INTER_AREA)
        frames.append(np.ascontiguousarray(rgb.astype(np.uint8)))
    x = np.stack(frames, axis=0)  # [T,3,H,W] after transpose - frames are H,W,3
    x = np.transpose(x, (0, 3, 1, 2))
    if num_frames is not None and int(num_frames) > 0:
        Tt = int(num_frames)
        t0 = int(x.shape[0])
        if t0 >= Tt:
            x = x[:Tt]
        else:
            pad = np.repeat(x[-1:], Tt - t0, axis=0)
            x = np.concatenate([x, pad], axis=0)
    return np.ascontiguousarray(x, dtype=np.uint8)


def write_sample_arch4_lmdb(
    sample_dir: Union[str, Path],
    *,
    resize: int = 224,
    env_rel: str = "arch4_data.lmdb",
    force_mask_subdir: str = "force_mask",
    object_mask_subdir: str = "object_mask",
    include_object_mask: bool = False,
    num_frames: Optional[int] = None,
    map_size_gb: float = 8.0,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    在 ``sample_dir / env_rel`` 下创建 LMDB，写入各视角 rgb/stress/flow/force_mask（uint8）。
    可选追加 object_mask 键（独立于 force_mask）。

    不删除磁盘上的 PNG；视频合成可继续使用原 PNG 目录。
    """
    try:
        import lmdb
    except ImportError as e:
        raise RuntimeError("需要安装 py-lmdb：pip install lmdb") from e

    from my_utils.pack_tensors import _sorted_pngs

    sample = Path(sample_dir).resolve()
    img_root = sample / "images"
    stress_root = sample / "stress_gaussian"
    flow_root = sample / "flow_gaussian"
    force_root = sample / str(force_mask_subdir).strip().strip("/\\")
    object_root = sample / str(object_mask_subdir).strip().strip("/\\")
    env_path = sample / str(env_rel).strip().strip("/\\")
    if not img_root.is_dir() or not stress_root.is_dir() or not flow_root.is_dir():
        return {"written": 0, "skipped": 1, "reason": "missing image roots"}

    views = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
    if not views:
        return {"written": 0, "skipped": 1, "reason": "no views"}

    if env_path.exists():
        if not overwrite:
            return {"written": 0, "skipped": 1, "reason": "lmdb exists"}
        if env_path.is_dir():
            shutil.rmtree(env_path, ignore_errors=True)
        else:
            env_path.unlink(missing_ok=True)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.mkdir(parents=True, exist_ok=True)

    ms = max(int(1 << 20), int(float(map_size_gb) * (1 << 30)))
    R = max(8, int(resize))

    modalities = ["rgb", "stress", "flow", "force_mask"]
    if bool(include_object_mask):
        modalities.append("object_mask")
    meta: Dict[str, Any] = {
        "format_version": 1,
        "magic": "PG4_LMDB_v1",
        "img_size": R,
        "views": views,
        "modalities": modalities,
        "force_mask_source_dir": str(force_root.name),
        "object_mask_source_dir": str(object_root.name),
    }

    env = lmdb.open(str(env_path), map_size=ms, subdir=True, lock=True, metasync=False, sync=True)
    views_done: List[str] = []
    t_eff_max = 0
    try:
        with env.begin(write=True) as txn:
            for v in views:
                d_img = img_root / v
                d_st = stress_root / v
                d_fl = flow_root / v
                d_fm = force_root / v
                d_om = object_root / v
                if not d_st.is_dir() or not d_fl.is_dir():
                    continue
                f_img = _sorted_pngs(d_img)
                f_st = _sorted_pngs(d_st)
                f_fl = _sorted_pngs(d_fl)
                if not f_img or not f_st or not f_fl:
                    continue
                rgb = _read_view_pngs_to_thwc_uint8(f_img, R, num_frames)
                stress = _read_view_pngs_to_thwc_uint8(f_st, R, num_frames)
                flow = _read_view_pngs_to_thwc_uint8(f_fl, R, num_frames)
                if d_fm.is_dir() and _sorted_pngs(d_fm):
                    force_mask = _read_view_pngs_to_thwc_uint8(_sorted_pngs(d_fm), R, num_frames)
                else:
                    force_mask = np.zeros_like(rgb, dtype=np.uint8)
                if bool(include_object_mask):
                    if d_om.is_dir() and _sorted_pngs(d_om):
                        object_mask = _read_view_pngs_to_thwc_uint8(
                            _sorted_pngs(d_om), R, num_frames
                        )
                    else:
                        object_mask = np.zeros_like(rgb, dtype=np.uint8)
                t_eff = int(rgb.shape[0])
                t_eff_max = max(t_eff_max, t_eff)
                mod_arrs = [
                    ("rgb", rgb),
                    ("stress", stress),
                    ("flow", flow),
                    ("force_mask", force_mask),
                ]
                if bool(include_object_mask):
                    mod_arrs.append(("object_mask", object_mask))
                for mod, arr in mod_arrs:
                    key = f"{v}/{mod}".encode("utf-8")
                    txn.put(key, _pack_uint8_thwc(arr))
                views_done.append(v)
            if not views_done:
                raise RuntimeError("无可用视角写入 LMDB（检查 PNG 是否齐全）")
            meta["views"] = views_done
            meta["num_frames"] = int(t_eff_max)
            txn.put(_KEY_META, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    except Exception:
        shutil.rmtree(env_path, ignore_errors=True)
        raise
    finally:
        env.close()

    return {
        "written": int(len(views_done)),
        "skipped": 0,
        "lmdb_path": str(env_path),
        "img_size": R,
        "num_frames": int(t_eff_max),
    }


def _resample_thwc_float(x_thwc: np.ndarray, target_t: int) -> np.ndarray:
    """``[T,3,H,W]`` float32 → 时间维插值为 ``target_t``。"""
    t = int(x_thwc.shape[0])
    T = int(target_t)
    if t == T:
        return x_thwc
    idx = np.linspace(0, t - 1, T).astype(np.int64)
    return np.ascontiguousarray(x_thwc[idx], dtype=np.float32)


def read_arch4_lmdb_view_tensors(
    env_path: Union[str, Path],
    *,
    num_frames: int,
    img_size: int,
    max_views: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    从 LMDB 读出各视角 ``[3,T,H,W]`` float32（0~1），按 ``num_frames`` 重采样时间维。
    视角顺序与 ``__meta__`` 中 ``views`` 一致，至多 ``max_views``。
    """
    try:
        import lmdb
    except ImportError as e:
        raise RuntimeError("需要安装 py-lmdb：pip install lmdb") from e

    path = Path(env_path)
    if not path.is_dir() or not (path / "data.mdb").is_file():
        raise FileNotFoundError(f"无效 LMDB 目录: {path}")

    env = lmdb.open(str(path), readonly=True, lock=False, readahead=True, max_readers=256)
    try:
        with env.begin() as txn:
            raw = txn.get(_KEY_META)
            if raw is None:
                raise ValueError("缺少 __meta__")
            meta = json.loads(raw.decode("utf-8"))
            views_all: List[str] = list(meta.get("views") or [])
            stored_h = int(meta.get("img_size") or 0)
            if stored_h > 0 and int(img_size) != stored_h:
                raise ValueError(
                    f"LMDB img_size={stored_h} 与请求的 img_size={img_size} 不一致"
                )
            rgb_l: List[np.ndarray] = []
            s_l: List[np.ndarray] = []
            f_l: List[np.ndarray] = []
            fm_l: List[np.ndarray] = []
            for v in views_all[: int(max_views)]:
                for mod, out_list in (
                    ("rgb", rgb_l),
                    ("stress", s_l),
                    ("flow", f_l),
                    ("force_mask", fm_l),
                ):
                    key = f"{v}/{mod}".encode("utf-8")
                    blob = txn.get(key)
                    if blob is None:
                        raise KeyError(f"缺少键: {key!r}")
                    u8 = unpack_uint8_thwc(bytes(blob))
                    thwc = u8.astype(np.float32) / 255.0
                    thwc = _resample_thwc_float(thwc, int(num_frames))
                    chw = np.transpose(thwc, (1, 0, 2, 3))
                    out_list.append(np.ascontiguousarray(chw, dtype=np.float32))
            return rgb_l, s_l, f_l, fm_l
    finally:
        env.close()


def lmdb_arch4_is_valid(env_path: Union[str, Path]) -> bool:
    p = Path(env_path)
    if not p.is_dir() or not (p / "data.mdb").is_file():
        return False
    try:
        import lmdb

        env = lmdb.open(str(p), readonly=True, lock=False, max_readers=8)
        try:
            with env.begin() as txn:
                return txn.get(_KEY_META) is not None
        finally:
            env.close()
    except Exception:
        return False
