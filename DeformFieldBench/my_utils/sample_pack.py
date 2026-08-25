from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from my_utils.arch4_lmdb import (
    _read_view_pngs_to_thwc_uint8,
    _resample_thwc_float,
    unpack_uint8_thwc,
)

_REQ_MODALITIES = ("rgb", "stress", "flow", "force_mask")
_OPT_MODALITIES = ("object_mask",)
_KEY_META = b"__meta__"


def _open_lmdb_readonly(env_path: Path) -> Any:
    try:
        import lmdb
    except ImportError as e:
        raise RuntimeError("需要安装 py-lmdb：pip install lmdb") from e
    return lmdb.open(
        str(env_path),
        readonly=True,
        lock=False,
        readahead=True,
        max_readers=256,
    )


def _read_lmdb_meta(env_path: Path) -> Dict[str, Any]:
    env = _open_lmdb_readonly(env_path)
    try:
        with env.begin() as txn:
            raw = txn.get(_KEY_META)
            if raw is None:
                raise ValueError(f"缺少 __meta__: {env_path}")
            meta = json.loads(raw.decode("utf-8"))
            if not isinstance(meta, dict):
                raise ValueError(f"LMDB __meta__ 不是 dict: {env_path}")
            return meta
    finally:
        env.close()


def _read_lmdb_uint8_arrays(
    env_path: Path,
    *,
    max_views: Optional[int] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    meta = _read_lmdb_meta(env_path)
    views = list(meta.get("views") or [])
    if max_views is not None and int(max_views) > 0:
        views = views[: int(max_views)]
    if not views:
        raise ValueError(f"LMDB 无可用视角: {env_path}")

    modalities: List[str] = list(_REQ_MODALITIES)
    env = _open_lmdb_readonly(env_path)
    try:
        out_buf: Dict[str, List[np.ndarray]] = {m: [] for m in (_REQ_MODALITIES + _OPT_MODALITIES)}
        with env.begin() as txn:
            for view_name in views:
                for mod in _REQ_MODALITIES:
                    blob = txn.get(f"{view_name}/{mod}".encode("utf-8"))
                    if blob is None:
                        raise KeyError(f"缺少键: {view_name}/{mod}")
                    out_buf[mod].append(unpack_uint8_thwc(bytes(blob)))

                blob_obj = txn.get(f"{view_name}/object_mask".encode("utf-8"))
                if blob_obj is not None:
                    out_buf["object_mask"].append(unpack_uint8_thwc(bytes(blob_obj)))

        arrays_u8: Dict[str, np.ndarray] = {}
        for mod in _REQ_MODALITIES:
            arrays_u8[mod] = np.stack(out_buf[mod], axis=0)  # [V,T,3,H,W]
        if out_buf["object_mask"]:
            if len(out_buf["object_mask"]) != len(views):
                raise ValueError(f"object_mask 视角数不完整: {env_path}")
            arrays_u8["object_mask"] = np.stack(out_buf["object_mask"], axis=0)
            modalities.append("object_mask")

        meta_out = dict(meta)
        meta_out["views"] = views
        meta_out["modalities"] = modalities
        meta_out["storage_backend"] = "sample_pack_npz"
        meta_out["sample_pack_format_version"] = 1
        return arrays_u8, meta_out
    finally:
        env.close()


def _sample_pack_path(
    sample_dir_or_pack_path: Union[str, Path],
    sample_pack_name: str = "sample_pack.npz",
) -> Path:
    p = Path(sample_dir_or_pack_path)
    if p.is_dir():
        return p / str(sample_pack_name).strip()
    return p


def write_sample_pack(
    sample_dir: Union[str, Path],
    *,
    resize: int = 224,
    sample_pack_name: str = "sample_pack.npz",
    lmdb_env_subdir: str = "arch4_data.lmdb",
    force_mask_subdir: str = "force_mask",
    object_mask_subdir: str = "object_mask",
    include_object_mask: bool = False,
    num_frames: Optional[int] = None,
    overwrite: bool = True,
    compressed: bool = True,
) -> Dict[str, Any]:
    """
    在 ``sample_dir / sample_pack_name`` 下写入单文件 sample_pack.npz。
    数据来自 images/stress_gaussian/flow_gaussian/force_mask[/object_mask]。
    """
    from my_utils.pack_tensors import _sorted_pngs

    sample = Path(sample_dir).resolve()
    img_root = sample / "images"
    stress_root = sample / "stress_gaussian"
    flow_root = sample / "flow_gaussian"
    env_path = sample / str(lmdb_env_subdir).strip().strip("/\\")
    force_root = sample / str(force_mask_subdir).strip().strip("/\\")
    object_root = sample / str(object_mask_subdir).strip().strip("/\\")
    out_path = sample / str(sample_pack_name).strip().strip("/\\")
    if not img_root.is_dir() or not stress_root.is_dir() or not flow_root.is_dir():
        if env_path.is_dir() and (env_path / "data.mdb").is_file():
            return write_sample_pack_from_lmdb(
                sample,
                sample_pack_name=sample_pack_name,
                lmdb_env_subdir=lmdb_env_subdir,
                overwrite=overwrite,
                max_views=None,
                compressed=compressed,
            )
        return {"written": 0, "skipped": 1, "reason": "missing image roots"}

    views = sorted([p.name for p in img_root.iterdir() if p.is_dir()])
    if not views:
        if env_path.is_dir() and (env_path / "data.mdb").is_file():
            return write_sample_pack_from_lmdb(
                sample,
                sample_pack_name=sample_pack_name,
                lmdb_env_subdir=lmdb_env_subdir,
                overwrite=overwrite,
                max_views=None,
                compressed=compressed,
            )
        return {"written": 0, "skipped": 1, "reason": "no views"}

    if out_path.exists():
        if not overwrite and sample_pack_is_valid(out_path):
            return {
                "written": 0,
                "skipped": 1,
                "reason": "sample_pack exists",
                "sample_pack_path": str(out_path),
                "sample_pack_bytes": int(out_path.stat().st_size),
            }
        out_path.unlink(missing_ok=True)

    R = max(8, int(resize))
    modalities = ["rgb", "stress", "flow", "force_mask"]
    if bool(include_object_mask):
        modalities.append("object_mask")
    meta: Dict[str, Any] = {
        "format_version": 1,
        "magic": "PG4_SAMPLE_PACK_v1",
        "img_size": R,
        "views": [],
        "modalities": modalities,
        "force_mask_source_dir": str(force_root.name),
        "object_mask_source_dir": str(object_root.name),
    }

    buf: Dict[str, List[np.ndarray]] = {m: [] for m in modalities}
    t_eff_max = 0
    views_done: List[str] = []
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
        buf["rgb"].append(rgb)
        buf["stress"].append(stress)
        buf["flow"].append(flow)
        buf["force_mask"].append(force_mask)
        if bool(include_object_mask):
            if d_om.is_dir() and _sorted_pngs(d_om):
                object_mask = _read_view_pngs_to_thwc_uint8(_sorted_pngs(d_om), R, num_frames)
            else:
                object_mask = np.zeros_like(rgb, dtype=np.uint8)
            buf["object_mask"].append(object_mask)
        views_done.append(v)
        t_eff_max = max(t_eff_max, int(rgb.shape[0]))

    if not views_done:
        raise RuntimeError("无可用视角写入 sample_pack（检查 PNG 是否齐全）")

    arrays_u8: Dict[str, np.ndarray] = {}
    for mod in modalities:
        arrays_u8[mod] = np.stack(buf[mod], axis=0)
    meta["views"] = views_done
    meta["num_frames"] = int(t_eff_max)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if bool(compressed) else np.savez
    meta_bytes = np.frombuffer(
        json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        dtype=np.uint8,
    )
    save_fn(out_path, meta_json=meta_bytes, **arrays_u8)
    return {
        "written": int(len(views_done)),
        "skipped": 0,
        "sample_pack_path": str(out_path),
        "sample_pack_bytes": int(out_path.stat().st_size),
        "img_size": R,
        "num_frames": int(t_eff_max),
        "modalities": modalities,
    }


def write_sample_pack_from_lmdb(
    sample_dir: Union[str, Path],
    *,
    sample_pack_name: str = "sample_pack.npz",
    lmdb_env_subdir: str = "arch4_data.lmdb",
    overwrite: bool = False,
    max_views: Optional[int] = None,
    compressed: bool = True,
) -> Dict[str, Any]:
    sample = Path(sample_dir).resolve()
    env_path = sample / str(lmdb_env_subdir).strip().strip("/\\")
    if not env_path.is_dir() or not (env_path / "data.mdb").is_file():
        raise FileNotFoundError(f"LMDB 不存在: {env_path}")

    out_path = sample / str(sample_pack_name).strip().strip("/\\")
    if out_path.exists() and not overwrite and sample_pack_is_valid(out_path):
        return {
            "written": 0,
            "skipped": 1,
            "reason": "sample_pack exists",
            "sample_pack_path": str(out_path),
            "sample_pack_bytes": int(out_path.stat().st_size),
        }

    arrays_u8, meta = _read_lmdb_uint8_arrays(env_path, max_views=max_views)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if bool(compressed) else np.savez
    meta_bytes = np.frombuffer(
        json.dumps(meta, ensure_ascii=False).encode("utf-8"),
        dtype=np.uint8,
    )
    save_fn(
        out_path,
        meta_json=meta_bytes,
        **arrays_u8,
    )
    if not out_path.is_file():
        raise RuntimeError(f"sample_pack 写入失败: {out_path}")
    return {
        "written": 1,
        "skipped": 0,
        "sample_pack_path": str(out_path),
        "sample_pack_bytes": int(out_path.stat().st_size),
        "views": int(arrays_u8["rgb"].shape[0]),
        "num_frames": int(arrays_u8["rgb"].shape[1]),
        "img_size": int(arrays_u8["rgb"].shape[-1]),
        "modalities": list(meta.get("modalities") or []),
    }


def _load_sample_pack_npz(pack_path: Path) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    if not pack_path.is_file():
        raise FileNotFoundError(f"sample_pack 不存在: {pack_path}")

    arrays: Dict[str, np.ndarray] = {}
    with np.load(pack_path, allow_pickle=False) as npz:
        if "meta_json" not in npz.files:
            raise ValueError(f"sample_pack 缺少 meta_json: {pack_path}")
        meta_raw = npz["meta_json"]
        if meta_raw.dtype == np.uint8:
            meta_s = bytes(np.asarray(meta_raw, dtype=np.uint8).tolist()).decode("utf-8")
        elif meta_raw.ndim == 0:
            meta_s = str(meta_raw.tolist())
        else:
            meta_s = "".join(meta_raw.tolist())
        meta = json.loads(meta_s)
        if not isinstance(meta, dict):
            raise ValueError(f"sample_pack meta_json 非 dict: {pack_path}")

        for mod in (_REQ_MODALITIES + _OPT_MODALITIES):
            if mod in npz.files:
                arrays[mod] = np.asarray(npz[mod])

    return meta, arrays


def sample_pack_is_valid(
    sample_dir_or_pack_path: Union[str, Path],
    sample_pack_name: str = "sample_pack.npz",
) -> bool:
    try:
        pack_path = _sample_pack_path(sample_dir_or_pack_path, sample_pack_name)
        meta, arrays = _load_sample_pack_npz(pack_path)
        views = list(meta.get("views") or [])
        if not views:
            return False
        for mod in _REQ_MODALITIES:
            arr = arrays.get(mod)
            if arr is None or arr.ndim != 5:
                return False
            if int(arr.shape[0]) != len(views) or int(arr.shape[2]) != 3:
                return False
        obj = arrays.get("object_mask")
        if obj is not None and (obj.ndim != 5 or int(obj.shape[2]) != 3):
            return False
        return True
    except Exception:
        return False


def read_lmdb_arrays_from_sample_dir(
    sample_dir: Union[str, Path],
    *,
    lmdb_env_subdir: str = "arch4_data.lmdb",
    num_frames: Optional[int] = None,
    img_size: Optional[int] = None,
    max_views: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    sample = Path(sample_dir).resolve()
    env_path = sample / str(lmdb_env_subdir).strip().strip("/\\")
    meta = _read_lmdb_meta(env_path)

    views = list(meta.get("views") or [])
    if max_views is not None and int(max_views) > 0:
        views = views[: int(max_views)]
    if not views:
        raise ValueError(f"LMDB 无可用 views: {env_path}")

    stored_h = int(meta.get("img_size") or 0)
    if img_size is not None and int(img_size) > 0 and stored_h > 0 and int(img_size) != stored_h:
        raise ValueError(f"LMDB img_size={stored_h} 与请求值 {int(img_size)} 不一致")

    env = _open_lmdb_readonly(env_path)
    try:
        resample = num_frames is not None and int(num_frames) > 0
        nf = int(num_frames) if resample else 0
        out_buf: Dict[str, List[np.ndarray]] = {m: [] for m in (_REQ_MODALITIES + _OPT_MODALITIES)}
        with env.begin() as txn:
            for view_name in views:
                for mod in _REQ_MODALITIES:
                    blob = txn.get(f"{view_name}/{mod}".encode("utf-8"))
                    if blob is None:
                        raise KeyError(f"缺少键: {view_name}/{mod}")
                    thwc = unpack_uint8_thwc(bytes(blob)).astype(np.float32) / 255.0
                    if resample:
                        thwc = _resample_thwc_float(thwc, nf)
                    out_buf[mod].append(np.ascontiguousarray(np.transpose(thwc, (1, 0, 2, 3)), dtype=np.float32))

                blob_obj = txn.get(f"{view_name}/object_mask".encode("utf-8"))
                if blob_obj is None:
                    out_buf["object_mask"].append(np.ones_like(out_buf["force_mask"][-1], dtype=np.float32))
                else:
                    thwc = unpack_uint8_thwc(bytes(blob_obj)).astype(np.float32) / 255.0
                    if resample:
                        thwc = _resample_thwc_float(thwc, nf)
                    out_buf["object_mask"].append(
                        np.ascontiguousarray(np.transpose(thwc, (1, 0, 2, 3)), dtype=np.float32)
                    )

        out: Dict[str, np.ndarray] = {}
        for mod in (_REQ_MODALITIES + _OPT_MODALITIES):
            out[mod] = np.stack(out_buf[mod], axis=0)
        return out
    finally:
        env.close()


def read_sample_pack_arrays(
    sample_dir_or_pack_path: Union[str, Path],
    *,
    sample_pack_name: str = "sample_pack.npz",
    num_frames: Optional[int] = None,
    img_size: Optional[int] = None,
    max_views: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    pack_path = _sample_pack_path(sample_dir_or_pack_path, sample_pack_name)
    meta, arrays_u8 = _load_sample_pack_npz(pack_path)
    views = list(meta.get("views") or [])
    if not views:
        raise ValueError(f"sample_pack 无可用 views: {pack_path}")

    v_lim = len(views)
    if max_views is not None and int(max_views) > 0:
        v_lim = min(v_lim, int(max_views))

    stored_h = int(meta.get("img_size") or 0)
    if img_size is not None and int(img_size) > 0 and stored_h > 0 and int(img_size) != stored_h:
        raise ValueError(f"sample_pack img_size={stored_h} 与请求值 {int(img_size)} 不一致")

    resample = num_frames is not None and int(num_frames) > 0
    nf = int(num_frames) if resample else 0

    out: Dict[str, np.ndarray] = {}
    for mod in _REQ_MODALITIES:
        arr_vtchw = arrays_u8.get(mod)
        if arr_vtchw is None:
            raise KeyError(f"sample_pack 缺少模态: {mod}")
        arr_vtchw = np.asarray(arr_vtchw[:v_lim], dtype=np.uint8)
        per_view: List[np.ndarray] = []
        for vi in range(int(arr_vtchw.shape[0])):
            thwc = arr_vtchw[vi].astype(np.float32) / 255.0
            if resample:
                thwc = _resample_thwc_float(thwc, nf)
            chw_t = np.transpose(thwc, (1, 0, 2, 3))
            per_view.append(np.ascontiguousarray(chw_t, dtype=np.float32))
        out[mod] = np.stack(per_view, axis=0)

    arr_obj = arrays_u8.get("object_mask")
    if arr_obj is None:
        out["object_mask"] = np.ones_like(out["force_mask"], dtype=np.float32)
    else:
        arr_obj = np.asarray(arr_obj[:v_lim], dtype=np.uint8)
        per_view_obj: List[np.ndarray] = []
        for vi in range(int(arr_obj.shape[0])):
            thwc = arr_obj[vi].astype(np.float32) / 255.0
            if resample:
                thwc = _resample_thwc_float(thwc, nf)
            chw_t = np.transpose(thwc, (1, 0, 2, 3))
            per_view_obj.append(np.ascontiguousarray(chw_t, dtype=np.float32))
        out["object_mask"] = np.stack(per_view_obj, axis=0)

    return out
