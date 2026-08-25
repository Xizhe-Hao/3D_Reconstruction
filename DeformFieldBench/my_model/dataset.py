# -*- coding: utf-8 -*-
"""
扁平集 ``auto_output/<dataset>/{train,test}/<id>/``（skills/数据集构建）。

- 单一真值：``gt.json``。
- 张量布局：Dataset 输出 ``x [V, 1, T, H, W]``（仅 images 时序单通道），以及 stress/flow/force_mask
  真值各 ``[V, 1, T, H, W]``。
- 可选返回 ``object_mask [V, 1, T, H, W]``，供 eval 仅在物体区域内统计 field 指标。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from my_utils.arch4_lmdb import lmdb_arch4_is_valid, read_arch4_lmdb_view_tensors
from my_utils.sample_pack import read_lmdb_arrays_from_sample_dir, read_sample_pack_arrays


def resolve_flat_dataset_root(dataset_dir: str, auto_output: Path) -> Path:
    p = Path(dataset_dir).expanduser()
    if p.is_absolute():
        return p.resolve()
    parts = p.parts
    if parts and parts[0] == "auto_output":
        return (auto_output.parent.joinpath(*parts)).resolve()
    return (auto_output / p).resolve()


def list_dataset400_sample_dirs(split_root: Path) -> List[Path]:
    root = Path(split_root)
    if not root.is_dir():
        return []
    numeric = re.compile(r"^\d+$")
    return sorted([d for d in root.iterdir() if d.is_dir() and numeric.match(d.name)], key=lambda x: int(x.name))


def _coerce_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _normalize_regression(reg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, float]:
    """``gt.json`` 里 ``regression`` 可能含 ``null``（如非金属无 ``yield_stress``），统一为 float。"""
    keys = ("E", "nu", "density", "yield_stress")
    out: Dict[str, float] = {}
    for k in keys:
        v = reg.get(k) if isinstance(reg, dict) else None
        if v is None and isinstance(params, dict):
            v = params.get(k)
        out[k] = _coerce_float(v, 0.0)
    return out


def _load_gt_from_dir(sample_dir: Path) -> Tuple[str, dict, dict]:
    with open(sample_dir / "gt.json", "r", encoding="utf-8") as f:
        gt = json.load(f)
    action = str(gt.get("action", ""))
    params = gt.get("parameters") or {}
    reg = gt.get("regression") or {}
    if not reg and params:
        reg = {
            "E": params.get("E", 0),
            "nu": params.get("nu", 0),
            "density": params.get("density", 0),
            "yield_stress": params.get("yield_stress", 0),
        }
    reg = _normalize_regression(reg, params if isinstance(params, dict) else {})
    return action, params, reg


def list_multiview_camera_dirs(root: Path) -> List[str]:
    if not root.is_dir():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def _sorted_image_files(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    fs = [x for x in d.iterdir() if x.is_file() and x.suffix.lower() in exts]
    return sorted(fs, key=lambda p: p.name)


def extract_frames_from_images(view_dir: Path, num_frames: int) -> List[np.ndarray]:
    files = _sorted_image_files(view_dir)
    if not files:
        raise ValueError(f"无图像: {view_dir}")
    n = len(files)
    idx = np.linspace(0, n - 1, num=min(int(num_frames), n)).astype(int).tolist()
    out = []
    for i in idx:
        im = cv2.imread(str(files[int(i)]), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"无法读取: {files[int(i)]}")
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    while len(out) < int(num_frames) and out:
        out.append(out[-1].copy())
    return out[: int(num_frames)]


def extract_frames_from_video(path: Path, sample_fps: float, max_frames: int, out_size: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {path}")
    v_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_meta <= 0:
        cap.release()
        raise ValueError(f"视频无帧元数据: {path}")
    duration = n_meta / max(v_fps, 1e-3)
    n_target = max(1, min(int(max_frames), int(round(duration * float(sample_fps)))))
    idxs = np.linspace(0, n_meta - 1, num=min(n_target, n_meta)).astype(int).tolist()
    want = set(idxs)
    got: Dict[int, np.ndarray] = {}
    fi = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fi in want:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, (out_size, out_size))
            got[fi] = fr.astype(np.float32) / 255.0
        fi += 1
    cap.release()
    if not got:
        raise ValueError(f"视频无帧: {path}")
    ordered = [got[i] for i in sorted(got.keys())]
    while len(ordered) < int(max_frames) and ordered:
        ordered.append(ordered[-1])
    return np.stack(ordered[: int(max_frames)], axis=0)


def list_arch3_video_view_groups(videos_dir: Path) -> List[Tuple[str, Path, Path, Path]]:
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
        elif stem.endswith("_deformation"):
            base = stem[: -len("_deformation")]
            groups.setdefault(base, {})["flow"] = p
        else:
            groups.setdefault(stem, {})["rgb"] = p
    out: List[Tuple[str, Path, Path, Path]] = []
    for base in sorted(groups.keys()):
        g = groups[base]
        if "rgb" in g and "stress" in g and "flow" in g:
            out.append((base, g["rgb"], g["stress"], g["flow"]))
    return out


def list_arch4_video_view_groups(videos_dir: Path) -> List[Tuple[str, Path, Path, Path, Path]]:
    """每组须同时含 base RGB、stress_gaussian、flow_gaussian、force_mask 四个衍生 MP4。"""
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
        elif stem.endswith("_deformation"):
            base = stem[: -len("_deformation")]
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


def _stack_view_images(view_dir: Path, num_frames: int, img_size: int) -> np.ndarray:
    frames = extract_frames_from_images(view_dir, num_frames)
    out = []
    for f in frames:
        f = cv2.resize(f, (img_size, img_size))
        out.append(f)
    arr = np.stack(out, axis=0).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)))


def _pad_views_vcthw(x: np.ndarray, max_views: int) -> np.ndarray:
    v = int(x.shape[0])
    if v >= max_views:
        return x[:max_views]
    out = np.zeros((max_views,) + tuple(x.shape[1:]), dtype=np.float32)
    if v <= 0:
        return out
    out[:v] = x
    if v < max_views:
        out[v:] = x[v - 1 : v]
    return out


def _torch_load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def list_arch4_tensor_stems(cache_dir: Path) -> List[str]:
    p = Path(cache_dir)
    if not p.is_dir():
        return []
    return sorted({x.stem for x in p.glob("*.pt") if x.is_file()})


def _sample_pack_exists(sample_dir: Path, sample_pack_name: str) -> bool:
    return (sample_dir / str(sample_pack_name).strip()).is_file()


def _pt_full_pack(obj: object) -> bool:
    if not isinstance(obj, dict) or int(obj.get("version", 1)) < 2:
        return False
    for k in ("rgb", "stress", "flow"):
        if k not in obj:
            return False
        t = obj[k]
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        if t.dim() != 4 or int(t.shape[0]) != 3:
            return False
    return True


def _pt_has_force_mask_tensor(obj: dict) -> bool:
    if "force_mask" not in obj or obj["force_mask"] is None:
        return False
    t = obj["force_mask"]
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.dim() == 4 and int(t.shape[0]) == 3


def _resample_chw(x: torch.Tensor, target_t: int) -> torch.Tensor:
    t = int(x.shape[1])
    T = int(target_t)
    if t == T:
        return x.float()
    idx = torch.linspace(0, t - 1, T).long()
    return x[:, idx].contiguous().float()


def _load_full_pack(obj: dict, target_t: int, img_size: int, path_hint: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _pt_full_pack(obj):
        raise ValueError(f"{path_hint}: 需要 version>=2 且含 rgb,stress,flow")
    outs = []
    for k in ("rgb", "stress", "flow"):
        t = _resample_chw(obj[k] if isinstance(obj[k], torch.Tensor) else torch.as_tensor(obj[k]), target_t)
        if int(t.shape[2]) != img_size or int(t.shape[3]) != img_size:
            raise ValueError(f"{path_hint} {k}: 空间应为 {img_size}")
        outs.append(t)
    return outs[0], outs[1], outs[2]


def _load_force_mask_from_pt(obj: dict, target_t: int, img_size: int, path_hint: str) -> torch.Tensor:
    if not _pt_has_force_mask_tensor(obj):
        raise ValueError(f"{path_hint}: .pt 中缺少有效的 force_mask [3,T,H,W]")
    t = obj["force_mask"] if isinstance(obj["force_mask"], torch.Tensor) else torch.as_tensor(obj["force_mask"])
    t = _resample_chw(t, target_t)
    if int(t.shape[2]) != img_size or int(t.shape[3]) != img_size:
        raise ValueError(f"{path_hint} force_mask: 空间应为 {img_size}")
    return t


def _tensor_views_ok(sample_dir: Path, subdir: str, max_views: int) -> bool:
    cache = sample_dir / subdir
    stems = list_arch4_tensor_stems(cache)[:max_views]
    if not stems:
        return False
    img_root = sample_dir / "images"
    sg = sample_dir / "stress_gaussian"
    fg = sample_dir / "flow_gaussian"
    fm_root = sample_dir / "force_mask"
    for vn in stems:
        p = cache / f"{vn}.pt"
        if not p.is_file():
            return False
        try:
            o = _torch_load_pt(p)
        except Exception:
            return False
        if isinstance(o, dict) and _pt_full_pack(o):
            if _pt_has_force_mask_tensor(o):
                continue
            fmd = fm_root / vn
            if fmd.is_dir() and _sorted_image_files(fmd):
                continue
            return False
        if not (img_root / vn).is_dir() or not _sorted_image_files(img_root / vn):
            return False
        if not (sg / vn).is_dir() or not (fg / vn).is_dir():
            return False
        if not (fm_root / vn).is_dir():
            return False
        if (
            not _sorted_image_files(sg / vn)
            or not _sorted_image_files(fg / vn)
            or not _sorted_image_files(fm_root / vn)
        ):
            return False
    return True


class DatasetArch4(Dataset):
    """images 单通道时序输入，stress/flow/force_mask 单通道时序监督 + ``gt.json`` 参数回归。"""

    def __init__(
        self,
        split_root: Path,
        img_size: int = 224,
        max_views: int = 4,
        source: str = "auto",
        input_mode: str = "rgb+force_mask",
        tensor_subdir: str = "arch4_tensors",
        lmdb_env_subdir: str = "arch4_data.lmdb",
        sample_pack_name: str = "sample_pack.npz",
        video_sample_fps: float = 30.0,
        num_frames: int = 30,
        preflight: bool = True,
        verbose: bool = True,
        sample_ids: Optional[List[str]] = None,
        return_sample_id: bool = False,
        return_object_mask: bool = False,
    ):
        self.split_root = Path(split_root)
        self.img_size = int(img_size)
        self.max_views = int(max_views)
        self.source = str(source).strip().lower() or "auto"
        self.input_mode = str(input_mode).strip().lower()
        self.tensor_subdir = str(tensor_subdir).strip().strip("/\\") or "arch4_tensors"
        self.lmdb_env_subdir = str(lmdb_env_subdir).strip().strip("/\\") or "arch4_data.lmdb"
        self.sample_pack_name = str(sample_pack_name).strip().strip("/\\") or "sample_pack.npz"
        self.video_sample_fps = float(video_sample_fps)
        self.num_frames = int(num_frames)
        self.sample_ids = list(sample_ids) if sample_ids is not None else None
        self.return_sample_id = bool(return_sample_id)
        self.return_object_mask = bool(return_object_mask)

        self.samples: List[Path] = []
        self.layout: Dict[str, str] = {}
        T, H = self.num_frames, self.img_size

        if self.sample_ids is None:
            iter_dirs = list_dataset400_sample_dirs(self.split_root)
        else:
            iter_dirs = [self.split_root / str(sid) for sid in self.sample_ids]

        for d in iter_dirs:
            if not d.is_dir():
                continue
            if not (d / "gt.json").is_file():
                continue
            lay = self._resolve_layout(d)
            if lay is None:
                continue
            if preflight and not self._preflight(d, lay, T, H, verbose):
                continue
            k = str(d.resolve())
            self.layout[k] = lay
            self.samples.append(d)

        if not self.samples:
            raise ValueError(
                f"DatasetArch4: 无可用样本（需 sample_pack / LMDB / 视频五元组 / PNG 四目录 / .pt）；source={self.source!r}"
            )

    def _has_sample_pack(self, d: Path) -> bool:
        return _sample_pack_exists(d, self.sample_pack_name)

    def _has_arch4_lmdb(self, d: Path) -> bool:
        return lmdb_arch4_is_valid(d / self.lmdb_env_subdir)

    def _has_png_quad(self, d: Path) -> bool:
        sg = d / "stress_gaussian"
        if not sg.is_dir():
            return False
        views = list_multiview_camera_dirs(sg)[: self.max_views]
        if not views:
            return False
        img_root, fg, fm_root = d / "images", d / "flow_gaussian", d / "force_mask"
        for vn in views:
            if not (img_root / vn).is_dir() or not _sorted_image_files(img_root / vn):
                return False
            if not (fg / vn).is_dir() or not _sorted_image_files(fg / vn):
                return False
            if not _sorted_image_files(sg / vn):
                return False
            if not (fm_root / vn).is_dir() or not _sorted_image_files(fm_root / vn):
                return False
        return True

    def _resolve_layout(self, d: Path) -> Optional[str]:
        has_p = self._has_sample_pack(d)
        has_m = self._has_arch4_lmdb(d)
        has_v = len(list_arch4_video_view_groups(d / "videos")) > 0
        has_t = _tensor_views_ok(d, self.tensor_subdir, self.max_views)
        has_i = self._has_png_quad(d)
        if self.source == "sample_pack":
            return "sample_pack" if has_p else None
        if self.source == "lmdb":
            return "lmdb" if has_m else None
        if self.source == "video":
            return "video" if has_v else None
        if self.source == "tensors":
            return "tensors" if has_t else None
        if self.source == "images":
            return "images" if has_i else None
        if has_p:
            return "sample_pack"
        if has_m:
            return "lmdb"
        if has_v:
            return "video"
        if has_t:
            return "tensors"
        if has_i:
            return "images"
        return None

    def _preflight(self, d: Path, lay: str, T: int, H: int, verbose: bool) -> bool:
        try:
            if lay == "sample_pack":
                read_sample_pack_arrays(
                    d,
                    sample_pack_name=self.sample_pack_name,
                    num_frames=T,
                    img_size=H,
                    max_views=self.max_views,
                )
            elif lay == "lmdb":
                read_arch4_lmdb_view_tensors(
                    d / self.lmdb_env_subdir,
                    num_frames=T,
                    img_size=H,
                    max_views=self.max_views,
                )
            elif lay == "video":
                for _b, pr, ps, pf, pfm in list_arch4_video_view_groups(d / "videos")[: self.max_views]:
                    extract_frames_from_video(pr, self.video_sample_fps, T, H)
                    extract_frames_from_video(ps, self.video_sample_fps, T, H)
                    extract_frames_from_video(pf, self.video_sample_fps, T, H)
                    extract_frames_from_video(pfm, self.video_sample_fps, T, H)
            elif lay == "tensors":
                cache = d / self.tensor_subdir
                for vn in list_arch4_tensor_stems(cache)[: self.max_views]:
                    p = cache / f"{vn}.pt"
                    o = _torch_load_pt(p)
                    if isinstance(o, dict) and _pt_full_pack(o):
                        _load_full_pack(o, T, H, str(p))
                        if _pt_has_force_mask_tensor(o):
                            _load_force_mask_from_pt(o, T, H, str(p))
                        else:
                            _stack_view_images(d / "force_mask" / vn, T, H)
                    else:
                        _stack_view_images(d / "images" / vn, T, H)
                        _stack_view_images(d / "stress_gaussian" / vn, T, H)
                        _stack_view_images(d / "flow_gaussian" / vn, T, H)
                        _stack_view_images(d / "force_mask" / vn, T, H)
            else:
                sg = d / "stress_gaussian"
                for vn in list_multiview_camera_dirs(sg)[: self.max_views]:
                    _stack_view_images(d / "images" / vn, T, H)
                    _stack_view_images(sg / vn, T, H)
                    _stack_view_images(d / "flow_gaussian" / vn, T, H)
                    _stack_view_images(d / "force_mask" / vn, T, H)
            return True
        except Exception as e:
            if verbose:
                print(f"[DatasetArch4] 跳过 {d}: {e}", flush=True)
            return False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, ...]:
        d = self.samples[idx]
        lay = self.layout[str(d.resolve())]
        T, H, mv = self.num_frames, self.img_size, self.max_views

        def thwc_tchw(a: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(np.transpose(a.astype(np.float32), (0, 3, 1, 2)))

        rgb_l, s_l, f_l, fm_l, om_l = [], [], [], [], []
        if lay == "sample_pack":
            arrays = read_sample_pack_arrays(
                d,
                sample_pack_name=self.sample_pack_name,
                num_frames=T,
                img_size=H,
                max_views=mv,
            )
            rgb = np.ascontiguousarray(arrays["rgb"], dtype=np.float32)
            s = np.ascontiguousarray(arrays["stress"], dtype=np.float32)
            f = np.ascontiguousarray(arrays["flow"], dtype=np.float32)
            fm = np.ascontiguousarray(arrays["force_mask"], dtype=np.float32)
            om = np.ascontiguousarray(arrays["object_mask"], dtype=np.float32)
        elif lay == "lmdb":
            arrays = read_lmdb_arrays_from_sample_dir(
                d,
                lmdb_env_subdir=self.lmdb_env_subdir,
                num_frames=T,
                img_size=H,
            )
            for a in arrays["rgb"][:mv]:
                rgb_l.append(np.ascontiguousarray(a))
            for a in arrays["stress"][:mv]:
                s_l.append(np.ascontiguousarray(a))
            for a in arrays["flow"][:mv]:
                f_l.append(np.ascontiguousarray(a))
            for a in arrays["force_mask"][:mv]:
                fm_l.append(np.ascontiguousarray(a))
            for a in arrays["object_mask"][:mv]:
                om_l.append(np.ascontiguousarray(a))
        elif lay == "video":
            for _b, pr, ps, pf, pfm in list_arch4_video_view_groups(d / "videos")[:mv]:
                rgb_l.append(thwc_tchw(extract_frames_from_video(pr, self.video_sample_fps, T, H)))
                s_l.append(thwc_tchw(extract_frames_from_video(ps, self.video_sample_fps, T, H)))
                f_l.append(thwc_tchw(extract_frames_from_video(pf, self.video_sample_fps, T, H)))
                fm_l.append(thwc_tchw(extract_frames_from_video(pfm, self.video_sample_fps, T, H)))
                obj_view = d / "object_mask" / _b
                if obj_view.is_dir() and _sorted_image_files(obj_view):
                    om_l.append(_stack_view_images(obj_view, T, H))
                else:
                    om_l.append(np.ones_like(fm_l[-1], dtype=np.float32))
        elif lay == "tensors":
            cache = d / self.tensor_subdir
            for vn in list_arch4_tensor_stems(cache)[:mv]:
                p = cache / f"{vn}.pt"
                o = _torch_load_pt(p)
                if isinstance(o, dict) and _pt_full_pack(o):
                    r, ss, ff = _load_full_pack(o, T, H, str(p))
                    rgb_l.append(np.ascontiguousarray(r.cpu().numpy()))
                    s_l.append(np.ascontiguousarray(ss.cpu().numpy()))
                    f_l.append(np.ascontiguousarray(ff.cpu().numpy()))
                    if _pt_has_force_mask_tensor(o):
                        fmm = _load_force_mask_from_pt(o, T, H, str(p))
                        fm_l.append(np.ascontiguousarray(fmm.cpu().numpy()))
                    else:
                        fm_l.append(_stack_view_images(d / "force_mask" / vn, T, H))
                    obj_view = d / "object_mask" / vn
                    if obj_view.is_dir() and _sorted_image_files(obj_view):
                        om_l.append(_stack_view_images(obj_view, T, H))
                    else:
                        om_l.append(np.ones_like(fm_l[-1], dtype=np.float32))
                else:
                    rgb_l.append(_stack_view_images(d / "images" / vn, T, H))
                    s_l.append(_stack_view_images(d / "stress_gaussian" / vn, T, H))
                    f_l.append(_stack_view_images(d / "flow_gaussian" / vn, T, H))
                    fm_l.append(_stack_view_images(d / "force_mask" / vn, T, H))
                    obj_view = d / "object_mask" / vn
                    if obj_view.is_dir() and _sorted_image_files(obj_view):
                        om_l.append(_stack_view_images(obj_view, T, H))
                    else:
                        om_l.append(np.ones_like(fm_l[-1], dtype=np.float32))
        else:
            sg = d / "stress_gaussian"
            for vn in list_multiview_camera_dirs(sg)[:mv]:
                rgb_l.append(_stack_view_images(d / "images" / vn, T, H))
                s_l.append(_stack_view_images(sg / vn, T, H))
                f_l.append(_stack_view_images(d / "flow_gaussian" / vn, T, H))
                fm_l.append(_stack_view_images(d / "force_mask" / vn, T, H))
                obj_view = d / "object_mask" / vn
                if obj_view.is_dir() and _sorted_image_files(obj_view):
                    om_l.append(_stack_view_images(obj_view, T, H))
                else:
                    om_l.append(np.ones_like(fm_l[-1], dtype=np.float32))

        if lay != "sample_pack":
            if not rgb_l:
                z3 = np.zeros((1, 3, T, H, H), np.float32)
                rgb = s = f = fm = z3
                om = np.ones_like(z3, dtype=np.float32)
            else:
                rgb = np.stack(rgb_l, 0)
                s = np.stack(s_l, 0)
                f = np.stack(f_l, 0)
                fm = np.stack(fm_l, 0)
                om = np.stack(om_l, 0)
        else:
            rgb = np.ascontiguousarray(rgb, dtype=np.float32)
            s = np.ascontiguousarray(s, dtype=np.float32)
            f = np.ascontiguousarray(f, dtype=np.float32)
            fm = np.ascontiguousarray(fm, dtype=np.float32)
            om = np.ascontiguousarray(om, dtype=np.float32)

        rgb, s, f, fm, om = (
            _pad_views_vcthw(rgb, mv),
            _pad_views_vcthw(s, mv),
            _pad_views_vcthw(f, mv),
            _pad_views_vcthw(fm, mv),
            _pad_views_vcthw(om, mv),
        )
        # LMDB/PNG里各模态是“彩色存储但三通道等值”，训练统一转为单通道时序。
        def _to_single_channel(vcthw: np.ndarray) -> np.ndarray:
            # 输入 [V,3,T,H,W] -> 输出 [V,1,T,H,W]，直接取第0通道避免重复信息和计算浪费。
            if vcthw.ndim != 5 or int(vcthw.shape[1]) < 1:
                raise ValueError(f"expect [V,C,T,H,W], got {vcthw.shape}")
            return np.ascontiguousarray(vcthw[:, 0:1, ...], dtype=np.float32)

        rgb = _to_single_channel(rgb)
        s = _to_single_channel(s)
        f = _to_single_channel(f)
        fm = _to_single_channel(fm)
        om = _to_single_channel(om)

        # 重新设计后输入固定为 images 时序（不再把 force_mask 叠到输入通道）
        x_in = rgb

        _, _, reg = _load_gt_from_dir(d)
        params = torch.tensor(
            [reg["E"], reg["nu"], reg["density"], reg["yield_stress"]], dtype=torch.float32
        )

        base_out = [
            torch.from_numpy(np.ascontiguousarray(x_in)),
            torch.from_numpy(np.ascontiguousarray(s)),
            torch.from_numpy(np.ascontiguousarray(f)),
            torch.from_numpy(np.ascontiguousarray(fm)),
        ]
        if self.return_object_mask:
            base_out.append(torch.from_numpy(np.ascontiguousarray(om)))
        base_out.append(params)

        if self.return_sample_id:
            return (*base_out, d.name)
        return tuple(base_out)
