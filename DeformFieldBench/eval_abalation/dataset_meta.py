from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List


PARAM_KEYS = ("E", "nu", "density", "yield_stress")


@dataclass(frozen=True)
class SampleMeta:
    sample_id: str
    object_name: str
    material: str
    material_group: str
    action: str
    E: float
    nu: float
    density: float
    yield_stress: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _safe_float(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def normalize_material(material: str) -> str:
    raw = str(material).strip().lower()
    if raw in ("jelly", "metal", "plasticine"):
        return raw
    if "jelly" in raw:
        return "jelly"
    if "metal" in raw:
        return "metal"
    if "plastic" in raw:
        return "plasticine"
    return raw or "unknown"


def load_sample_meta(sample_dir: Path) -> SampleMeta:
    gt_path = sample_dir / "gt.json"
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"gt.json 顶层必须是 object: {gt_path}")
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    reg = raw.get("regression") if isinstance(raw.get("regression"), dict) else {}

    def _pick_param(key: str) -> float:
        if key in reg:
            return _safe_float(reg.get(key))
        return _safe_float(params.get(key))

    object_name = str(raw.get("object", "")).strip() or "unknown"
    action = str(raw.get("action", "")).strip() or "unknown"
    material = str(params.get("material", raw.get("material", ""))).strip() or "unknown"
    return SampleMeta(
        sample_id=sample_dir.name,
        object_name=object_name,
        material=material,
        material_group=normalize_material(material),
        action=action,
        E=_pick_param("E"),
        nu=_pick_param("nu"),
        density=_pick_param("density"),
        yield_stress=_pick_param("yield_stress"),
    )


def load_metadata_map(train_root: Path, sample_ids: Iterable[str]) -> Dict[str, SampleMeta]:
    out: Dict[str, SampleMeta] = {}
    for sid in sample_ids:
        sample_dir = train_root / str(sid)
        if not sample_dir.is_dir():
            continue
        out[str(sid)] = load_sample_meta(sample_dir)
    return out


def metadata_rows(meta_map: Dict[str, SampleMeta]) -> List[Dict[str, object]]:
    return [meta.to_dict() for _, meta in sorted(meta_map.items(), key=lambda kv: kv[0])]
