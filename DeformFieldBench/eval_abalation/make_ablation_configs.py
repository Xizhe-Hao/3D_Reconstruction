from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval_abalation import paths


def _load_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config 顶层必须是 object: {path}")
    return raw


def _set_path(d: Dict[str, Any], key_path: List[str], value: Any) -> None:
    cur: Dict[str, Any] = d
    for key in key_path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[key_path[-1]] = value


def _train_output_root(exp_name: str) -> str:
    return str((paths.TRAIN_RUN_DIR / exp_name).resolve())


def _build_variants(base_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    variants: List[Tuple[str, Dict[str, Any]]] = []

    baseline = copy.deepcopy(base_cfg)
    baseline["_generated_by"] = "eval_abalation/make_ablation_configs.py"
    baseline["_variant_name"] = "logic_baseline"
    variants.append(("logic_baseline", baseline))

    no_field = copy.deepcopy(base_cfg)
    no_field["_variant_name"] = "logic_abl_no_field_aux"
    _set_path(no_field, ["train", "output_root"], _train_output_root("logic_abl_no_field_aux"))
    _set_path(no_field, ["train", "lambda_stress"], 0.0)
    _set_path(no_field, ["train", "lambda_flow"], 0.0)
    _set_path(no_field, ["train", "lambda_force"], 0.0)
    variants.append(("logic_abl_no_field_aux", no_field))

    no_stage = copy.deepcopy(base_cfg)
    no_stage["_variant_name"] = "logic_abl_no_multistage"
    _set_path(no_stage, ["train", "output_root"], _train_output_root("logic_abl_no_multistage"))
    _set_path(no_stage, ["train", "stage1_field_epochs"], 0)
    _set_path(no_stage, ["train", "stage_flow_force_epochs"], 0)
    _set_path(no_stage, ["train", "stage_stress_epochs"], 0)
    _set_path(no_stage, ["train", "stage_joint_epochs"], 0)
    variants.append(("logic_abl_no_multistage", no_stage))

    no_boundary = copy.deepcopy(base_cfg)
    no_boundary["_variant_name"] = "logic_abl_no_boundary"
    _set_path(no_boundary, ["train", "output_root"], _train_output_root("logic_abl_no_boundary"))
    _set_path(no_boundary, ["train", "lambda_stress_edge"], 0.0)
    _set_path(no_boundary, ["train", "lambda_flow_edge"], 0.0)
    variants.append(("logic_abl_no_boundary", no_boundary))

    for suffix, value in (("0001", 0.001), ("001", 0.01), ("01", 0.1)):
        plus_phys = copy.deepcopy(base_cfg)
        name = f"logic_plus_phys_{suffix}"
        plus_phys["_variant_name"] = name
        _set_path(plus_phys, ["train", "output_root"], _train_output_root(name))
        _set_path(plus_phys, ["train", "use_phys_loss"], True)
        _set_path(plus_phys, ["train", "lambda_phys"], float(value))
        variants.append((name, plus_phys))

    return variants


def generate_configs(
    *,
    baseline_config: Path = paths.LOGIC_BASELINE_CONFIG,
    out_dir: Path = paths.GENERATED_CONFIG_DIR,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = _load_json(baseline_config)
    saved: List[Path] = []
    for name, cfg in _build_variants(base_cfg):
        out_path = out_dir / f"{name}.json"
        out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(out_path)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser("Generate logic_model baseline/ablation configs")
    ap.add_argument("--baseline_config", type=str, default=str(paths.LOGIC_BASELINE_CONFIG))
    ap.add_argument("--out_dir", type=str, default=str(paths.GENERATED_CONFIG_DIR))
    args = ap.parse_args()
    saved = generate_configs(
        baseline_config=Path(args.baseline_config).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
    )
    print(json.dumps({"generated": [str(p) for p in saved]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
