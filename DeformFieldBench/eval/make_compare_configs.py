from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

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


def generate_compare_configs(
    *,
    my_model_base_config: Path = paths.MY_MODEL_BASE_CONFIG,
    out_dir: Path = paths.COMPARE_CONFIG_DIR,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_json(my_model_base_config)
    my_cfg = copy.deepcopy(cfg)
    _set_path(my_cfg, ["data", "split_root"], "auto_output/dataset_5000/train")
    _set_path(my_cfg, ["data", "auto_output"], "auto_output")
    _set_path(my_cfg, ["train", "train_ids_json"], str(paths.SPLIT_JSON))
    _set_path(my_cfg, ["train", "quick_eval", "enabled"], False)
    _set_path(my_cfg, ["train", "device"], "cuda")
    _set_path(my_cfg, ["_generated_by"], "eval_abalation/make_compare_configs.py")
    my_out = out_dir / "my_model_compare.json"
    my_out.write_text(json.dumps(my_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"my_model_compare": my_out}


def main() -> None:
    ap = argparse.ArgumentParser("Generate comparison configs")
    ap.add_argument("--my_model_base_config", type=str, default=str(paths.MY_MODEL_BASE_CONFIG))
    ap.add_argument("--out_dir", type=str, default=str(paths.COMPARE_CONFIG_DIR))
    args = ap.parse_args()
    saved = generate_compare_configs(
        my_model_base_config=Path(args.my_model_base_config).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
    )
    print(json.dumps({k: str(v) for k, v in saved.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
