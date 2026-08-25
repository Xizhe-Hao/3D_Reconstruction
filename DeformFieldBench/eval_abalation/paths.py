from __future__ import annotations

from pathlib import Path


EVAL_ABLATION_ROOT = Path(__file__).resolve().parent
PHYS_ROOT = EVAL_ABLATION_ROOT.parent

DATASET_ROOT = PHYS_ROOT / "auto_output" / "dataset_5000"
DATASET_TRAIN_ROOT = DATASET_ROOT / "train"
SPLIT_JSON = DATASET_ROOT / "train_test_split.cleaned.json"

LOGIC_BASELINE_CONFIG = PHYS_ROOT / "logic_model" / "configs" / "final_logic_413_5000_new.json"
LOGIC_BASELINE_OUTPUT = PHYS_ROOT / "logic_model" / "output" / "output_413_5000_new"
LOGIC_BASELINE_CHECKPOINT_DIR = LOGIC_BASELINE_OUTPUT / "checkpoints"
MY_MODEL_BASE_CONFIG = PHYS_ROOT / "my_model" / "configs.json"

GENERATED_CONFIG_DIR = EVAL_ABLATION_ROOT / "generated_configs"
OUTPUT_ROOT = EVAL_ABLATION_ROOT / "outputs"

BASELINE_SCAN_DIR = OUTPUT_ROOT / "logic_model" / "baseline" / "checkpoint_scan"
BASELINE_MAIN_DIR = OUTPUT_ROOT / "logic_model" / "baseline" / "main"
LOGIC_ABLATION_DIR = OUTPUT_ROOT / "logic_model" / "ablations"
LOGIC_EXTENSION_DIR = OUTPUT_ROOT / "logic_model" / "extensions"
TRAIN_RUN_DIR = OUTPUT_ROOT / "train_runs"
COMPARE_DIR = OUTPUT_ROOT / "comparison"
COMPARE_CONFIG_DIR = EVAL_ABLATION_ROOT / "generated_compare_configs"
MY_MODEL_COMPARE_DIR = COMPARE_DIR / "my_model"
VLM_COMPARE_DIR = COMPARE_DIR / "vlm"
FINAL_TABLE_DIR = OUTPUT_ROOT / "final_tables"

USAGE_DOC = EVAL_ABLATION_ROOT / "使用说明.md"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def baseline_scan_out_dir(ckpt_name: str) -> Path:
    return BASELINE_SCAN_DIR / ckpt_name


def ablation_eval_out_dir(exp_name: str) -> Path:
    return LOGIC_ABLATION_DIR / exp_name


def my_model_compare_out_dir(tag: str) -> Path:
    return MY_MODEL_COMPARE_DIR / tag


def vlm_compare_out_dir(tag: str) -> Path:
    return VLM_COMPARE_DIR / tag
