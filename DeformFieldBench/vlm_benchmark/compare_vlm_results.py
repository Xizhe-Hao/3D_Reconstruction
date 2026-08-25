import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 eval_abalation 标准口径对 VLM 结果做参数评估与绘图。"
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
        help="要评估的模型标签（与 vlm_model_registry.py 中一致）",
    )
    args = parser.parse_args()
    root = Path(args.physgaussian_root).resolve()
    output_root = root / "vlm_benchmark" / "output" / args.vlm_tag
    if not output_root.is_dir():
        raise FileNotFoundError(f"VLM 输出目录不存在: {output_root}")

    cmd = [
        sys.executable,
        "-m",
        "eval_abalation.eval_vlm",
        "--vlm_tag",
        str(args.vlm_tag),
        "--phys_root",
        str(root),
        "--out_dir",
        str(output_root),
    ]
    subprocess.run(cmd, cwd=str(root), check=True)

    overall_csv = output_root / "param_metrics" / "overall.csv"
    classification_json = output_root / "classification.json"
    overall_scatter = output_root / "param_plots_grouped" / "overall_scatter.png"
    overall_heatmap = output_root / "param_plots_grouped" / "overall_heatmap.png"

    if overall_csv.is_file():
        print(f"参数指标已保存到: {overall_csv}")
    if classification_json.is_file():
        cls = json.loads(classification_json.read_text(encoding='utf-8'))
        print("=== Classification accuracy ===")
        print(f"Material accuracy: {float(cls.get('material_accuracy', 0.0)) * 100:.2f}%")
        print(f"Action   accuracy: {float(cls.get('action_accuracy', 0.0)) * 100:.2f}%")
    print(f"散点图已保存到: {overall_scatter}")
    print(f"热力图已保存到: {overall_heatmap}")

if __name__ == "__main__":
    main()