# -*- coding: utf-8 -*-
"""
对 `auto_output/<dataset>/{train,test}/<id>/` 这种扁平样本目录做 train/test 划分。

注意：
- 该脚本只负责对目录名 `<id>` 做随机划分，不会触碰磁盘上的样本内容。
- DatasetArch4 现在支持通过 `sample_ids` 直接读取子集，因此评估时无需生成新的物理目录。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

from .dataset import list_dataset400_sample_dirs


def train_test_split_ids(
    split_root: Path, *, test_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str]]:
    split_root = Path(split_root)
    ids = [d.name for d in list_dataset400_sample_dirs(split_root)]
    if not ids:
        raise ValueError(f"split_root 下未发现数字样本目录: {split_root}")

    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio 必须在 (0,1) 内，当前: {test_ratio}")

    rng = random.Random(int(seed))
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * float(test_ratio))))
    test_ids = ids[:n_test]
    train_ids = ids[n_test:]
    return train_ids, test_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Arch4 扁平样本 train/test 划分")
    ap.add_argument("--split_root", type=str, required=True, help="样本根目录，如 auto_output/.../train")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="train_test_split.json")
    args = ap.parse_args()

    split_root = Path(args.split_root).expanduser().resolve()
    train_ids, test_ids = train_test_split_ids(split_root, test_ratio=args.test_ratio, seed=args.seed)

    out = {
        "split_root": str(split_root),
        "seed": int(args.seed),
        "test_ratio": float(args.test_ratio),
        "train_ids": train_ids,
        "test_ids": test_ids,
    }

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] train={len(train_ids)} test={len(test_ids)} -> {out_path}")


if __name__ == "__main__":
    main()

