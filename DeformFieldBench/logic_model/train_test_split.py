# -*- coding: utf-8 -*-
"""
按 ``gt.json`` 中的 ``object`` 字段划分 train / test：

- **test**：``object`` 属于用户指定的若干类型（字符串与 ``gt.json`` 完全一致）。
- **train**：其余所有样本。

默认只统计同时含 ``sample_pack.npz`` 与 ``gt.json`` 的目录；
也可通过参数切换为 ``lmdb`` 或 ``auto``。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _load_object(gt_path: Path) -> str:
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return ""
    return str(data.get("object", "")).strip()


def _normalize_sample_storage_backend(v: str | None) -> str:
    x = str(v or "sample_pack").strip().lower()
    return x if x in ("sample_pack", "lmdb", "auto") else "sample_pack"


def _sample_matches_storage(
    sample_dir: Path,
    *,
    sample_storage_backend: str,
    sample_pack_name: str,
    lmdb_env_subdir: str,
) -> bool:
    backend = _normalize_sample_storage_backend(sample_storage_backend)
    pack_ok = (sample_dir / sample_pack_name).is_file()
    lmdb_env = sample_dir / lmdb_env_subdir
    lmdb_ok = lmdb_env.is_dir() and (lmdb_env / "data.mdb").is_file()
    if backend == "sample_pack":
        return pack_ok
    if backend == "lmdb":
        return lmdb_ok
    return pack_ok or lmdb_ok


def _iter_sample_dirs(
    split_root: Path,
    *,
    require_storage: bool,
    sample_storage_backend: str,
    sample_pack_name: str,
    lmdb_env_subdir: str,
) -> List[Path]:
    split_root = split_root.resolve()
    if not split_root.is_dir():
        raise ValueError(f"split_root 不是目录: {split_root}")

    out: List[Path] = []
    for d in sorted(split_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        if not (d / "gt.json").is_file():
            continue
        if require_storage and not _sample_matches_storage(
            d,
            sample_storage_backend=sample_storage_backend,
            sample_pack_name=sample_pack_name,
            lmdb_env_subdir=lmdb_env_subdir,
        ):
            continue
        out.append(d.resolve())
    return out


def split_by_test_objects(
    split_root: Path,
    test_objects: Set[str],
    *,
    require_storage: bool = True,
    sample_storage_backend: str = "sample_pack",
    sample_pack_name: str = "sample_pack.npz",
    lmdb_env_subdir: str = "arch4_data.lmdb",
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Returns:
        train_ids, test_ids, sample_id -> object string
    """
    if not test_objects:
        raise ValueError("test_objects 不能为空")

    dirs = _iter_sample_dirs(
        split_root,
        require_storage=require_storage,
        sample_storage_backend=sample_storage_backend,
        sample_pack_name=sample_pack_name,
        lmdb_env_subdir=lmdb_env_subdir,
    )
    if not dirs:
        raise ValueError(f"未找到可用样本目录: {split_root}")

    id_to_obj: Dict[str, str] = {}
    for d in dirs:
        obj = _load_object(d / "gt.json")
        id_to_obj[d.name] = obj

    train_ids: List[str] = []
    test_ids: List[str] = []
    for sid, obj in sorted(id_to_obj.items(), key=lambda x: x[0]):
        if obj in test_objects:
            test_ids.append(sid)
        else:
            train_ids.append(sid)

    return train_ids, test_ids, id_to_obj


def select_random_test_objects(
    split_root: Path,
    *,
    test_object_count: int,
    require_storage: bool = True,
    sample_storage_backend: str = "sample_pack",
    sample_pack_name: str = "sample_pack.npz",
    lmdb_env_subdir: str = "arch4_data.lmdb",
    random_seed: int = 42,
) -> Set[str]:
    """
    从 split_root 的样本中读取 gt.json.object，随机抽取 N 个 object 类型作为 test。
    """
    n = int(test_object_count)
    if n <= 0:
        raise ValueError("test_object_count 必须 > 0")

    dirs = _iter_sample_dirs(
        split_root,
        require_storage=require_storage,
        sample_storage_backend=sample_storage_backend,
        sample_pack_name=sample_pack_name,
        lmdb_env_subdir=lmdb_env_subdir,
    )
    if not dirs:
        raise ValueError(f"未找到可用样本目录: {split_root}")

    all_objects: Set[str] = set()
    for d in dirs:
        obj = _load_object(d / "gt.json")
        if obj:
            all_objects.add(obj)

    uniq = sorted(all_objects)
    if not uniq:
        raise ValueError("未在 gt.json 中解析到有效 object 字段")
    if n > len(uniq):
        raise ValueError(
            f"test_object_count={n} 超过 object 类型总数 {len(uniq)}"
        )

    rng = random.Random(int(random_seed))
    chosen = set(rng.sample(uniq, n))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(
        description="按 gt.json 的 object 划分 train/test（test 仅含指定 object 类型）"
    )
    ap.add_argument(
        "--split_root",
        type=str,
        required=True,
        help="数据集 split 根目录，如 auto_output/dataset_mask_1000/train",
    )
    ap.add_argument(
        "--test_objects",
        type=str,
        nargs="+",
        default=None,
        help="用于 test 的 object 字符串（与 gt.json 中 object 字段完全一致），可多个",
    )
    ap.add_argument(
        "--test_object_count",
        type=int,
        default=None,
        help="随机抽取 N 个 object 类型作为 test（只给数量即可）",
    )
    ap.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="随机抽样 object 的种子（用于复现）",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default="logic_model/train_test_split_by_object.json",
        help="输出 JSON 路径（相对当前工作目录或绝对路径）",
    )
    ap.add_argument(
        "--no_require_storage",
        "--no_require_lmdb",
        dest="no_require_storage",
        action="store_true",
        help="不要求 sample_pack / lmdb，仅要求子目录下存在 gt.json",
    )
    ap.add_argument(
        "--sample_storage_backend",
        type=str,
        default="sample_pack",
        choices=("sample_pack", "lmdb", "auto"),
        help="样本存储形式：sample_pack / lmdb / auto（任一存在即可）",
    )
    ap.add_argument(
        "--sample_pack_name",
        type=str,
        default="sample_pack.npz",
        help="样本目录下 sample_pack 文件名",
    )
    ap.add_argument(
        "--lmdb_env_subdir",
        type=str,
        default="arch4_data.lmdb",
        help="LMDB 环境子目录名（sample_storage_backend=lmdb/auto 时使用）",
    )
    args = ap.parse_args()

    split_root = Path(args.split_root).expanduser().resolve()
    test_set_cli = {
        str(x).strip() for x in (args.test_objects or []) if str(x).strip()
    }
    n_test_obj = args.test_object_count
    if (not test_set_cli) and (n_test_obj is None):
        raise SystemExit("请至少提供 --test_objects 或 --test_object_count")
    if test_set_cli and (n_test_obj is not None):
        raise SystemExit("--test_objects 与 --test_object_count 只能二选一")

    if n_test_obj is not None:
        test_set = select_random_test_objects(
            split_root,
            test_object_count=int(n_test_obj),
            require_storage=not bool(args.no_require_storage),
            sample_storage_backend=str(args.sample_storage_backend).strip() or "sample_pack",
            sample_pack_name=str(args.sample_pack_name).strip() or "sample_pack.npz",
            lmdb_env_subdir=str(args.lmdb_env_subdir).strip() or "arch4_data.lmdb",
            random_seed=int(args.random_seed),
        )
    else:
        test_set = test_set_cli

    train_ids, test_ids, id_to_obj = split_by_test_objects(
        split_root,
        test_set,
        require_storage=not bool(args.no_require_storage),
        sample_storage_backend=str(args.sample_storage_backend).strip() or "sample_pack",
        sample_pack_name=str(args.sample_pack_name).strip() or "sample_pack.npz",
        lmdb_env_subdir=str(args.lmdb_env_subdir).strip() or "arch4_data.lmdb",
    )

    if not test_ids:
        raise SystemExit(
            f"test 集合为空：没有任何样本的 object 属于 {sorted(test_set)}。"
            "请检查拼写是否与 gt.json 一致，或尝试 --no_require_storage。"
        )

    obj_counter = Counter(id_to_obj.values())
    test_counter = Counter(id_to_obj[s] for s in test_ids)
    train_counter = Counter(id_to_obj[s] for s in train_ids)

    out = {
        "split_root": str(split_root),
        "mode": "by_object",
        "test_objects": sorted(test_set),
        "selection": (
            {"type": "random_n_objects", "n": int(n_test_obj), "random_seed": int(args.random_seed)}
            if n_test_obj is not None
            else {"type": "manual_list"}
        ),
        "require_storage": not bool(args.no_require_storage),
        "sample_storage_backend": str(args.sample_storage_backend).strip() or "sample_pack",
        "sample_pack_name": str(args.sample_pack_name).strip() or "sample_pack.npz",
        "lmdb_env_subdir": str(args.lmdb_env_subdir).strip() or "arch4_data.lmdb",
        "train_ids": train_ids,
        "test_ids": test_ids,
        "stats": {
            "n_samples": len(id_to_obj),
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "object_counts_all": dict(sorted(obj_counter.items(), key=lambda x: (-x[1], x[0]))),
            "object_counts_train": dict(sorted(train_counter.items(), key=lambda x: (-x[1], x[0]))),
            "object_counts_test": dict(sorted(test_counter.items(), key=lambda x: (-x[1], x[0]))),
        },
    }

    out_path = Path(args.out_json).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[OK] split_root={split_root}\n"
        f"     test_objects={sorted(test_set)}\n"
        f"     train={len(train_ids)} test={len(test_ids)} (total samples={len(id_to_obj)})\n"
        f"     -> {out_path}"
    )


if __name__ == "__main__":
    main()
