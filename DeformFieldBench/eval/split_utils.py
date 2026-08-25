from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SplitInfo:
    split_root: Path
    train_ids: List[str]
    test_ids: List[str]
    raw: Dict[str, object]

    @property
    def split_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for sid in self.train_ids:
            out[sid] = "train"
        for sid in self.test_ids:
            out[sid] = "test"
        return out


def load_split_info(path: Path) -> SplitInfo:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"split json 顶层必须是 object: {path}")
    split_root = Path(str(raw.get("split_root") or "")).expanduser().resolve()
    train_ids = [str(x) for x in (raw.get("train_ids") or [])]
    test_ids = [str(x) for x in (raw.get("test_ids") or [])]
    return SplitInfo(
        split_root=split_root,
        train_ids=train_ids,
        test_ids=test_ids,
        raw=raw,
    )


def pick_eval_ids(split_info: SplitInfo, eval_split: str) -> List[str]:
    key = str(eval_split).strip().lower()
    if key == "train":
        return list(split_info.train_ids)
    if key == "test":
        return list(split_info.test_ids)
    raise ValueError(f"eval_split must be train/test, got {eval_split!r}")
