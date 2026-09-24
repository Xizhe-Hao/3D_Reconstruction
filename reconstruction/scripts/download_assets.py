"""Download the pretrained MVTracker checkpoint and optionally warm MoGe-2 cache."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).resolve().parents[1] / "submodule/mvtracker/checkpoints")
    parser.add_argument("--with-moge2", action="store_true")
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cached = Path(hf_hub_download("ethz-vlg/mvtracker", "mvtracker_200000_june2025.pth", token=False))
    target = args.checkpoint_dir / cached.name
    if not target.exists():
        shutil.copy2(cached, target)
    print(f"MVTracker checkpoint: {target.resolve()}")
    if args.with_moge2:
        from moge.model.v2 import MoGeModel

        MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal", token=False)
        print("MoGe-2 cache is ready: Ruicheng/moge-2-vitl-normal")


if __name__ == "__main__":
    main()
