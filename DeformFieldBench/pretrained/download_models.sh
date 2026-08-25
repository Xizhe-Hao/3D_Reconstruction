#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
from pathlib import Path
import urllib.request
import yaml

manifest = yaml.safe_load(Path("pretrained/models.yaml").read_text(encoding="utf-8"))
for name, item in manifest["models"].items():
    dst = Path(item["file"])
    url = str(item["url"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        print(f"[skip] {name}: {dst}")
        continue
    print(f"[download] {name}: {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)
PY
