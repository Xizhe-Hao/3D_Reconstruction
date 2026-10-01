#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -f /etc/network_turbo ]]; then
  set +u
  source /etc/network_turbo
  set -u
fi
git submodule update --init -- submodule/depth-anything-3
expected_commit=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
[[ "$(git -C submodule/depth-anything-3 rev-parse HEAD)" == "$expected_commit" ]] || { echo 'DA3 source commit mismatch' >&2; exit 1; }
# Reuse matching base dependencies, but keep clean CUDA wheels inside this venv.
"${DA3_BASE_PYTHON:-python}" -m venv --system-site-packages .venv-da3
if ! command -v uv >/dev/null 2>&1; then
  .venv-da3/bin/python -m pip install --index-url "${DA3_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}" uv
  export PATH="$project_dir/.venv-da3/bin:$PATH"
fi
export DA3_PIP_INDEX="${DA3_PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
.venv-da3/bin/python - <<'DEPS'
import importlib.metadata as metadata
import os
import subprocess
import sys
from pathlib import Path
root = Path.cwd()
todo = []
for line in (root / 'scripts/requirements_da3.lock').read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    name, version = line.split('==')
    try:
        dist = metadata.distribution(name)
        matches = dist.version == version
        if name in ('torch', 'torchvision'):
            matches = matches and Path(dist.locate_file('')).is_relative_to(Path(sys.prefix))
    except metadata.PackageNotFoundError:
        matches = False
    if not matches:
        todo.append(line)
command = ['uv', 'pip', 'install', '--cache-dir', str(root / '.cache/uv'),
           '--python', sys.executable, '--index-url', os.environ['DA3_PIP_INDEX'], '--no-deps']
if todo:
    subprocess.run(command + todo, check=True)
subprocess.run(command + ['-e', str(root / 'submodule/depth-anything-3')], check=True)
DEPS
export HF_HOME="$project_dir/.cache/huggingface"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_XET=1
.venv-da3/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from depth_anything_3.api import DepthAnything3
path = snapshot_download('depth-anything/DA3-GIANT', revision='7cd62ae9315b9dff094d2d300e4ad012640607dd',
                         local_dir='outputs/models/DA3-GIANT', allow_patterns=['config.json', 'model.safetensors'])
model = DepthAnything3.from_pretrained(path)
print('DA3 GIANT checkpoint loaded:', sum(p.numel() for p in model.parameters()), 'parameters')
PY
