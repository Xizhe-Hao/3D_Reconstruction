#!/usr/bin/env bash
set -euo pipefail

export PIP_INDEX_URL="${FOUNDATIONSTEREO_PIP_INDEX:-https://pypi.org/simple}"

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
foundation_dir="${project_dir}/submodule/FoundationStereo"
environment_name="${FOUNDATIONSTEREO_ENV:-foundation_stereo}"
checkpoint_dir="${foundation_dir}/pretrained_models/23-51-11"
checkpoint_path="${checkpoint_dir}/model_best_bp2.pth"
cfg_file_id="1tidGICH1_kTUUqi42aboKscuMY4IK_Xr"
checkpoint_file_id="1Yh_2o9QCUrVqZrnAXZ7RUr0zTp3JrMKe"
expected_commit="6e8806816b533e4d13ddbb95ffa907b797060a62"

git -C "${project_dir}" submodule update --init -- submodule/FoundationStereo
actual_commit="$(git -C "${foundation_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  echo "FoundationStereo commit mismatch: expected ${expected_commit}, got ${actual_commit}" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${environment_name}"; then
  conda env create -n "${environment_name}" -f "${foundation_dir}/environment.yml"
fi
if ! conda run -n "${environment_name}" python -c "import flash_attn" >/dev/null 2>&1; then
  conda run -n "${environment_name}" python -m pip install flash-attn --no-build-isolation
fi
if ! conda run -n "${environment_name}" python -c "import pandas" >/dev/null 2>&1; then
  conda run -n "${environment_name}" python -m pip install pandas
fi

if [[ ! -f "${checkpoint_path}" || ! -f "${checkpoint_dir}/cfg.yaml" ]]; then
  mkdir -p "${checkpoint_dir}"
  if [[ ! -s "${checkpoint_dir}/cfg.yaml" ]]; then
    conda run -n "${environment_name}" gdown "${cfg_file_id}" -O "${checkpoint_dir}/cfg.yaml"
  fi
  if [[ ! -s "${checkpoint_path}" ]]; then
    conda run -n "${environment_name}" gdown "${checkpoint_file_id}" -O "${checkpoint_path}"
  fi
fi
if [[ ! -s "${checkpoint_path}" || ! -s "${checkpoint_dir}/cfg.yaml" ]]; then
  echo "FoundationStereo ViT-L checkpoint download is incomplete under ${checkpoint_dir}" >&2
  exit 1
fi

PYTHONPATH="${foundation_dir}:${PYTHONPATH:-}" conda run -n "${environment_name}" python -c \
  "from core.foundation_stereo import FoundationStereo; from omegaconf import OmegaConf; cfg=OmegaConf.load('${checkpoint_dir}/cfg.yaml'); cfg.setdefault('vit_size', 'vitl'); model=FoundationStereo(cfg); import torch; state=torch.load('${checkpoint_path}', map_location='cpu', mmap=True, weights_only=False); model.load_state_dict(state['model']); print('FoundationStereo checkpoint OK')"
echo "FoundationStereo ${actual_commit} is ready: ${checkpoint_path}"
