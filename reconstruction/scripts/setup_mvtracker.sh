#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mvtracker_dir="${project_dir}/submodule/mvtracker"
git -C "${project_dir}" submodule update --init --recursive -- submodule/mvtracker
if ! conda env list | awk '{print $1}' | grep -qx mvtracker; then
    conda create -n mvtracker python=3.10.12 -y
fi
conda install -n mvtracker pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -n mvtracker --override-channels -c https://repo.anaconda.com/pkgs/main mkl=2023.1.0 intel-openmp=2023.1.0 -y
conda run -n mvtracker python -m pip install -r "${mvtracker_dir}/requirements.txt"
if ! git -C "${mvtracker_dir}/../MoGe" rev-parse --git-dir > /dev/null 2>&1; then
    git clone https://github.com/microsoft/MoGe.git "${mvtracker_dir}/../MoGe"
fi
git -C "${mvtracker_dir}/../MoGe" checkout 0286b495230a074aadf1c76cc5c679e943e5d1c6
conda run -n mvtracker python -m pip install --no-deps -e "${mvtracker_dir}/../MoGe"
conda run -n mvtracker python -m pip install --no-deps "git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"
conda run -n mvtracker python -m pip install trimesh==4.5.1 plyfile==1.0.3 gradio==4.44.1 fastapi==0.112.2 starlette==0.38.6 pydantic==2.9.2 moderngl==5.12.0 glcontext==3.0.0 click
conda run -n mvtracker python -m pip install --force-reinstall --no-deps numpy==1.24.3
conda run -n mvtracker python "${project_dir}/scripts/download_assets.py" --with-moge2
