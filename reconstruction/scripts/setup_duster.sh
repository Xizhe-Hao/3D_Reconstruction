#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mvtracker_dir="${project_dir}/submodule/mvtracker"
duster_dir="${mvtracker_dir}/../duster"
checkpoint_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
checkpoint_url="https://download.europe.naverlabs.com/ComputerVision/DUSt3R/${checkpoint_name}"
checkpoint_md5="c3fab9b455b03f23d20e6bf77f2607bb"
checkpoint_path="${duster_dir}/checkpoints/${checkpoint_name}"

if ! conda env list | awk '{print $1}' | grep -qx mvtracker; then
    echo "Missing conda environment 'mvtracker'. Run scripts/setup_mvtracker.sh first." >&2
    exit 1
fi
git -C "${project_dir}" submodule update --init --recursive -- submodule/duster
duster_commit="$(git -C "${duster_dir}" rev-parse HEAD)"

# MVTracker's official helper only needs this package beyond its existing env.
conda run -n mvtracker python -m pip install roma==1.5.1
mkdir -p "${duster_dir}/checkpoints"
if [[ ! -f "${checkpoint_path}" ]] || ! echo "${checkpoint_md5}  ${checkpoint_path}" | md5sum --check --status; then
    wget -c "${checkpoint_url}" -P "${duster_dir}/checkpoints"
fi
echo "${checkpoint_md5}  ${checkpoint_path}" | md5sum --check

PYTHONPATH="${duster_dir}:${PYTHONPATH:-}" conda run -n mvtracker python -c \
    "from dust3r.model import AsymmetricCroCo3DStereo; from dust3r.cloud_opt import global_aligner; print('DUSt3R import OK')"
echo "DUSt3R ${duster_commit} is ready: ${checkpoint_path}"
