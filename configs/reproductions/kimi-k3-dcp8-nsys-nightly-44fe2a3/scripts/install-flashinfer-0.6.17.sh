#!/bin/bash

set -euo pipefail

node_name="${SLURMD_NODENAME:-${HOSTNAME:-local}}"
lock_dir="${SRT_FLASHINFER_INSTALL_LOCK_DIR:-/logs/.runtime-setup/${node_name}/flashinfer-0.6.17}"
lock_file="${lock_dir}/install.lock"
complete_file="${lock_dir}/install.complete"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/srt-flashinfer-uv-${SLURM_JOB_ID:-local}-${node_name}}"
mkdir -p "${lock_dir}"
exec 200>"${lock_file}"
flock -x 200

verify_versions() {
    local package
    for package in flashinfer-python flashinfer-cubin flashinfer-jit-cache; do
        uv pip show --system "${package}" \
            | grep -E '^Version: 0\.6\.17([+.]|$)'
    done
}

if [[ -f "${complete_file}" ]] && verify_versions; then
    exit 0
fi

uv pip install --system --no-deps --reinstall \
    "flashinfer-python==0.6.17"
uv pip install --system --no-deps --reinstall \
    "flashinfer-cubin==0.6.17" \
    --index-url https://flashinfer.ai/whl/
uv pip install --system --no-deps --reinstall \
    "flashinfer-jit-cache==0.6.17" \
    --index-url https://flashinfer.ai/whl/cu130

verify_versions
touch "${complete_file}"
