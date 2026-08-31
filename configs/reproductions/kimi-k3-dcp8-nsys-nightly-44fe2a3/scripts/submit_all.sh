#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRT_SLURM_ROOT="$(cd "${BUNDLE_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${1:-${SRT_SLURM_ROOT}/output_nsys_reproduction/nightly-44fe2a3}"

: "${KIMI_K3_MXFP4_MODEL:?set KIMI_K3_MXFP4_MODEL to the local moonshotai/Kimi-K3 snapshot}"
: "${KIMI_K3_NVFP4_MODEL:?set KIMI_K3_NVFP4_MODEL to the local nvidia/Kimi-K3-NVFP4 snapshot}"
: "${KIMI_K3_DSPARK_MODEL:?set KIMI_K3_DSPARK_MODEL to the local Inferact/Kimi-K3-DSpark snapshot}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
unset ALL_PROXY all_proxy NO_PROXY no_proxy
unset GIT_HTTP_PROXY GIT_HTTPS_PROXY
unset VIRTUAL_ENV PYTHONPATH PYTHONHOME

"${SCRIPT_DIR}/verify_bundle.sh"

mapfile -t configs < <(find "${BUNDLE_DIR}" -maxdepth 1 -name '*.yaml' -type f | sort)

for config in "${configs[@]}"; do
    uv run --project "${SRT_SLURM_ROOT}" --python 3.12 --no-sync \
        srtctl dry-run -f "${config}" >/dev/null
done

for config in "${configs[@]}"; do
    name="$(sed -n '1s/^name: "\(.*\)"$/\1/p' "${config}")"
    if [[ -z "${name}" ]]; then
        echo "failed to read name from ${config}" >&2
        exit 1
    fi
    output_dir="${OUTPUT_ROOT}/${name}"
    echo "submitting ${name} -> ${output_dir}"
    uv run --project "${SRT_SLURM_ROOT}" --python 3.12 --no-sync \
        srtctl apply -f "${config}" -o "${output_dir}"
done
