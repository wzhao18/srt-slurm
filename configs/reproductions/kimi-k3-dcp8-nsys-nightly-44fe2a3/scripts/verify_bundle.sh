#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="vllm/vllm-openai:nightly-44fe2a392b71d52a8d72faf2f8278834379482c9"
CORPUS_SHA256="e0e13a826912a4a81bb3a582aa73c4af0675bdeee6ddf6d505efb63d562d496f"
NSYS_HOST_ROOT="${NSYS_HOST_ROOT:-/cm/shared/apps/nvidia/nsight-systems-cli/2025.4.1}"

if [[ ! -x "${NSYS_HOST_ROOT}/bin/nsys" ]]; then
    echo "nsys is unavailable at ${NSYS_HOST_ROOT}/bin/nsys" >&2
    exit 1
fi

echo "${CORPUS_SHA256}  ${BUNDLE_DIR}/assets/shakespeare.txt" | sha256sum --check
bash -n "${BUNDLE_DIR}/scripts/profile_decode.sh"
bash -n "${BUNDLE_DIR}/scripts/submit_all.sh"

mapfile -t configs < <(find "${BUNDLE_DIR}" -maxdepth 1 -name '*.yaml' -type f | sort)
if [[ "${#configs[@]}" -ne 4 ]]; then
    echo "expected four configs, found ${#configs[@]}" >&2
    exit 1
fi

for config in "${configs[@]}"; do
    image_count="$(grep -F -c "${IMAGE}" "${config}")"
    if [[ "${image_count}" -ne 2 ]]; then
        echo "${config}: expected two pinned image declarations" >&2
        exit 1
    fi
    grep -Fq 'version: "1.2.1"' "${config}"
    grep -Fq 'decode-context-parallel-size: 8' "${config}"
    grep -Fq 'tensor-parallel-size: 8' "${config}"
    grep -Fq 'cp-kv-cache-interleave-size: 1' "${config}"
    grep -Fq 'type: "nsys"' "${config}"
    grep -Fq '${NSYS_HOST_ROOT}:/opt/nsight-systems' "${config}"
    grep -Fq 'PROFILE_ISL: "131072"' "${config}"
    if grep -Fq 'router-session-affinity-ttl-secs' "${config}"; then
        echo "${config}: ai-dynamo 1.2.1 does not accept router-session-affinity-ttl-secs" >&2
        exit 1
    fi
    if grep -Eq 'kv-transfer-config:|mooncake_kv_store:' "${config}"; then
        echo "${config}: the pinned nightly cannot combine DCP, DSpark, and a KV connector" >&2
        exit 1
    fi
    if grep -Eq '/vllm-worktree|/vllm/.venv|VIRTUAL_ENV:|PYTHONPATH:' "${config}"; then
        echo "${config}: contains a forbidden local runtime dependency" >&2
        exit 1
    fi
done

echo "bundle validation passed"
