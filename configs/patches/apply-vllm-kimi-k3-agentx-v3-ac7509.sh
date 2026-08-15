#!/usr/bin/env bash
# Port wzhao/kimi-k3-agentx-v3@75c2eef602 runtime changes onto the ac7509e2b1
# nightly in place.

set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly VERSION_FILE="${VLLM_ROOT}/_version.py"
readonly PATCH_FILE="${VLLM_DCP_PATCH_FILE:-/configs/patches/vllm-wzhao-kimi-k3-agentx-v3-on-nightly-ac7509.patch}"
readonly MARKER_FILE="${VLLM_ROOT}/.wzhao_kimi_k3_agentx_v3_75c2eef602_on_gac7509e2b"

if [[ -f "${MARKER_FILE}" ]]; then
  echo "Kimi-K3 DCP runtime patch is already applied."
  exit 0
fi

if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "Missing vLLM patch: ${PATCH_FILE}" >&2
  exit 1
fi

if [[ ! -r "${VERSION_FILE}" ]] || ! grep -q "gac7509e2b" "${VERSION_FILE}"; then
  echo "Refusing to patch: expected vLLM nightly commit gac7509e2b." >&2
  echo "Version file: ${VERSION_FILE}" >&2
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  echo "Kimi-K3 DCP runtime patch content is already present."
else
  echo "Patch neither applies cleanly nor appears already applied." >&2
  exit 1
fi

touch "${MARKER_FILE}"
echo "Applied Kimi-K3 agentx-v3 runtime patch to nightly-ac7509e2b."
