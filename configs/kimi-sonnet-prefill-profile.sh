#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PROFILE_REPO_VENV:-/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/weizha/vllm/.venv}/bin/python"
CLIENT=/configs/kimi-sonnet-prefill-profile.py
source /srtctl-benchmarks/lib/profiling.sh
profiling_init_from_env
[[ "${PROFILE_TYPE}" == nsys && -n "${PROFILE_AGG_ENDPOINTS}" ]]
trap stop_all_profiling EXIT

"${PYTHON_BIN}" -u "${CLIENT}" prepare
"${PYTHON_BIN}" -u "${CLIENT}" seed
snapshot_metrics() {
    local phase="$1"
    local endpoint
    for endpoint in ${PROFILE_AGG_ENDPOINTS}; do
        endpoint="$(profiling__normalize_endpoint "${endpoint}" "${WORKER_PORT}")"
        curl --max-time 10 -fsS "http://${endpoint}/metrics" \
            >"/logs/profile-benchmark/prefill-${phase}-metrics-${endpoint//:/_}.txt" \
            || echo "Worker metrics unavailable at ${endpoint}; verify cache reuse from engine logs."
    done
}
snapshot_metrics before
start_all_profiling
"${PYTHON_BIN}" -u "${CLIENT}" extend
stop_all_profiling
snapshot_metrics after
trap - EXIT
