#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRT_SLURM_ROOT="$(cd "${BUNDLE_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${1:-${SRT_SLURM_ROOT}/output_nsys_reproduction/nightly-44fe2a3}"
ANALYSIS_ROOT="${2:-${OUTPUT_ROOT}/analysis}"
NSYS_HOST_ROOT="${NSYS_HOST_ROOT:-/cm/shared/apps/nvidia/nsight-systems-cli/2025.4.1}"
NSYS="${NSYS_HOST_ROOT}/bin/nsys"
PYTHON="${SRT_SLURM_ROOT}/.venv/bin/python"

[[ -x "${NSYS}" ]] || {
    echo "nsys is unavailable at ${NSYS}" >&2
    exit 1
}
[[ -x "${PYTHON}" ]] || {
    echo "repository Python is unavailable at ${PYTHON}" >&2
    exit 1
}

run_specs=(
    "natural|mxfp4|kimi-k3-mxfp4-dcp8-dspark4-natural-nightly44fe2a3-128k-nsys"
    "sonnet|mxfp4|kimi-k3-mxfp4-dcp8-dspark4-sonnet-nightly44fe2a3-128k-nsys"
    "natural|nvfp4|kimi-k3-nvfp4-dcp8-dspark4-natural-nightly44fe2a3-128k-nsys"
    "sonnet|nvfp4|kimi-k3-nvfp4-dcp8-dspark4-sonnet-nightly44fe2a3-128k-nsys"
)

mkdir -p "${ANALYSIS_ROOT}"

for spec in "${run_specs[@]}"; do
    IFS='|' read -r workload precision run_name <<<"${spec}"
    run_root="${OUTPUT_ROOT}/${run_name}"
    mapfile -t job_dirs < <(
        find "${run_root}" -mindepth 1 -maxdepth 1 -type d \
            -name '[0-9]*' | sort
    )
    if [[ "${#job_dirs[@]}" -ne 1 ]]; then
        echo "${run_name}: expected one job directory" >&2
        exit 1
    fi
    job_dir="${job_dirs[0]}"
    mapfile -t reports < <(
        find "${job_dir}/logs/profiles/agg" -maxdepth 1 -type f \
            -name '*.nsys-rep' -size +0c | sort
    )
    if [[ "${#reports[@]}" -ne 2 ]]; then
        echo "${run_name}: expected two reports" >&2
        exit 1
    fi
    worker_log="$(
        find "${job_dir}/logs" -maxdepth 1 -type f \
            -name '*_agg_w0.out' | sort | head -1
    )"
    [[ -n "${worker_log}" ]] || {
        echo "${run_name}: worker log is missing" >&2
        exit 1
    }

    analysis_dir="${ANALYSIS_ROOT}/${precision}-${workload}"
    mkdir -p "${analysis_dir}"
    sqlite_files=()
    for index in "${!reports[@]}"; do
        sqlite="${analysis_dir}/node-${index}.sqlite"
        "${NSYS}" export \
            --quiet true \
            --force-overwrite true \
            --type sqlite \
            --tables CUPTI_ACTIVITY_KIND_KERNEL,StringIds \
            --output "${sqlite}" \
            "${reports[${index}]}"
        sqlite_files+=("${sqlite}")
    done

    "${PYTHON}" "${SCRIPT_DIR}/analyze_dcp_trace.py" \
        --variant "${precision}_dcp8" \
        --aggregation mean \
        --output-json "${analysis_dir}/summary.json" \
        --worker-log "${worker_log}" \
        "${sqlite_files[@]}" \
        >"${analysis_dir}/summary.md"
done

{
    "${PYTHON}" "${SCRIPT_DIR}/render_dcp_comparison.py" \
        --title "Natural routing" \
        --mxfp4 "${ANALYSIS_ROOT}/mxfp4-natural/summary.json" \
        --nvfp4 "${ANALYSIS_ROOT}/nvfp4-natural/summary.json"
    echo
    "${PYTHON}" "${SCRIPT_DIR}/render_dcp_comparison.py" \
        --title "Sonnet text" \
        --mxfp4 "${ANALYSIS_ROOT}/mxfp4-sonnet/summary.json" \
        --nvfp4 "${ANALYSIS_ROOT}/nvfp4-sonnet/summary.json"
} | tee "${ANALYSIS_ROOT}/comparison.md"
