#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRT_SLURM_ROOT="$(cd "${BUNDLE_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${1:-${SRT_SLURM_ROOT}/output_nsys_collection}"
NSYS_HOST_ROOT="${NSYS_HOST_ROOT:-/cm/shared/apps/nvidia/nsight-systems-cli/2025.4.1}"
EXPECTED_SRTCTL_COMMIT="${EXPECTED_SRTCTL_COMMIT:-}"
EXPECTED_IMAGE="vllm/vllm-openai:nightly-44fe2a392b71d52a8d72faf2f8278834379482c9"
EXPECTED_VLLM="0.28.1rc1.dev130+g44fe2a392"
EXPECTED_DYNAMO_HASH="ba83080ecd31c1ce918559e576d3c5bc9e092ff1"
EXPECTED_DYNAMO_VERSION="1.3.0"

expected_runs=(
    kimi-k3-mxfp4-dcp8-dspark4-natural-nightly44fe2a3-128k-nsys
    kimi-k3-mxfp4-dcp8-dspark4-sonnet-nightly44fe2a3-128k-nsys
    kimi-k3-nvfp4-dcp8-dspark4-natural-nightly44fe2a3-128k-nsys
    kimi-k3-nvfp4-dcp8-dspark4-sonnet-nightly44fe2a3-128k-nsys
)

fail() {
    echo "audit failed: $*" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -s "${path}" ]] || fail "missing or empty file: ${path}"
}

[[ -d "${OUTPUT_ROOT}" ]] || fail "output root does not exist: ${OUTPUT_ROOT}"
[[ -x "${NSYS_HOST_ROOT}/bin/nsys" ]] || \
    fail "nsys is unavailable at ${NSYS_HOST_ROOT}/bin/nsys"

temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "${temporary_dir}"' EXIT

for run_name in "${expected_runs[@]}"; do
    run_root="${OUTPUT_ROOT}/${run_name}"
    [[ -d "${run_root}" ]] || fail "missing run directory: ${run_root}"

    mapfile -t job_dirs < <(
        find "${run_root}" -mindepth 1 -maxdepth 1 -type d \
            -printf '%f\t%p\n' \
            | awk -F '\t' '$1 ~ /^[0-9]+$/ {print $2}' \
            | sort
    )
    [[ "${#job_dirs[@]}" -eq 1 ]] || \
        fail "${run_name}: expected one job directory, found ${#job_dirs[@]}"
    job_dir="${job_dirs[0]}"
    job_id="$(basename "${job_dir}")"

    config="${job_dir}/config.yaml"
    lock="${job_dir}/recipe.lock.yaml"
    fingerprint="${job_dir}/logs/fingerprint_agg_w0.json"
    benchmark="${job_dir}/logs/benchmark.out"
    fill_result="${job_dir}/logs/profile-benchmark/cache_fill_isl131072_osl1_c64.json"
    replay_osl=8192
    result="${job_dir}/logs/profile-benchmark/results_isl131072_osl${replay_osl}_c64.json"

    for path in \
        "${config}" \
        "${job_dir}/git_state.txt" \
        "${lock}" \
        "${benchmark}" \
        "${fingerprint}" \
        "${fill_result}" \
        "${result}"; do
        require_file "${path}"
    done

    grep -Fq "${EXPECTED_IMAGE}" "${config}" || \
        fail "${job_id}: config does not use the pinned image"
    grep -Fq "hash: ${EXPECTED_DYNAMO_HASH}" "${lock}" || \
        fail "${job_id}: recipe lock does not use the pinned Dynamo commit"
    grep -Fq "install: true" "${lock}" || \
        fail "${job_id}: recipe lock does not install pinned Dynamo"
    lock_commit="$(awk '$1 == "srtctl_commit:" {print $2}' "${lock}")"
    [[ -n "${lock_commit}" ]] || fail "${job_id}: srtctl commit is missing"
    if [[ -z "${EXPECTED_SRTCTL_COMMIT}" ]]; then
        EXPECTED_SRTCTL_COMMIT="${lock_commit}"
    fi
    [[ "${lock_commit}" == "${EXPECTED_SRTCTL_COMMIT}" ]] || \
        fail "${job_id}: unexpected srtctl commit ${lock_commit}"
    grep -Fq "\"vllm\": \"${EXPECTED_VLLM}\"" "${fingerprint}" || \
        fail "${job_id}: unexpected vLLM version"
    grep -Fq "\"dynamo\": \"${EXPECTED_DYNAMO_VERSION}\"" "${fingerprint}" || \
        fail "${job_id}: unexpected Dynamo version"
    grep -Fq '"flashinfer-python==0.6.17"' "${fingerprint}" || \
        fail "${job_id}: flashinfer-python is not pinned to 0.6.17"
    grep -Fq '"flashinfer-cubin==0.6.17"' "${fingerprint}" || \
        fail "${job_id}: flashinfer-cubin is not pinned to 0.6.17"
    grep -Eq '"flashinfer-jit-cache==0\.6\.17([+"]|$)' "${fingerprint}" || \
        fail "${job_id}: flashinfer-jit-cache is not pinned to 0.6.17"
    grep -Fq "\"VLLM_IMAGE_TAG\": \"${EXPECTED_IMAGE}\"" "${fingerprint}" || \
        fail "${job_id}: runtime image fingerprint does not match"

    mapfile -t worker_logs < <(
        find "${job_dir}/logs" -maxdepth 1 -type f -name '*_agg_w0.out' | sort
    )
    [[ "${#worker_logs[@]}" -eq 2 ]] || \
        fail "${job_id}: expected two aggregate worker logs"
    for worker_log in "${worker_logs[@]}"; do
        grep -Fq "vLLM PR #54277 reverted in the job-local container overlay" \
            "${worker_log}" || fail "${job_id}: PR #54277 was not reverted"
        grep -Fq "All 2 nodes completed the patched runtime setup" \
            "${worker_log}" || fail "${job_id}: runtime setup barrier was not reached"
    done

    grep -Fq "benchmark python: /usr/bin/python3" "${benchmark}" || \
        fail "${job_id}: benchmark did not use container python3"
    grep -Fq "Cache fill complete; launching the identical-prompt replay" \
        "${benchmark}" || fail "${job_id}: cache fill did not complete"
    grep -Fq "Decode-only profile window reached:" "${benchmark}" || \
        fail "${job_id}: decode-only window was not reached"
    grep -Eq 'Successful requests:[[:space:]]+64' "${benchmark}" || \
        fail "${job_id}: benchmark did not complete 64 requests"
    grep -Fq "Profiling results saved to /logs/profiles" "${benchmark}" || \
        fail "${job_id}: profiler did not save its results"

    grep -Fq '"num_prompts": 64' "${result}" || \
        fail "${job_id}: unexpected prompt count in result"
    grep -Fq '"completed": 64' "${fill_result}" || \
        fail "${job_id}: cache fill did not complete 64 requests"
    grep -Fq '"completed": 64' "${result}" || \
        fail "${job_id}: result does not contain 64 completed requests"
    grep -Fq '"total_input_tokens": 8388608' "${result}" || \
        fail "${job_id}: result does not contain 64 x 131072 input tokens"
    expected_output_tokens=$((64 * replay_osl))
    grep -Fq "\"total_output_tokens\": ${expected_output_tokens}" "${result}" || \
        fail "${job_id}: result does not contain 64 x ${replay_osl} output tokens"

    if [[ "${run_name}" == *-sonnet-* ]]; then
        request_cache="sonnet-input-requests.json"
    else
        request_cache="random-input-requests.json"
    fi
    require_file "${job_dir}/logs/profile-benchmark/${request_cache}"

    grep -Fq "External prefix cache hit rate:" "${worker_logs[0]}" || \
        fail "${job_id}: Mooncake external-prefix metrics are absent"

    if find "${job_dir}/logs" -type f \
        \( -name '*.out' -o -name '*.log' \) \
        -exec grep -Eq \
        'Traceback \(most recent call last\)|CUDA out of memory|illegal memory access|WorkerProc hit an exception|Critical process failure|Server did not become healthy' \
        {} +; then
        fail "${job_id}: fatal error found in logs"
    fi

    mapfile -t traces < <(
        find "${job_dir}/logs/profiles/agg" -maxdepth 1 -type f \
            -name '*.nsys-rep' -size +0c | sort
    )
    [[ "${#traces[@]}" -eq 2 ]] || \
        fail "${job_id}: expected two non-empty Nsight reports, found ${#traces[@]}"

    for trace in "${traces[@]}"; do
        sqlite="${temporary_dir}/${job_id}-$(basename "${trace}" .nsys-rep).sqlite"
        "${NSYS_HOST_ROOT}/bin/nsys" stats \
            --quiet \
            --sqlite "${sqlite}" \
            --report cuda_gpu_kern_sum \
            --format csv \
            "${trace}" >/dev/null
    done

    echo "audited ${run_name} (job ${job_id}, two readable traces)"
done

echo "all four nightly DCP8 trace reproductions passed the artifact audit"
echo "srt-slurm commit: ${EXPECTED_SRTCTL_COMMIT}"
