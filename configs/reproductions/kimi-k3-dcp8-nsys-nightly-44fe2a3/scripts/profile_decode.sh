#!/bin/bash

set -euo pipefail

ISL="${PROFILE_ISL:-131072}"
OSL="${PROFILE_OSL:-1024}"
REPLAY_OSL="${PROFILE_REPLAY_OSL:-$((OSL * 4))}"
CACHE_FILL_OSL="${PROFILE_CACHE_FILL_OSL:-1}"
CONCURRENCY="${PROFILE_CONCURRENCY:-64}"
DECODE_ACTIVE_TARGET="${PROFILE_DECODE_ACTIVE_TARGET:-${CONCURRENCY}}"
ENGINE_RUNNING_TARGET="${PROFILE_ENGINE_RUNNING_TARGET:-40}"
ENGINE_STABLE_SAMPLES="${PROFILE_ENGINE_STABLE_SAMPLES:-3}"
DECODE_WINDOW_TIMEOUT="${PROFILE_DECODE_WINDOW_TIMEOUT:-600}"
WAIT_FOR_EPLB_REBALANCE="${PROFILE_WAIT_FOR_EPLB_REBALANCE:-0}"
NUM_PROMPTS="${PROFILE_NUM_PROMPTS:-${CONCURRENCY}}"
SEED="${PROFILE_SEED:-0}"
DATASET_NAME="${PROFILE_DATASET_NAME:-random}"
TEXT_CORPUS_PATH="${PROFILE_TEXT_CORPUS_PATH:-}"
SONNET_PREFIX_LEN="${PROFILE_SONNET_PREFIX_LEN:-128}"
MODEL_NAME="${PROFILE_MODEL_NAME:-moonshotai/Kimi-K3}"
TOKENIZER_PATH="${PROFILE_TOKENIZER_PATH:-/model}"
ENDPOINT="http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}"
BENCHMARK_SCRIPT="/configs/reproductions/kimi-k3-dcp8-nsys-nightly-44fe2a3/scripts/benchmark_serving_wrapper.py"
PYTHON_BIN="${PROFILE_PYTHON_BIN:-$(type -P python3)}"

if ((DECODE_ACTIVE_TARGET <= 0 || DECODE_ACTIVE_TARGET > CONCURRENCY)); then
    echo "PROFILE_DECODE_ACTIVE_TARGET must be in [1, ${CONCURRENCY}]" >&2
    exit 1
fi
if ((ENGINE_RUNNING_TARGET < 0 || ENGINE_RUNNING_TARGET > CONCURRENCY)); then
    echo "PROFILE_ENGINE_RUNNING_TARGET must be in [0, ${CONCURRENCY}]" >&2
    exit 1
fi
if ((ENGINE_STABLE_SAMPLES <= 0)); then
    echo "PROFILE_ENGINE_STABLE_SAMPLES must be positive" >&2
    exit 1
fi
if ((DECODE_WINDOW_TIMEOUT <= 0)); then
    echo "PROFILE_DECODE_WINDOW_TIMEOUT must be positive" >&2
    exit 1
fi
if [[ "${DATASET_NAME}" != "random" && "${DATASET_NAME}" != "sonnet" ]]; then
    echo "PROFILE_DATASET_NAME must be random or sonnet" >&2
    exit 1
fi
if [[ "${DATASET_NAME}" == "sonnet" ]]; then
    if [[ ! -f "${TEXT_CORPUS_PATH}" ]]; then
        echo "Sonnet text corpus is unavailable at ${TEXT_CORPUS_PATH}" >&2
        exit 1
    fi
    if ((SONNET_PREFIX_LEN <= 0 || SONNET_PREFIX_LEN >= ISL)); then
        echo "PROFILE_SONNET_PREFIX_LEN must be in [1, ${ISL})" >&2
        exit 1
    fi
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "python3 is unavailable in the runtime image" >&2
    exit 1
fi
unset VIRTUAL_ENV
if ! "${PYTHON_BIN}" -c 'import aiohttp, numpy, transformers'; then
    echo "Benchmark dependencies are unavailable in the runtime image" >&2
    exit 1
fi
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import sys

packages = ("aiohttp", "ai-dynamo", "numpy", "transformers", "vllm")
print(f"benchmark python: {sys.executable}")
for package in packages:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"
    print(f"benchmark package: {package}=={version}")
PY
if [[ "${PROFILE_VALIDATE_CLIENT_ONLY:-0}" == "1" ]]; then
    "${PYTHON_BIN}" "${BENCHMARK_SCRIPT}" --help >/dev/null
    echo "CLIENT_VALIDATION_OK"
    exit 0
fi

source /srtctl-benchmarks/lib/profiling.sh
profiling_init_from_env

cleanup() {
    stop_all_profiling
    if [[ -n "${replay_pid:-}" ]] && kill -0 "${replay_pid}" 2>/dev/null; then
        kill "${replay_pid}" 2>/dev/null || true
        wait "${replay_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

run_phase() {
    local output_len="$1"
    local result_file="${2:-}"
    local request_cache_mode="${3:-}"
    local -a save_args=()
    local -a dataset_args=()
    local request_cache_path="/logs/profile-benchmark/${DATASET_NAME}-input-requests.json"
    if [[ -n "${result_file}" ]]; then
        save_args=(
            --save-result
            --result-dir /logs/profile-benchmark
            --result-filename "${result_file}"
        )
    fi

    if [[ "${DATASET_NAME}" == "sonnet" ]]; then
        dataset_args=(
            --dataset-name sonnet
            --dataset-path "${TEXT_CORPUS_PATH}"
            --sonnet-input-len "${ISL}"
            --sonnet-output-len "${output_len}"
            --sonnet-prefix-len "${SONNET_PREFIX_LEN}"
            --sonnet-exact-input-len
        )
        if [[ "${request_cache_mode}" == "save" ]]; then
            dataset_args+=(--save-input-requests "${request_cache_path}")
        elif [[ "${request_cache_mode}" == "load" ]]; then
            dataset_args+=(--load-input-requests "${request_cache_path}")
        fi
    else
        dataset_args=(
            --dataset-name random
            --random-input-len "${ISL}"
            --random-output-len "${output_len}"
            --random-range-ratio 1.0
            --random-num-workers 8
        )
        if [[ "${request_cache_mode}" == "save" ]]; then
            dataset_args+=(--save-input-requests "${request_cache_path}")
        elif [[ "${request_cache_mode}" == "load" ]]; then
            dataset_args+=(--load-input-requests "${request_cache_path}")
        fi
    fi

    "${PYTHON_BIN}" -u "${BENCHMARK_SCRIPT}" \
        --model "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --base-url "${ENDPOINT}" \
        --backend dynamo \
        --endpoint /v1/completions \
        "${dataset_args[@]}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency "${CONCURRENCY}" \
        --request-rate inf \
        --seed "${SEED}" \
        --ignore-eos \
        --disable-tqdm \
        --trust-remote-code \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        "${save_args[@]}"
}

require_completed_requests() {
    local result_path="$1"
    "${PYTHON_BIN}" - "${result_path}" "${NUM_PROMPTS}" <<'PY'
import json
import sys

result_path, expected_text = sys.argv[1:]
with open(result_path) as result_file:
    result = json.load(result_file)
completed = result.get("completed")
expected = int(expected_text)
if completed != expected:
    raise SystemExit(
        f"Expected {expected} completed requests in {result_path}, got {completed}"
    )
PY
}

wait_for_decode_window() {
    local marker="$1"
    local replay_pid="$2"
    local deadline=$((SECONDS + DECODE_WINDOW_TIMEOUT))

    while kill -0 "${replay_pid}" 2>/dev/null; do
        if [[ -e "${marker}" ]]; then
            echo "Decode-only profile window reached: ${DECODE_ACTIVE_TARGET} active streams"
            return 0
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for ${DECODE_ACTIVE_TARGET} active decode streams" >&2
            return 1
        fi
        sleep 0.25
    done

    echo "Replay exited before reaching ${DECODE_ACTIVE_TARGET} active decode streams" >&2
    return 1
}

wait_for_stable_engine_batch() {
    local replay_pid="$1"
    local deadline=$((SECONDS + 600))
    local stable_samples=0
    local last_sample_count=0
    local sample_count=0
    local engine_totals=""
    local running=0
    local waiting=0

    if ((ENGINE_RUNNING_TARGET == 0)); then
        return 0
    fi

    while kill -0 "${replay_pid}" 2>/dev/null; do
        sample_count="$(
            grep -h -c "Engine .*Running:" /logs/*_agg_w0.out 2>/dev/null \
                | awk '{total += $1} END {print total + 0}'
        )"
        if ((sample_count > last_sample_count)); then
            last_sample_count="${sample_count}"
            engine_totals="$(
                sed -nE \
                    's/.*Engine ([0-9]+):.*Running: ([0-9]+) reqs, Waiting: ([0-9]+) reqs.*/\1 \2 \3/p' \
                    /logs/*_agg_w0.out 2>/dev/null \
                    | awk '{running[$1]=$2; waiting[$1]=$3} END {for (e in running) {r += running[e]; w += waiting[e]} print r + 0, w + 0}'
            )"
            read -r running waiting <<<"${engine_totals}"
            if ((running >= ENGINE_RUNNING_TARGET)); then
                ((stable_samples += 1))
                echo "Stable engine decode sample ${stable_samples}/${ENGINE_STABLE_SAMPLES}: ${running} running, ${waiting} waiting (minimum ${ENGINE_RUNNING_TARGET})"
                if ((stable_samples >= ENGINE_STABLE_SAMPLES)); then
                    printf '{"minimum_running":%d,"stable_samples":%d,"actual_running":%d,"actual_waiting":%d}\n' \
                        "${ENGINE_RUNNING_TARGET}" \
                        "${ENGINE_STABLE_SAMPLES}" \
                        "${running}" \
                        "${waiting}" \
                        > /logs/profile-benchmark/engine-batch-at-profile-start.json
                    return 0
                fi
            else
                stable_samples=0
            fi
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for ${ENGINE_STABLE_SAMPLES} stable engine samples with at least ${ENGINE_RUNNING_TARGET} running requests" >&2
            echo "Latest engine totals: ${running} running, ${waiting} waiting" >&2
            return 1
        fi
        sleep 0.25
    done

    echo "Replay exited before the engine sustained at least ${ENGINE_RUNNING_TARGET} running requests" >&2
    return 1
}

eplb_rearrangement_count() {
    { grep -h -c "Rearranged experts .* in .* s\." \
        /logs/*_agg_w0.out 2>/dev/null || true; } \
        | awk '{total += $1} END {print total + 0}'
}

wait_for_eplb_rebalance() {
    local replay_pid="$1"
    local baseline_count="$2"
    local deadline=$((SECONDS + 600))
    local count=0

    while kill -0 "${replay_pid}" 2>/dev/null; do
        count="$(eplb_rearrangement_count)"
        if ((count > baseline_count)); then
            echo "EPLB rearrangement completed during decode replay: ${baseline_count} -> ${count}"
            touch "/logs/profile-benchmark/eplb-rebalanced-${baseline_count}-to-${count}"
            return 0
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for an EPLB rearrangement after decode replay began" >&2
            return 1
        fi
        sleep 0.25
    done

    echo "Replay exited before EPLB completed a new rearrangement" >&2
    return 1
}

mkdir -p /logs/profile-benchmark

echo "Cache fill: dataset=${DATASET_NAME}, ${NUM_PROMPTS} prompts, ISL=${ISL}, OSL=${CACHE_FILL_OSL}, concurrency=${CONCURRENCY}"
fill_result_name="cache_fill_isl${ISL}_osl${CACHE_FILL_OSL}_c${CONCURRENCY}.json"
run_phase "${CACHE_FILL_OSL}" "${fill_result_name}" save
require_completed_requests "/logs/profile-benchmark/${fill_result_name}"

echo "Cache fill complete; launching the identical-prompt replay"
if [[ "${PROFILE_FILL_ONLY:-0}" == "1" ]]; then
    echo "Fill-only validation complete"
    trap - EXIT
    exit 0
fi
echo "Profile replay: dataset=${DATASET_NAME}, ${NUM_PROMPTS} prompts, ISL=${ISL}, OSL=${REPLAY_OSL}, concurrency=${CONCURRENCY}"
decode_marker="/logs/profile-benchmark/decode-active-c${DECODE_ACTIVE_TARGET}"
rm -f "${decode_marker}"
export SRT_BENCH_DECODE_ACTIVE_MARKER="${decode_marker}"
export SRT_BENCH_DECODE_ACTIVE_TARGET="${DECODE_ACTIVE_TARGET}"
eplb_count_before_replay=0
if [[ "${WAIT_FOR_EPLB_REBALANCE}" == "1" ]]; then
    eplb_count_before_replay="$(eplb_rearrangement_count)"
fi
run_phase "${REPLAY_OSL}" \
    "results_isl${ISL}_osl${REPLAY_OSL}_c${CONCURRENCY}.json" load &
replay_pid=$!
unset SRT_BENCH_DECODE_ACTIVE_MARKER SRT_BENCH_DECODE_ACTIVE_TARGET
wait_for_decode_window "${decode_marker}" "${replay_pid}"
if [[ "${WAIT_FOR_EPLB_REBALANCE}" == "1" ]]; then
    wait_for_eplb_rebalance "${replay_pid}" "${eplb_count_before_replay}"
fi
wait_for_stable_engine_batch "${replay_pid}"
start_all_profiling
wait "${replay_pid}"

stop_all_profiling
trap - EXIT
