#!/bin/bash

set -euo pipefail

ISL="${PROFILE_ISL:-131072}"
OSL="${PROFILE_OSL:-1024}"
REPLAY_OSL="${PROFILE_REPLAY_OSL:-$((OSL * 4))}"
CACHE_FILL_OSL="${PROFILE_CACHE_FILL_OSL:-1}"
CONCURRENCY="${PROFILE_CONCURRENCY:-64}"
DECODE_ACTIVE_TARGET="${PROFILE_DECODE_ACTIVE_TARGET:-${CONCURRENCY}}"
ENGINE_RUNNING_TARGET="${PROFILE_ENGINE_RUNNING_TARGET:-${CONCURRENCY}}"
ENGINE_STABLE_SAMPLES="${PROFILE_ENGINE_STABLE_SAMPLES:-3}"
NUM_PROMPTS="${PROFILE_NUM_PROMPTS:-${CONCURRENCY}}"
SEED="${PROFILE_SEED:-0}"
MODEL_NAME="${PROFILE_MODEL_NAME:-moonshotai/Kimi-K3}"
TOKENIZER_PATH="${PROFILE_TOKENIZER_PATH:-/model}"
ENDPOINT="http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}"
BENCHMARK_SCRIPT="/configs/kimi-wideep-benchmark-serving.py"
REPO_VENV="${PROFILE_REPO_VENV:-/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/weizha/vllm/.venv}"
PYTHON_BIN="${REPO_VENV}/bin/python"

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

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Repository virtualenv is unavailable at ${REPO_VENV}" >&2
    exit 1
fi
unset VIRTUAL_ENV
if ! "${PYTHON_BIN}" -c 'import transformers'; then
    echo "Benchmark dependencies are unavailable to ${PYTHON_BIN}" >&2
    exit 1
fi
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
    local -a save_args=()
    if [[ -n "${result_file}" ]]; then
        save_args=(
            --save-result
            --result-dir /logs/profile-benchmark
            --result-filename "${result_file}"
        )
    fi

    "${PYTHON_BIN}" -u "${BENCHMARK_SCRIPT}" \
        --model "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --base-url "${ENDPOINT}" \
        --backend dynamo \
        --endpoint /v1/completions \
        --dataset-name random \
        --random-input-len "${ISL}" \
        --random-output-len "${output_len}" \
        --random-range-ratio 1.0 \
        --random-num-workers 8 \
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

wait_for_decode_window() {
    local marker="$1"
    local replay_pid="$2"
    local deadline=$((SECONDS + 600))

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
            if ((running == ENGINE_RUNNING_TARGET && waiting == 0)); then
                ((stable_samples += 1))
                echo "Stable engine decode sample ${stable_samples}/${ENGINE_STABLE_SAMPLES}: ${ENGINE_RUNNING_TARGET} running"
                if ((stable_samples >= ENGINE_STABLE_SAMPLES)); then
                    touch "/logs/profile-benchmark/engine-running-c${ENGINE_RUNNING_TARGET}-stable${ENGINE_STABLE_SAMPLES}"
                    return 0
                fi
            else
                stable_samples=0
            fi
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for ${ENGINE_STABLE_SAMPLES} stable engine samples with ${ENGINE_RUNNING_TARGET} running requests" >&2
            echo "Latest engine totals: ${running} running, ${waiting} waiting" >&2
            return 1
        fi
        sleep 0.25
    done

    echo "Replay exited before the engine sustained ${ENGINE_RUNNING_TARGET} running requests" >&2
    return 1
}

mkdir -p /logs/profile-benchmark

echo "Cache fill: ${NUM_PROMPTS} prompts, ISL=${ISL}, OSL=${CACHE_FILL_OSL}, concurrency=${CONCURRENCY}"
run_phase "${CACHE_FILL_OSL}"

echo "Cache fill complete; launching the identical-prompt replay"
echo "Profile replay: ${NUM_PROMPTS} prompts, ISL=${ISL}, OSL=${REPLAY_OSL}, concurrency=${CONCURRENCY}"
decode_marker="/logs/profile-benchmark/decode-active-c${DECODE_ACTIVE_TARGET}"
rm -f "${decode_marker}"
export SRT_BENCH_DECODE_ACTIVE_MARKER="${decode_marker}"
export SRT_BENCH_DECODE_ACTIVE_TARGET="${DECODE_ACTIVE_TARGET}"
run_phase "${REPLAY_OSL}" \
    "results_isl${ISL}_osl${REPLAY_OSL}_c${CONCURRENCY}.json" &
replay_pid=$!
unset SRT_BENCH_DECODE_ACTIVE_MARKER SRT_BENCH_DECODE_ACTIVE_TARGET
wait_for_decode_window "${decode_marker}" "${replay_pid}"
wait_for_stable_engine_batch "${replay_pid}"
start_all_profiling
wait "${replay_pid}"

stop_all_profiling
trap - EXIT
