#!/bin/bash
set -euo pipefail
TASK_ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/weizha
TRACE_ROOT="${TASK_ROOT}/vllm/srt-slurm"
PYTHON_BIN="${TASK_ROOT}/vllm/.venv/bin/python"
export PROFILE_TOKENIZER_PATH="${TASK_ROOT}/models/kimi-k3-model"
export PROFILE_TEXT_CORPUS_PATH="${TRACE_ROOT}/configs/shakespeare.txt"
export PROFILE_ISL=131072 PROFILE_EXTENSION_TOKENS=49152
export KIMI_WIDEEP_SA_BENCHMARK="${TRACE_ROOT}/src/srtctl/benchmarks/scripts/sa-bench/benchmark_serving.py"
"${PYTHON_BIN}" -c '
import torch
import mooncake.store
import flashinfer
for device in range(torch.cuda.device_count()):
    x = torch.ones(8, device=f"cuda:{device}")
    torch.cuda.synchronize(device)
print("GPU and package imports OK", torch.__version__, flashinfer.__version__)
'
"${PYTHON_BIN}" "${TRACE_ROOT}/configs/kimi-wideep-benchmark-serving.py" --help
"${PYTHON_BIN}" "${TRACE_ROOT}/configs/kimi-sonnet-prefill-profile.py" prepare
echo TRACE_CLIENT_PREFLIGHT_OK
