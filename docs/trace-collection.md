# Prefill-only and decode-only traces

The reusable clients live in `src/srtctl/benchmarks/scripts/profiling/` and
are mounted at `/srtctl-benchmarks/profiling` in benchmark containers.
Use a custom benchmark command to invoke either client:

```bash
bash /srtctl-benchmarks/profiling/prefill-only.sh
bash /srtctl-benchmarks/profiling/decode-only.sh
```

These clients currently use the Dynamo completions frontend and aggregated
worker profiling endpoints. They do not change model or server settings.
Enable Nsight profiling in the recipe and supply `PROFILE_MODEL_NAME` and,
if needed, `PROFILE_PYTHON_BIN` (defaults to `python3`). The interpreter must
have the benchmark dependencies and the model tokenizer installed.
The frontend address and profiling endpoints are supplied by srtctl.

For repeated runs with FlashInfer autotuning enabled, set the worker environment
variable `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` to a persistent writable directory.
Job-specific temporary cache directories do not reuse previous tuning results.
First-time tuning can still compile CuTeDSL kernel candidates on the CPU;
an idle GPU during this stage does not by itself establish a startup hang.

## Prefill

Set `PROFILE_TOKENIZER_PATH` (default `/model`), `PROFILE_TEXT_CORPUS_PATH`,
`PROFILE_ISL` (seed length, default 131072), and
`PROFILE_EXTENSION_TOKENS` (default 49152).
The client prepares a token-counted text prefix, seeds it outside the capture,
then profiles its extension with one output token. It records response usage
and worker metrics before/after capture under `/logs/profile-benchmark`.
It rejects a cold extension when response cache details are available;
otherwise cache reuse must be verified from metrics or engine logs.

Use manual profile stopping (`stop_step: null`) for this workload: empty
connector calls can consume a worker-step limit before prefill starts.
The script stops profiling when the extension finishes, including on failure.
The capture can include idle calls and the final sampling work; verify the
actual nonempty prefill ranges in Nsight before interpreting the result.

## Decode

Set `PROFILE_CONCURRENCY`, `PROFILE_ISL`, and `PROFILE_REPLAY_OSL`.
`PROFILE_DATASET_NAME` selects `random` (default) or `sonnet`;
Sonnet also requires `PROFILE_TEXT_CORPUS_PATH`.
The client seeds the cache with one output token, then replays the identical
prompts with a long decode. Sonnet prompts are saved and reused exactly.
Capture starts only after the requested number of streams has received its
first token and the engine reports that many running requests with no waiting
requests for three fresh log samples.

`PROFILE_ENGINE_STABLE_SAMPLES` controls the sample count;
`PROFILE_DECODE_WINDOW_TIMEOUT` controls the gate timeout (default 1800 seconds).
`PROFILE_ENGINE_RUNNING_TARGET=0` explicitly disables the engine-log gate;
this weakens validation and should not be used for latency comparisons.
Configure the server profile step limit for the desired number of decode steps.
Choose enough output tokens to keep the full batch active through the capture.

Client success or a nonempty report file is not proof of a valid capture.
Check captured NVTX batch shapes, GPU kernel activity on every rank, and absence
of prefill kernels in the selected decode window. Keep cache-fill and replay
results alongside the trace so failed requests or preemptions are visible.
