# Kimi-K3 DCP8 Nsight Systems reproduction

This bundle reproduces the DCP8 decode traces represented by the four runs below:

| Checkpoint | Input routing | Historical run |
| --- | --- | --- |
| MXFP4 | natural, random-token input | `output_nsys_collection/kimi-k3-mxfp4-dcp8-trtllm-moe-dspark4-natural-routing-autotune-conc64-128k-nsys/597280` |
| MXFP4 | natural, Sonnet text | `output_nsys_collection/kimi-k3-mxfp4-dcp8-trtllm-moe-dspark4-sonnet-realtext-autotune-conc64-128k-nsys/597291` |
| NVFP4 | natural, random-token input | `output_nsys_collection/kimi-k3-nvfp4-dcp8-trtllm-moe-dspark4-natural-routing-autotune-conc64-128k-nsys/597283` |
| NVFP4 | natural, Sonnet text | `output_nsys_collection/kimi-k3-nvfp4-dcp8-trtllm-moe-dspark4-sonnet-realtext-autotune-conc64-128k-nsys/597833` |

The runtime is intentionally independent of a local vLLM or Dynamo checkout. All four recipes use:

- `vllm/vllm-openai:nightly-44fe2a392b71d52a8d72faf2f8278834379482c9`
- FlashInfer Python, cubin, and CUDA 13 JIT-cache packages at version `0.6.17`
- Dynamo commit `ba83080ecd31c1ce918559e576d3c5bc9e092ff1`, installed
  into the job container by `srt-slurm`
- Nsight Systems CLI 2025.4.1, mounted read-only from the cluster installation
- the container's `python3` and Python packages for the benchmark client
- the bundled Sonnet corpus with SHA-256 `e0e13a826912a4a81bb3a582aa73c4af0675bdeee6ddf6d505efb63d562d496f`

`dynamo.install: true` performs the pinned install in the container's writable
overlay. It does not read or modify the submitter's local vLLM or Dynamo
virtual environments. The pinned commit reports package version `1.3.0` at
runtime and contains the Kimi-K3 frontend tokenizer support required here.

The nightly image contains vLLM PR #54277. With DCP8 and DSpark, that code
reproducibly triggered a CUDA illegal-memory-access failure after 53 of 54
131,072-token cache-fill requests. Job `631140` applied the complete reverse
diff in its container overlay and completed all 54 requests without an IMA.
The historical successful traces also used FlashInfer 0.6.17, so each recipe
pins the Python, cubin, and CUDA 13 JIT-cache packages to 0.6.17 and reverses
PR #54277 before vLLM starts. A cross-node barrier prevents any rank from
entering the distributed rendezvous before every node has finished this
runtime setup. `srt-slurm` labels setup invocations by process context, so
Mooncake infrastructure and Dynamo frontend processes skip this worker-only
overlay and cannot deadlock on the worker barrier. These operations do not modify the image, host Python
installation, or a local virtual environment. The artifact audit verifies the
effective package versions and patch/barrier markers from both worker logs.

The benchmark wrapper supplies fail-closed stubs for the unused Hugging Face Dataset and pandas CSV loaders. The random-token and Sonnet modes do not call either loader, so this avoids installing optional client packages into the runtime image while still failing clearly if an unsupported dataset mode is selected.

The checked-out `srt-slurm` benchmark client provides exact Sonnet token-length
truncation and save/load of finalized Sonnet or random prompt lists. Saving
after tokenization and loading the same list for replay guarantees that cache
fill and profiled decode receive byte-for-byte identical prompts.
`verify_bundle.sh` rejects an `srt-slurm` checkout that lacks these options.

Before starting the servers on a new cluster, the same pinned image can test
the tokenizer-level Sonnet path with:

```bash
python3 /configs/reproductions/kimi-k3-dcp8-nsys-nightly-44fe2a3/scripts/smoke_sonnet_requests.py
```

Run that command inside the image with this checkout's `configs/` mounted at
`/configs`, `src/srtctl/benchmarks/scripts` at `/srtctl-benchmarks`, and either
Kimi checkpoint at `/model`. It generates exact 512-token prompts and verifies
that the finalized prompt list survives a save/load round trip unchanged.

There is no vLLM source mount, `PYTHONPATH`, `VIRTUAL_ENV`, or local runtime `.venv` in these recipes. The only local inputs are the three model snapshots, the checked-out `srt-slurm` source, and the cluster's Nsight Systems installation. Runtime package downloads come from the public FlashInfer wheel indexes and use a job-local cache under `/tmp`. The pinned nightly image does not contain `nsys`, so the recipes mount the complete CLI installation instead of borrowing a Python environment.

## Hardware and software prerequisites

- Slurm with Pyxis/Enroot and two GB300 nodes per trace, four GPUs per node.
- The `coreai_comparch_inferencex` account and `batch` partition, or equivalent values edited in all four YAML files.
- `uv` on the login node for the submission-side `srtctl` environment.
- Nsight Systems CLI 2025.4.1. It is expected at `/cm/shared/apps/nvidia/nsight-systems-cli/2025.4.1` by default; set `NSYS_HOST_ROOT` if the cluster installs it elsewhere.
- A checkout of the revision documented with the final reproduction results.

The model snapshots are pinned by Hugging Face identity:

| Environment variable | Repository | Revision |
| --- | --- | --- |
| `KIMI_K3_MXFP4_MODEL` | `moonshotai/Kimi-K3` | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` |
| `KIMI_K3_NVFP4_MODEL` | `nvidia/Kimi-K3-NVFP4` | `5b6e714eeda742347195f6045e642f3498e674b1` |
| `KIMI_K3_DSPARK_MODEL` | `Inferact/Kimi-K3-DSpark` | `cf6b8244620e7ea4b0651d214f28e89eac75bed6` |

The recipes mount the target checkpoint as `/model` and the draft checkpoint as `/dspark-model`. Model storage paths are not otherwise embedded in the bundle.

## Prepare `srtctl`

From the repository root:

```bash
git checkout wzhao/kimi-k3-dcp8-nsys-nightly44fe2a3
make setup ARCH=aarch64
uv sync --python 3.12
```

`make setup` downloads the pinned NATS, etcd, and compute-architecture `uv` binaries used by `srt-slurm`. The `uv` environment is used only to submit and orchestrate Slurm jobs. None of these components is mounted as the vLLM, Dynamo, or benchmark Python runtime.
The submission helper invokes `.venv/bin/srtctl` directly after `uv sync`; this avoids accidentally selecting an unrelated active virtual environment.

## Validate and submit

```bash
export KIMI_K3_MXFP4_MODEL=/path/to/moonshotai--Kimi-K3
export KIMI_K3_NVFP4_MODEL=/path/to/nvidia--Kimi-K3-NVFP4
export KIMI_K3_DSPARK_MODEL=/path/to/Inferact--Kimi-K3-DSpark
export NSYS_HOST_ROOT=/path/to/nsight-systems-cli/2025.4.1  # optional on this cluster

configs/reproductions/kimi-k3-dcp8-nsys-nightly-44fe2a3/scripts/verify_bundle.sh
configs/reproductions/kimi-k3-dcp8-nsys-nightly-44fe2a3/scripts/submit_all.sh
```

The submission script first validates all four configs with `srtctl dry-run`, then submits each config separately with a meaningful output directory under:

```text
output_nsys_collection/<config-name>/<job-id>/
```

Pass an alternate output root as the first argument to `submit_all.sh` if desired.

`submit_all.sh` sets `SRTCTL_NSYS_BIN=/opt/nsight-systems/bin/nsys`, and every recipe mounts `${NSYS_HOST_ROOT}` at `/opt/nsight-systems`. This uses the `srt-slurm` profiler's explicit Nsight binary override, so it does not depend on the submitting shell's module state or `PATH`.

## Workload and capture semantics

Each job starts one aggregated TP8+DCP8 worker spanning two four-GPU nodes. It fills the prefix cache with 64 exact 131,072-token prompts and one output token, serializes those finalized prompts, then reloads the same requests with 4,096 output tokens. Nsight Systems starts only after the requested number of streams reaches decode.

- `*-natural.yaml` uses random-token prompts without forced expert balancing. The exact generated prompts are serialized before cache fill and reloaded for replay, so reproduction does not depend on multiprocessing order. This exercises the model's unmodified router.
- `*-sonnet.yaml` uses exact-length prompts generated from the bundled Shakespeare corpus and reuses the serialized requests for the replay.
- MXFP4 natural, MXFP4 Sonnet, and NVFP4 natural wait for 64 active decode streams and three stable engine samples.
- NVFP4 Sonnet starts at 53 active streams, matching historical run `597833`, which could not sustain all 64 streams simultaneously during the capture window.

All recipes use DSpark K=4 with the historical synthetic acceptance length of 3.36, FP8 KV cache, TokenSpeed MLA, FlashInfer autotuning, a 512-token maximum CUDA graph capture size, and the historical embedded Mooncake prefix store.

The historical decode windows materially used Mooncake: their cumulative external-prefix hit rates reached 26.9% to 36.5%. Without the connector, the pinned image admitted only 45 to 51 of the 64 replay requests because the GPU-local hybrid-cache working set was exhausted. PR #54277 in the pinned image also forces DCP KV interleave to the 1,536-token block size when a connector is present, making this DSpark configuration invalid and causing an illegal memory access during long cache fill. The job-local reverse patch restores the earlier DCP cache layout; the recipes then enable the historical Mooncake connector and explicitly retain token-level interleave (`cp-kv-cache-interleave-size: 1`).

## Required artifacts and checks

A successful output directory contains:

```text
config.yaml
git_state.txt
recipe.lock.yaml
logs/benchmark.out
logs/fingerprint_agg_w0.json
logs/profile-benchmark/results_isl131072_osl4096_c64.json
logs/profiles/agg/*_profile_gpu0-1-2-3.nsys-rep
```

For Sonnet runs, it also contains `logs/profile-benchmark/sonnet-input-requests.json`. There should be one `.nsys-rep` per worker node.

Before accepting a trace, verify:

1. `fingerprint_agg_w0.json` reports the vLLM version from the pinned image and
   Dynamo package version `1.3.0` and all three FlashInfer packages at `0.6.17`;
   `config.yaml` and `recipe.lock.yaml` record Dynamo commit
   `ba83080ecd31c1ce918559e576d3c5bc9e092ff1`.
2. `benchmark.out` reports the container `python3`, not a Lustre virtual environment.
3. `benchmark.out` reaches the decode-only window and exits successfully.
4. Both `.nsys-rep` files are non-empty and `nsys stats` can open them.
5. Worker logs contain no CUDA OOM or runtime failure.

After all four jobs complete, run the bundled fail-closed audit:

```bash
configs/reproductions/kimi-k3-dcp8-nsys-nightly-44fe2a3/scripts/audit_outputs.sh
```

The auditor requires exactly one job directory for each named reproduction,
checks the pinned image and runtime fingerprints, confirms all 64 requests and
their exact token counts completed, verifies every request cache, rejects
fatal worker errors, and opens all eight reports with `nsys stats`.

## Kernel-step analysis

The complete four-run analysis is automated:

```bash
configs/reproductions/kimi-k3-dcp8-nsys-nightly-44fe2a3/scripts/analyze_all.sh
```

It exports only the CUDA kernel and string tables, aggregates all eight GPUs
for each run, writes a JSON and Markdown summary per run under
`output_nsys_collection/analysis-nightly-44fe2a3/`, and renders separate
MXFP4-versus-NVFP4 tables for natural and Sonnet routing. The commands below
show the equivalent per-run workflow.

Export both node reports for one run to SQLite with the same Nsight Systems
installation that recorded them:

```bash
mkdir -p analysis/mxfp4-natural
index=0
for report in output_nsys_collection/\
kimi-k3-mxfp4-dcp8-dspark4-natural-nightly44fe2a3-128k-nsys/*/\
logs/profiles/agg/*.nsys-rep; do
    /cm/shared/apps/nvidia/nsight-systems-cli/2025.4.1/bin/nsys export \
        --type sqlite \
        --output "analysis/mxfp4-natural/node-${index}.sqlite" \
        "${report}"
    index=$((index + 1))
done
```

Then aggregate every captured GPU on both nodes. The analyzer identifies a
decode step from the KDA layer sequence, classifies all 93 target-model layers,
and reports both summed GPU activity and start-to-start decode cadence:

```bash
.venv/bin/python \
    configs/reproductions/kimi-k3-dcp8-nsys-nightly-44fe2a3/scripts/\
analyze_dcp_trace.py \
    --variant mxfp4_dcp8 \
    --aggregation mean \
    --output-json analysis/mxfp4-natural/summary.json \
    --worker-log "$(find output_nsys_collection/\
kimi-k3-mxfp4-dcp8-dspark4-natural-nightly44fe2a3-128k-nsys \
-name '*_agg_w0.out' | sort | head -1)" \
    analysis/mxfp4-natural/node-0.sqlite \
    analysis/mxfp4-natural/node-1.sqlite
```

Use `--variant nvfp4_dcp8` for the NVFP4 reports. Repeat this for natural and
Sonnet inputs rather than combining workloads; routing changes the MoE kernel
distribution and therefore must remain visible in the comparison.

After generating the MXFP4 and NVFP4 JSON summaries for one workload, render
the comparison table with:

```bash
.venv/bin/python \
    configs/reproductions/kimi-k3-dcp8-nsys-nightly-44fe2a3/scripts/\
render_dcp_comparison.py \
    --title "Natural routing" \
    --mxfp4 analysis/mxfp4-natural/summary.json \
    --nvfp4 analysis/nvfp4-natural/summary.json
```
