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
- `ai-dynamo==1.2.1`, installed into that container by `srt-slurm`
- Nsight Systems CLI 2025.4.1, mounted read-only from the cluster installation
- the container's `python3` and Python packages for the benchmark client
- the bundled Sonnet corpus with SHA-256 `e0e13a826912a4a81bb3a582aa73c4af0675bdeee6ddf6d505efb63d562d496f`

There is no vLLM source mount, `PYTHONPATH`, `VIRTUAL_ENV`, or local runtime `.venv` in these recipes. The only local inputs are the three model snapshots, the checked-out `srt-slurm` source, and the cluster's Nsight Systems installation. The pinned nightly image does not contain `nsys`, so the recipes mount the complete CLI installation instead of borrowing a Python environment.

## Hardware and software prerequisites

- Slurm with Pyxis/Enroot and two GB300 nodes per trace, four GPUs per node.
- The `coreai_comparch_inferencex` account and `batch` partition, or equivalent values edited in all four YAML files.
- RDMA device `mlx5_8`, or the corresponding device edited in all four YAML files.
- `uv` on the login node for the submission-side `srtctl` environment.
- Nsight Systems CLI 2025.4.1. It is expected at `/cm/shared/apps/nvidia/nsight-systems-cli/2025.4.1` by default; set `NSYS_HOST_ROOT` if the cluster installs it elsewhere.
- A checkout of this repository at tag `kimi-k3-dcp8-nsys-nightly-44fe2a3`.

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
git checkout kimi-k3-dcp8-nsys-nightly-44fe2a3
make setup ARCH=aarch64
uv sync --python 3.12
```

`make setup` downloads the pinned NATS, etcd, and compute-architecture `uv` binaries used by `srt-slurm`. The `uv` environment is used only to submit and orchestrate Slurm jobs. None of these components is mounted as the vLLM, Dynamo, or benchmark Python runtime.

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
output_nsys_reproduction/nightly-44fe2a3/<config-name>/<job-id>/
```

Pass an alternate output root as the first argument to `submit_all.sh` if desired.

`submit_all.sh` sets `SRTCTL_NSYS_BIN=/opt/nsight-systems/bin/nsys`, and every recipe mounts `${NSYS_HOST_ROOT}` at `/opt/nsight-systems`. This uses the `srt-slurm` profiler's explicit Nsight binary override, so it does not depend on the submitting shell's module state or `PATH`. The recipes omit `router-session-affinity-ttl-secs` because the pinned `ai-dynamo==1.2.1` CLI does not support that newer option.

## Workload and capture semantics

Each job starts one aggregated TP8+DCP8 worker spanning two four-GPU nodes. It fills the prefix cache with 64 exact 131,072-token prompts and one output token, then replays the same prompts with 4,096 output tokens. Nsight Systems starts only after the requested number of streams reaches decode.

- `*-natural.yaml` uses deterministic random-token prompts without forced expert balancing. This exercises the model's unmodified router.
- `*-sonnet.yaml` uses exact-length prompts generated from the bundled Shakespeare corpus and reuses the serialized requests for the replay.
- MXFP4 natural, MXFP4 Sonnet, and NVFP4 natural wait for 64 active decode streams and three stable engine samples.
- NVFP4 Sonnet starts at 53 active streams, matching historical run `597833`, which could not sustain all 64 streams simultaneously during the capture window.

All recipes use DSpark K=4 with the historical synthetic acceptance length of 3.36, FP8 KV cache, TokenSpeed MLA, FlashInfer autotuning, a 512-token maximum CUDA graph capture size, and Mooncake-backed prefix replay.

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

1. `fingerprint_agg_w0.json` reports the vLLM version from the pinned image and Dynamo `1.2.1`.
2. `benchmark.out` reports the container `python3`, not a Lustre virtual environment.
3. `benchmark.out` reaches the decode-only window and exits successfully.
4. Both `.nsys-rep` files are non-empty and `nsys stats` can open them.
5. Worker logs contain no CUDA OOM, Mooncake load failure, lease expiry, or transfer timeout.
