import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

KDA_LAYERS = 69
MODEL_LAYERS = 93


@dataclass(frozen=True)
class Kernel:
    device: int
    start: int
    end: int
    name: str
    node: int | None
    graph: int | None

    @property
    def duration(self) -> int:
        return self.end - self.start


def is_attn_res(name: str) -> bool:
    return "attn_res" in name.lower()


def is_communication(name: str) -> bool:
    lower = name.lower()
    return any(
        marker in lower
        for marker in (
            "nccl",
            "allreduce",
            "all_reduce",
            "allgather",
            "all_gather",
            "reducescatter",
            "reduce_scatter",
            "alltoall",
            "all_to_all",
            "direct_dcp",
            "correct_attn_cp_out",
        )
    )


def is_gemm(name: str) -> bool:
    lower = name.lower()
    return any(
        marker in lower
        for marker in (
            "deep_gemm",
            "nvjet_",
            "bmm_",
            "gemm_rs_ar",
            "splitkreduce_kernel",
            "cublas_gemm",
        )
    )


def is_mla(name: str) -> bool:
    lower = name.lower()
    return any(
        marker in lower
        for marker in (
            "tokenspeed_mla",
            "mladecode",
            "fusedkimik3mladecode",
            "concat_and_cache_mla",
            "_fused_q_kv_rmsnorm",
        )
    )


def is_deepgemm_mega_moe(name: str) -> bool:
    return "deep_gemm::sm100_fp8_fp4_mega_moe_impl" in name


def is_flashinfer_mega_moe(name: str) -> bool:
    lower = name.lower()
    return "kernel_cutlass_fc1fc2_kernel_impl" in lower and "megamoe" in lower


def is_routed_expert(name: str) -> bool:
    lower = name.lower()
    return (
        "bmm_mxe4m3_mxe2m1" in lower
        or "bmm_bfloat16_mxe2m1" in lower
        or "bmm_e2m1_e2m1" in lower
        or "bmm_bfloat16_e2m1" in lower
    )


def classify_dcp_shared(segment: list[Kernel]) -> dict[int, str]:
    categories: dict[int, str] = {}
    for activation_index, activation in enumerate(segment):
        if "situ_and_mul_kernel" not in activation.name.lower():
            continue
        assert activation.node is not None
        categories[activation.node] = "Shared experts MLP"
        for kernel in reversed(segment[:activation_index]):
            if is_gemm(kernel.name):
                assert kernel.node is not None
                categories[kernel.node] = "Shared experts MLP"
                break
        for kernel in segment[activation_index + 1 :]:
            if is_gemm(kernel.name):
                assert kernel.node is not None
                categories[kernel.node] = "Shared experts MLP"
                break
    return categories


def classify_dep_shared(segment: list[Kernel]) -> dict[int, str]:
    categories: dict[int, str] = {}
    in_shared = False
    for kernel in segment:
        lower = kernel.name.lower()
        if "nvjet_sm103_tst_128x24_64x11_4x1_v_bz_splitk_tnt" in lower:
            in_shared = True
            categories[kernel.node] = "Shared experts MLP"
        elif in_shared and "splitkreduce_kernel" in lower:
            categories[kernel.node] = "Shared experts MLP"
        elif "situ_and_mul_kernel" in lower:
            in_shared = True
            categories[kernel.node] = "Shared experts MLP"
        elif "nvjet_sm103_tst_64x24_64x16_4x1_v_bz_tnt" in lower:
            categories[kernel.node] = "Shared experts MLP"
            in_shared = False
    return categories


def classify_moe_segment(
    segment: list[Kernel], variant: str
) -> dict[int, str]:
    categories: dict[int, str] = {}
    dep_shared = classify_dep_shared(segment) if variant.endswith("dep16") else {}
    dcp_shared = classify_dcp_shared(segment) if variant.endswith("dcp8") else {}
    for kernel in segment:
        assert kernel.node is not None
        name = kernel.name
        if is_flashinfer_mega_moe(name):
            category = "Fused routed + shared experts + EP communication"
        elif is_deepgemm_mega_moe(name):
            category = "Fused routed experts + EP communication"
        elif is_communication(name):
            category = "MoE communication"
        elif is_routed_expert(name):
            category = "Routed experts"
        elif kernel.node in dep_shared or kernel.node in dcp_shared:
            category = "Shared experts MLP"
        else:
            category = "MoE routing, latent projections, and tail"
        categories[kernel.node] = category
    return categories


def classify_target_nodes(
    first_execution: list[Kernel], variant: str
) -> tuple[dict[int, str], tuple[int, int]]:
    categories: dict[int, str] = {}
    phase = "prelude"
    attention_segment: list[Kernel] = []
    ffn_segment: list[Kernel] = []
    attention_segments = 0
    ffn_segments = 0

    def flush_attention() -> None:
        nonlocal attention_segment, attention_segments
        if not attention_segment:
            return
        segment_category = (
            "MLA attention core and preparation"
            if any(is_mla(kernel.name) for kernel in attention_segment)
            else "KDA attention core and preparation"
        )
        for kernel in attention_segment:
            assert kernel.node is not None
            if is_communication(kernel.name):
                category = "Attention/DCP communication"
            elif is_gemm(kernel.name):
                category = "Attention projection GEMMs"
            else:
                category = segment_category
            categories[kernel.node] = category
        attention_segments += 1
        attention_segment = []

    def flush_ffn() -> None:
        nonlocal ffn_segment, ffn_segments
        if not ffn_segment:
            return
        if ffn_segments == 0:
            for kernel in ffn_segment:
                assert kernel.node is not None
                categories[kernel.node] = "Layer 0 dense MLP"
        else:
            categories.update(classify_moe_segment(ffn_segment, variant))
        ffn_segments += 1
        ffn_segment = []

    for kernel in first_execution:
        assert kernel.node is not None
        if is_attn_res(kernel.name):
            if phase == "attention":
                flush_attention()
                phase = "ffn"
            else:
                if phase == "ffn":
                    flush_ffn()
                phase = "attention"
            categories[kernel.node] = "AttnRes mixing"
        elif phase == "attention":
            attention_segment.append(kernel)
        elif phase == "ffn":
            ffn_segment.append(kernel)
        else:
            categories[kernel.node] = "Other target-model kernels"

    if phase == "attention":
        flush_attention()
    elif phase == "ffn":
        flush_ffn()

    if attention_segments != MODEL_LAYERS or ffn_segments != MODEL_LAYERS:
        raise ValueError(
            "Expected 93 attention and FFN segments, found "
            f"{attention_segments} and {ffn_segments}."
        )
    if len(categories) != len(first_execution):
        raise ValueError(
            f"Classified {len(categories)} of {len(first_execution)} target nodes."
        )
    return categories, (attention_segments, ffn_segments)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def find_model_end(step: list[Kernel]) -> int:
    phase = "prelude"
    attention_segments = 0
    ffn_segments = 0
    for index, kernel in enumerate(step):
        if not is_attn_res(kernel.name):
            continue
        if phase == "attention":
            attention_segments += 1
            phase = "ffn"
        else:
            if phase == "ffn":
                ffn_segments += 1
            phase = "attention"
        if ffn_segments == MODEL_LAYERS:
            if attention_segments != MODEL_LAYERS:
                raise ValueError(
                    "Reached the final FFN with "
                    f"{attention_segments} attention segments."
                )
            return index + 1
    raise ValueError(
        "Could not find 93 complete attention/FFN layer pairs in a decode step."
    )


def analyze(
    traces: list[Path],
    variant: str,
    aggregation: str,
    output_json: Path | None,
    worker_logs: list[Path],
) -> None:
    rows_by_device: dict[tuple[int, int], list[Kernel]] = defaultdict(list)
    for trace_index, trace in enumerate(traces):
        with sqlite3.connect(f"file:{trace.resolve()}?mode=ro", uri=True) as db:
            rows = [
                Kernel(*row)
                for row in db.execute(
                    """
                    SELECT k.deviceId, k.start, k.end, s.value,
                           k.graphNodeId, k.graphId
                    FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                    JOIN StringIds AS s ON s.id = k.demangledName
                    ORDER BY k.deviceId, k.start
                    """
                )
            ]
        for kernel in rows:
            rows_by_device[(trace_index, kernel.device)].append(kernel)

    totals_ns: dict[str, int] = defaultdict(int)
    activity_by_step: list[dict[str, int]] = []
    cadence_ms: list[float] = []
    replay_counts: list[int] = []

    for device, device_rows in rows_by_device.items():
        recurrent_indices = [
            index
            for index, kernel in enumerate(device_rows)
            if kernel.name == "fused_recurrent_kda_fwd_kernel"
        ]
        recurrent_count = len(recurrent_indices)
        if recurrent_count % KDA_LAYERS:
            raise ValueError(
                f"Device {device}: {recurrent_count} KDA kernels is not "
                f"divisible by {KDA_LAYERS}."
            )
        replays = recurrent_count // KDA_LAYERS
        replay_counts.append(replays)
        first_recurrent_indices = recurrent_indices[::KDA_LAYERS]
        target_start_indices: list[int] = []
        previous_start = -1
        for recurrent_index in first_recurrent_indices:
            start_index = recurrent_index
            while start_index > previous_start + 1:
                start_index -= 1
                if is_attn_res(device_rows[start_index].name):
                    break
            if not is_attn_res(device_rows[start_index].name):
                raise ValueError(
                    f"Device {device}: could not find layer-0 AttnRes boundary."
                )
            target_start_indices.append(start_index)
            previous_start = start_index

        for step_index in range(len(target_start_indices) - 1):
            start_index = target_start_indices[step_index]
            next_start_index = target_start_indices[step_index + 1]
            step = device_rows[start_index:next_start_index]
            model_end = find_model_end(step)
            model_rows = [
                Kernel(
                    kernel.device,
                    kernel.start,
                    kernel.end,
                    kernel.name,
                    index,
                    kernel.graph,
                )
                for index, kernel in enumerate(step[:model_end])
            ]
            categories, _ = classify_target_nodes(model_rows, variant)
            step_activity: dict[str, int] = defaultdict(int)
            for kernel in model_rows:
                assert kernel.node is not None
                category = categories[kernel.node]
                totals_ns[category] += kernel.duration
                step_activity[category] += kernel.duration
            draft_activity = sum(
                kernel.duration for kernel in step[model_end:]
            )
            totals_ns["DSpark draft, sampling, and metadata"] += draft_activity
            step_activity["DSpark draft, sampling, and metadata"] += draft_activity
            activity_by_step.append(step_activity)
            cadence_ms.append(
                (
                    device_rows[next_start_index].start
                    - device_rows[start_index].start
                )
                / 1e6
            )

    if not activity_by_step:
        raise ValueError("No complete decode steps were found.")
    denominator = len(activity_by_step)
    ordered_categories = (
        "Attention projection GEMMs",
        "Attention/DCP communication",
        "MLA attention core and preparation",
        "KDA attention core and preparation",
        "AttnRes mixing",
        "Layer 0 dense MLP",
        "Routed experts",
        "Shared experts MLP",
        "Fused routed experts + EP communication",
        "Fused routed + shared experts + EP communication",
        "MoE communication",
        "MoE routing, latent projections, and tail",
        "Other target-model kernels",
        "DSpark draft, sampling, and metadata",
    )
    if aggregation == "mean":
        activity_ms = {
            category: totals_ns.get(category, 0) / denominator / 1e6
            for category in ordered_categories
        }
    else:
        activity_ms = {
            category: statistics.median(
                step.get(category, 0) for step in activity_by_step
            )
            / 1e6
            for category in ordered_categories
        }
    summed_ms = sum(activity_ms.values())
    print("Traces:")
    for trace in traces:
        print(f"  {trace}")
    print(f"Variant: {variant}")
    print(f"Devices: {len(rows_by_device)}")
    print(f"Replays/device: {replay_counts}")
    label = "Mean" if aggregation == "mean" else "Median"
    print(f"Aggregation: {aggregation}")
    print(f"| Kernel group | {label} activity/device-step | Share |")
    print("| --- | ---: | ---: |")
    for category in ordered_categories:
        duration = activity_ms.get(category, 0.0)
        if duration:
            print(
                f"| {category} | {duration:.3f} ms | "
                f"{100 * duration / summed_ms:.1f}% |"
            )
    print(f"| **Summed GPU activity** | **{summed_ms:.3f} ms** | **100.0%** |")
    cadence_p50 = statistics.median(cadence_ms)
    cadence_p90 = percentile(cadence_ms, 0.9)
    print(f"Measured decode cadence p50: {cadence_p50:.3f} ms")
    print(f"Measured decode cadence p90: {cadence_p90:.3f} ms")
    kv_capacities = {
        int(match.replace(",", ""))
        for worker_log in worker_logs
        for match in re.findall(
            r"GPU KV cache size: ([0-9,]+) tokens", worker_log.read_text()
        )
    }
    if len(kv_capacities) > 1:
        raise ValueError(f"Worker logs disagree on KV capacity: {kv_capacities}")
    kv_capacity = next(iter(kv_capacities), None)
    if worker_logs and kv_capacity is None:
        raise ValueError("Worker logs do not report GPU KV cache capacity.")
    if kv_capacity is not None:
        print(f"Logged KV capacity: {kv_capacity:,} tokens")
    if output_json is not None:
        payload = {
            "traces": [str(trace) for trace in traces],
            "variant": variant,
            "aggregation": aggregation,
            "devices": len(rows_by_device),
            "replays_per_device": replay_counts,
            "activity_ms": activity_ms,
            "summed_gpu_activity_ms": summed_ms,
            "decode_step_latency_p50_ms": cadence_p50,
            "decode_step_latency_p90_ms": cadence_p90,
            "logged_kv_capacity_tokens": kv_capacity,
            "deployment_total_kv_capacity_tokens": kv_capacity,
        }
        output_json.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, nargs="+")
    parser.add_argument(
        "--variant",
        required=True,
        choices=("mxfp4_dcp8", "mxfp4_dep16", "nvfp4_dcp8", "nvfp4_dep16"),
    )
    parser.add_argument(
        "--aggregation",
        choices=("mean", "median"),
        default="mean",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--worker-log", type=Path, action="append", default=[])
    args = parser.parse_args()
    analyze(
        args.trace,
        args.variant,
        args.aggregation,
        args.output_json,
        args.worker_log,
    )


if __name__ == "__main__":
    main()
