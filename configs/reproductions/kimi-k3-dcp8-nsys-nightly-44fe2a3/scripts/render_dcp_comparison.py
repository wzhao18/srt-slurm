import argparse
import json
from pathlib import Path

GROUPS = {
    "Dense/projection GEMMs": (
        "Attention projection GEMMs",
        "Layer 0 dense MLP",
    ),
    "MoE expert GEMMs": (
        "Routed experts",
        "Shared experts MLP",
    ),
    "Communication": (
        "Attention/DCP communication",
        "MoE communication",
    ),
    "Fused EP MoE: dispatch + experts + combine": (
        "Fused routed experts + EP communication",
        "Fused routed + shared experts + EP communication",
    ),
    "MoE routing/preparation": (
        "MoE routing, latent projections, and tail",
    ),
    "MLA attention": ("MLA attention core and preparation",),
    "KDA linear attention": ("KDA attention core and preparation",),
    "AttnRes mixing": ("AttnRes mixing",),
    "Other kernels": ("Other target-model kernels",),
    "Speculative sampling/draft": (
        "DSpark draft, sampling, and metadata",
    ),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def activity(summary: dict, categories: tuple[str, ...]) -> float:
    values = summary["activity_ms"]
    return sum(values.get(category, 0.0) for category in categories)


def format_activity(summary: dict, categories: tuple[str, ...]) -> str:
    value = activity(summary, categories)
    if value == 0:
        return "—"
    total = summary["summed_gpu_activity_ms"]
    return f"{value:.2f} ms ({100 * value / total:.1f}%)"


def format_tokens(value: int | None) -> str:
    return "—" if value is None else f"{value:,} tokens"


def render(title: str, mxfp4: dict, nvfp4: dict) -> None:
    print(f"### {title}")
    print()
    print("| Kernel group | MXFP4 DCP8 | NVFP4 DCP8 |")
    print("| --- | ---: | ---: |")
    for label, categories in GROUPS.items():
        print(
            f"| {label} | {format_activity(mxfp4, categories)} | "
            f"{format_activity(nvfp4, categories)} |"
        )
    print(
        "| **Summed GPU activity** | "
        f"**{mxfp4['summed_gpu_activity_ms']:.2f} ms** | "
        f"**{nvfp4['summed_gpu_activity_ms']:.2f} ms** |"
    )
    print(
        '| **Measured decode-step latency, p50** | '
        f"**{mxfp4['decode_step_latency_p50_ms']:.2f} ms** | "
        f"**{nvfp4['decode_step_latency_p50_ms']:.2f} ms** |"
    )
    print(
        '| **Decode-step latency, p90** | '
        f"**{mxfp4['decode_step_latency_p90_ms']:.2f} ms** | "
        f"**{nvfp4['decode_step_latency_p90_ms']:.2f} ms** |"
    )
    print(
        "| Logged KV capacity | "
        f"{format_tokens(mxfp4['logged_kv_capacity_tokens'])} | "
        f"{format_tokens(nvfp4['logged_kv_capacity_tokens'])} |"
    )
    print(
        "| Deployment-total KV capacity | "
        f"{format_tokens(mxfp4['deployment_total_kv_capacity_tokens'])} | "
        f"{format_tokens(nvfp4['deployment_total_kv_capacity_tokens'])} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--mxfp4", type=Path, required=True)
    parser.add_argument("--nvfp4", type=Path, required=True)
    args = parser.parse_args()
    mxfp4 = load(args.mxfp4)
    nvfp4 = load(args.nvfp4)
    if mxfp4["variant"] != "mxfp4_dcp8":
        raise ValueError(f"Unexpected MXFP4 variant: {mxfp4['variant']}")
    if nvfp4["variant"] != "nvfp4_dcp8":
        raise ValueError(f"Unexpected NVFP4 variant: {nvfp4['variant']}")
    render(args.title, mxfp4, nvfp4)


if __name__ == "__main__":
    main()
