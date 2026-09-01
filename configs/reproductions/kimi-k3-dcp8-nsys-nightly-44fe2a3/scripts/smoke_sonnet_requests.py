"""Validate exact-length Sonnet request generation and replay caching."""

import argparse
import os
import random
import runpy
import sys
import tempfile
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import NoReturn


def _unsupported_loader(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("The Sonnet smoke test does not support optional datasets")


def _stub_optional_modules() -> None:
    datasets_module = types.ModuleType("datasets")
    datasets_module.__spec__ = ModuleSpec("datasets", loader=None)
    datasets_module.load_dataset = _unsupported_loader
    sys.modules["datasets"] = datasets_module

    pandas_module = types.ModuleType("pandas")
    pandas_module.__spec__ = ModuleSpec("pandas", loader=None)
    pandas_module.read_csv = _unsupported_loader
    sys.modules["pandas"] = pandas_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="/model")
    parser.add_argument(
        "--corpus",
        default=(
            "/configs/reproductions/kimi-k3-dcp8-nsys-nightly-44fe2a3/"
            "assets/shakespeare.txt"
        ),
    )
    args = parser.parse_args()

    _stub_optional_modules()
    benchmark_path = Path(
        os.environ.get(
            "KIMI_WIDEEP_SA_BENCHMARK",
            "/srtctl-benchmarks/sa-bench/benchmark_serving.py",
        )
    )
    sys.path.insert(0, str(benchmark_path.parent))
    benchmark = runpy.run_path(str(benchmark_path), run_name="_sa_benchmark")

    tokenizer = benchmark["load_tokenizer"](
        args.tokenizer,
        tokenizer_mode="auto",
        trust_remote_code=True,
        custom_tokenizer=None,
    )
    random.seed(0)
    sampled = benchmark["sample_sonnet_requests"](
        dataset_path=args.corpus,
        num_requests=2,
        input_len=512,
        output_len=17,
        prefix_len=128,
        tokenizer=tokenizer,
        exact_input_len=True,
    )
    finalized = [
        (prompt_formatted, prompt_len, output_len, None)
        for _, prompt_formatted, prompt_len, output_len, _ in sampled
    ]
    if [prompt_len for _, prompt_len, _, _ in finalized] != [512, 512]:
        raise AssertionError("Sonnet prompts were not exactly 512 tokens")

    metadata = argparse.Namespace(
        dataset_name="sonnet",
        dataset_path=args.corpus,
        num_prompts=2,
        seed=0,
        sonnet_input_len=512,
        sonnet_prefix_len=128,
        sonnet_exact_input_len=True,
        sonnet_output_len=17,
    )
    with tempfile.TemporaryDirectory() as temporary_dir:
        request_cache = str(Path(temporary_dir) / "requests.json")
        benchmark["save_input_requests"](request_cache, finalized, metadata)
        loaded = benchmark["load_input_requests"](request_cache, metadata)

    if loaded != finalized:
        raise AssertionError("Saved and loaded Sonnet requests differ")
    for prompt, prompt_len, _, _ in loaded:
        actual_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if actual_len != prompt_len:
            raise AssertionError(
                f"Cached prompt length changed: expected {prompt_len}, got {actual_len}"
            )

    print("SONNET_CLIENT_SMOKE_OK")


if __name__ == "__main__":
    main()
