"""Run the SA random-token or Sonnet benchmark without optional datasets."""

import os
import runpy
import sys
import types
from pathlib import Path
from typing import NoReturn


def _unsupported_load_dataset(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("The wide-EP profile workload does not support HF datasets")


def _unsupported_read_csv(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("The DCP profile workload does not support CSV datasets")


benchmark_path = Path(
    os.environ.get(
        "KIMI_WIDEEP_SA_BENCHMARK",
        "/srtctl-benchmarks/sa-bench/benchmark_serving.py",
    )
)
datasets_module = types.ModuleType("datasets")
datasets_module.load_dataset = _unsupported_load_dataset
sys.modules["datasets"] = datasets_module
pandas_module = types.ModuleType("pandas")
pandas_module.read_csv = _unsupported_read_csv
sys.modules["pandas"] = pandas_module
sys.path.insert(0, str(benchmark_path.parent))
if "--skip-initial-test" not in sys.argv:
    sys.argv.append("--skip-initial-test")
runpy.run_path(str(benchmark_path), run_name="__main__")
