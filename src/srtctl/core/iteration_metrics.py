# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-worker iteration-level metrics extraction for vLLM workers.

Parses lines emitted by vLLM when launched with ``--enable-logging-iteration-details``:

    (EngineCore_DP0 pid=2498766) INFO 04-22 12:53:55 [core.py:359] Iteration(15769):
    0 context requests, 0 context tokens, 181 generation requests,
    181 generation tokens, iteration elapsed time: 21.24 ms

Discovers ``*_prefill_w*.out`` and ``*_decode_w*.out`` in the run's logs/ directory
and emits one PNG per worker (the per-worker view is more useful than aggregating
across many DP-EP workers).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Matches both single-engine (`EngineCore`) and DP (`EngineCore_DP0`) prefixes.
ITER_RE = re.compile(
    r"\(EngineCore(?:_DP(?P<dp>\d+))?\s+pid=\d+\)\s+INFO\s+"
    r"(?P<ts>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"\[[^\]]+\]\s+Iteration\((?P<iter>\d+)\):\s+"
    r"(?P<ctx_req>\d+)\s+context requests,\s+"
    r"(?P<ctx_tok>\d+)\s+context tokens,\s+"
    r"(?P<gen_req>\d+)\s+generation requests,\s+"
    r"(?P<gen_tok>\d+)\s+generation tokens,\s+"
    r"iteration elapsed time:\s+(?P<elapsed>[\d.]+)\s+ms"
)


def parse_iter_log(path: Path, year: int) -> list[dict[str, Any]]:
    """Parse iteration-detail records from a single worker log file."""
    records: list[dict[str, Any]] = []
    with path.open("r", errors="replace") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)
            m = ITER_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            records.append(
                {
                    "dp": int(d["dp"]) if d["dp"] is not None else 0,
                    "iter": int(d["iter"]),
                    "ctx_req": int(d["ctx_req"]),
                    "ctx_tok": int(d["ctx_tok"]),
                    "gen_req": int(d["gen_req"]),
                    "gen_tok": int(d["gen_tok"]),
                    "elapsed_ms": float(d["elapsed"]),
                    "ts": datetime.strptime(f"{year}-{d['ts']}", "%Y-%m-%d %H:%M:%S"),
                }
            )
    return records


def _bucketize(records: list[dict[str, Any]], bucket_sec: int):
    """Aggregate records into per-(bucket, dp) rows. Returns a pandas DataFrame."""
    import pandas as pd

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"])
    df["bucket"] = df["ts"].dt.floor(f"{bucket_sec}s")
    agg = (
        df.groupby(["bucket", "dp"])
        .agg(
            ctx_req_mean=("ctx_req", "mean"),
            gen_req_mean=("gen_req", "mean"),
            ctx_tok_sum=("ctx_tok", "sum"),
            gen_tok_sum=("gen_tok", "sum"),
            iters=("iter", "count"),
        )
        .reset_index()
    )
    agg["ctx_tps"] = agg["ctx_tok_sum"] / bucket_sec
    agg["gen_tps"] = agg["gen_tok_sum"] / bucket_sec
    return agg


def _plot_worker(agg, out_fig: Path, title: str) -> None:
    """Render a 2-row figure (batch, throughput) for a single worker."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(title, fontsize=11)

    if agg.empty:
        for ax in axes:
            ax.set_title("no iteration data")
            ax.grid(True, alpha=0.3)
    else:
        for dp, g in agg.groupby("dp"):
            label = f"DP{dp}"
            axes[0].plot(g["bucket"], g["gen_req_mean"], label=f"{label} gen", alpha=0.85)
            axes[0].plot(g["bucket"], g["ctx_req_mean"], label=f"{label} ctx", alpha=0.6, linestyle="--")
            axes[1].plot(g["bucket"], g["gen_tps"], label=f"{label} gen tok/s", alpha=0.85)
            axes[1].plot(g["bucket"], g["ctx_tps"], label=f"{label} ctx tok/s", alpha=0.6, linestyle="--")
        axes[0].set_title("batch (reqs per iter, avg over bucket)")
        axes[0].set_ylabel("reqs")
        axes[1].set_title("throughput (tokens/s)")
        axes[1].set_ylabel("tokens/s")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("time (engine local)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)


def discover_worker_logs(log_dir: Path) -> list[Path]:
    """Return prefill+decode worker log files in deterministic order."""
    prefill = sorted(log_dir.glob("*_prefill_w*.out"))
    decode = sorted(log_dir.glob("*_decode_w*.out"))
    return [*prefill, *decode]


def extract_and_plot(log_dir: Path, *, bucket_sec: int = 1, year: int | None = None) -> dict[str, Any]:
    """Extract iteration metrics for every worker log under ``log_dir``.

    Writes ``iteration_metrics_<worker>.png`` next to each log and a single
    ``iteration_metrics.json`` summary keyed by worker name. Skips silently
    (with a warning) if matplotlib or pandas aren't installed.

    Args:
        log_dir: The run's logs directory containing ``*_prefill_w*.out`` /
                 ``*_decode_w*.out`` files.
        bucket_sec: Aggregation window in seconds.
        year: Year to attach to the MM-DD timestamps in the log; defaults to
              this calendar year.

    Returns:
        A summary dict with per-worker ``iters`` / ``buckets`` counts.
    """
    try:
        import pandas  # noqa: F401
    except ImportError:
        logger.warning("pandas not installed — skipping iteration metrics extraction")
        return {"workers": {}, "skipped": "pandas-missing"}

    has_matplotlib = True
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        has_matplotlib = False
        logger.warning("matplotlib not installed — extracting JSON only, no figures")

    if year is None:
        year = datetime.now().year

    workers = discover_worker_logs(log_dir)
    if not workers:
        logger.info("No worker logs found in %s; nothing to extract", log_dir)
        return {"workers": {}}

    summary: dict[str, Any] = {"bucket_sec": bucket_sec, "workers": {}}
    for log_path in workers:
        worker_name = log_path.stem  # e.g. "bia0063_prefill_w0"
        records = parse_iter_log(log_path, year)
        agg = _bucketize(records, bucket_sec)

        summary["workers"][worker_name] = {
            "log": str(log_path.name),
            "iters": len(records),
            "buckets": int(len(agg)),
        }

        if not records:
            logger.info(
                "%s: no Iteration(...) lines found — skipping figure (was the engine launched "
                "with --enable-logging-iteration-details?)",
                worker_name,
            )
            continue

        if has_matplotlib:
            out_fig = log_dir / f"iteration_metrics_{worker_name}.png"
            _plot_worker(agg, out_fig, title=f"iteration metrics — {worker_name} (bucket={bucket_sec}s)")
            logger.info("Wrote %s (%d iters, %d buckets)", out_fig.name, len(records), len(agg))

    out_json = log_dir / "iteration_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", out_json.name)
    return summary
