# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Process registry for managing and monitoring spawned processes.

This module provides lifecycle management for srun processes, including:
- Process registration and tracking
- Health monitoring via background thread
- Graceful cleanup on exit or failure
"""

import logging
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Terminal markers a worker may log while its OS process stays alive. e.g.
# dynamo.vllm keeps its runtime (and etcd leases) running after the vLLM
# EngineCore fails to initialize, so the srun step never exits and the plain
# exit-code check never trips -- the readiness gate then polls until the health
# timeout (hours). Scanning worker logs for these lets the monitor fail fast.
# Keep these unambiguous and non-recoverable to avoid false positives.
FATAL_LOG_MARKERS: tuple[str, ...] = (
    "EngineCore failed to start",
    "Engine core initialization failed",
    "Worker failed with error",
    "torch.OutOfMemoryError",
)


@dataclass
class ManagedProcess:
    """A process managed by the registry.

    Attributes:
        name: Human-readable process name (e.g., "prefill_0", "decode_1")
        popen: The subprocess.Popen object
        log_file: Path to the process log file
        node: Node hostname where the process runs
        critical: If True, failure triggers full cleanup
    """

    name: str
    popen: subprocess.Popen
    log_file: Path | None = None
    node: str | None = None
    critical: bool = True

    @property
    def is_running(self) -> bool:
        """Check if process is still running."""
        return self.popen.poll() is None

    @property
    def exit_code(self) -> int | None:
        """Get exit code if process has exited, None otherwise."""
        return self.popen.poll()

    def terminate(self, timeout: float = 10.0) -> None:
        """Terminate the process gracefully, then kill if needed."""
        if not self.is_running:
            return

        self.popen.terminate()
        try:
            self.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Process %s did not terminate, killing...", self.name)
            self.popen.kill()
            self.popen.wait(timeout=5)


# Type alias for named process collections
NamedProcesses = dict[str, ManagedProcess]


class ProcessRegistry:
    """Registry for managing multiple processes with health monitoring.

    Features:
    - Tracks all spawned processes by name
    - Background thread monitors for unexpected exits
    - Graceful cleanup on signal or failure
    - Detailed failure reporting with log tails

    Usage:
        registry = ProcessRegistry(job_id="12345")
        registry.add_process(managed_proc)
        # ... run workload ...
        if registry.check_failures():
            registry.cleanup()
    """

    def __init__(self, job_id: str):
        """Initialize the registry.

        Args:
            job_id: SLURM job ID for logging
        """
        self.job_id = job_id
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()
        self._failed_processes: list[str] = []
        # Byte offset scanned so far per process log, for incremental fatal-
        # marker detection (see _scan_log_for_fatal_marker).
        self._log_scan_pos: dict[str, int] = {}

    def add_process(self, process: ManagedProcess) -> None:
        """Add a process to the registry.

        Args:
            process: ManagedProcess to track
        """
        with self._lock:
            if process.name in self._processes:
                logger.warning("Replacing existing process '%s' in registry", process.name)
            self._processes[process.name] = process
            logger.debug("Registered process: %s (pid=%d)", process.name, process.popen.pid)

    def add_processes(self, processes: NamedProcesses) -> None:
        """Add multiple processes to the registry.

        Args:
            processes: Dict mapping names to ManagedProcess objects
        """
        for name, proc in processes.items():
            # Ensure the name matches
            if proc.name != name:
                proc = ManagedProcess(
                    name=name,
                    popen=proc.popen,
                    log_file=proc.log_file,
                    node=proc.node,
                    critical=proc.critical,
                )
            self.add_process(proc)

    def check_failures(self) -> bool:
        """Check if any critical process has failed.

        A process counts as failed if it either exited with a non-zero code, or
        is still running but logged a terminal engine-failure marker (a hung
        worker -- see FATAL_LOG_MARKERS). The latter catches dynamo.vllm workers
        that keep their runtime alive after the vLLM EngineCore dies, which would
        otherwise stall the readiness gate until the health-check timeout.

        Returns:
            True if any critical process has failed.
        """
        with self._lock:
            for name, proc in self._processes.items():
                if not proc.critical or name in self._failed_processes:
                    continue
                if not proc.is_running:
                    exit_code = proc.exit_code
                    if exit_code != 0:
                        self._failed_processes.append(name)
                        logger.error(
                            "Critical process '%s' exited with code %d",
                            name,
                            exit_code,
                        )
                else:
                    marker = self._scan_log_for_fatal_marker(name, proc.log_file)
                    if marker is not None:
                        self._failed_processes.append(name)
                        logger.error(
                            "Critical process '%s' logged a fatal error and is not "
                            "recovering (marker: %r)",
                            name,
                            marker,
                        )

            return len(self._failed_processes) > 0

    def _scan_log_for_fatal_marker(self, name: str, log_file: Path | None) -> str | None:
        """Scan a process log's newly-appended bytes for a fatal marker.

        Only content written since the previous scan is read, so a marker is
        caught when it first appears regardless of later log spam, and repeated
        scans stay cheap. Returns the matched marker, or None. Never raises --
        log I/O problems must not take down the monitor.
        """
        if log_file is None:
            return None
        try:
            if not log_file.exists():
                return None
            size = log_file.stat().st_size
            pos = self._log_scan_pos.get(name, 0)
            if size < pos:  # log rotated or truncated
                pos = 0
            if size <= pos:
                return None
            with log_file.open("rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
            self._log_scan_pos[name] = size
        except OSError:
            return None
        text = chunk.decode("utf-8", errors="replace")
        for marker in FATAL_LOG_MARKERS:
            if marker in text:
                return marker
        return None

    def cleanup(self) -> None:
        """Terminate all registered processes."""
        with self._lock:
            logger.info("Cleaning up %d processes...", len(self._processes))
            for name, proc in self._processes.items():
                if proc.is_running:
                    logger.debug("Terminating process: %s", name)
                    try:
                        proc.terminate()
                    except Exception as e:
                        logger.warning("Failed to terminate %s: %s", name, e)

    def print_failure_details(self, tail_lines: int = 50) -> None:
        """Print detailed failure information including log tails.

        Args:
            tail_lines: Number of lines to show from each failed process log
        """
        if not self._failed_processes:
            return

        logger.error("=" * 60)
        logger.error("FAILURE DETAILS")
        logger.error("=" * 60)

        with self._lock:
            for name in self._failed_processes:
                proc = self._processes.get(name)
                if not proc:
                    continue

                logger.error("\n--- Process: %s ---", name)
                logger.error("Exit code: %s", proc.exit_code)
                logger.error("Node: %s", proc.node or "unknown")
                logger.error("Log file: %s", proc.log_file or "none")

                # Tail the log file if available
                if proc.log_file and proc.log_file.exists():
                    try:
                        lines = proc.log_file.read_text().splitlines()
                        if lines:
                            logger.error("\nLast %d lines of log:", tail_lines)
                            for line in lines[-tail_lines:]:
                                logger.error("  %s", line)
                    except Exception as e:
                        logger.error("Could not read log file: %s", e)

        logger.error("=" * 60)

    def get_process(self, name: str) -> ManagedProcess | None:
        """Get a process by name."""
        with self._lock:
            return self._processes.get(name)

    def get_all_processes(self) -> dict[str, ManagedProcess]:
        """Get a copy of all registered processes."""
        with self._lock:
            return dict(self._processes)

    @property
    def process_count(self) -> int:
        """Get the number of registered processes."""
        with self._lock:
            return len(self._processes)


def setup_signal_handlers(
    stop_event: threading.Event,
    registry: ProcessRegistry,
) -> None:
    """Setup signal handlers for graceful shutdown.

    Args:
        stop_event: Event to signal shutdown
        registry: ProcessRegistry to cleanup on signal
    """

    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning("Received signal %s, initiating cleanup...", sig_name)
        stop_event.set()
        registry.cleanup()
        sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def start_process_monitor(
    stop_event: threading.Event,
    registry: ProcessRegistry,
    poll_interval: float = 2.0,
) -> threading.Thread:
    """Start a background thread that monitors for process failures.

    Args:
        stop_event: Event that signals the monitor to stop
        registry: ProcessRegistry to monitor
        poll_interval: Seconds between checks

    Returns:
        The monitoring thread (already started)
    """

    def monitor_loop():
        while not stop_event.is_set():
            if registry.check_failures():
                logger.error("Critical process failure detected!")
                stop_event.set()
                registry.cleanup()
                sys.exit(1)
            time.sleep(poll_interval)

    thread = threading.Thread(
        target=monitor_loop,
        daemon=True,
        name="ProcessMonitor",
    )
    thread.start()
    return thread
