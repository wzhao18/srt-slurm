# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for process-specific setup-script context."""

from types import SimpleNamespace

from srtctl.cli.do_sweep import _build_setup_script_preamble
from srtctl.cli.mixins.worker_stage import WorkerStageMixin
from srtctl.frontends.dynamo import DynamoFrontend


def test_mooncake_master_setup_context_is_exported():
    preamble = _build_setup_script_preamble(
        "setup.sh", context="mooncake-master"
    )

    assert preamble is not None
    assert "export SRT_SETUP_CONTEXT=mooncake-master" in preamble


def test_worker_setup_context_is_exported():
    stage = WorkerStageMixin()
    stage.config = SimpleNamespace(
        setup_script="setup.sh",
        frontend=SimpleNamespace(type="none"),
        dynamo=SimpleNamespace(install=False),
    )

    assert "export SRT_SETUP_CONTEXT=worker" in stage._build_worker_preamble()


def test_frontend_setup_context_is_exported():
    config = SimpleNamespace(
        setup_script="setup.sh",
        dynamo=SimpleNamespace(install=False),
    )

    preamble = DynamoFrontend()._build_preamble(config)

    assert preamble is not None
    assert "export SRT_SETUP_CONTEXT=frontend" in preamble
