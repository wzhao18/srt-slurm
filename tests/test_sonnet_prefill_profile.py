"""Do not accept a cold extension as a cached-prefix prefill trace."""

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("cached_tokens,accepted", [(130944, True), (0, False)])
def test_prefill_extension_requires_seeded_prefix(monkeypatch, cached_tokens, accepted):
    script = Path(__file__).resolve().parents[1] / "src/srtctl/benchmarks/scripts/profiling/prefill_client.py"
    spec = importlib.util.spec_from_file_location("prefill_profile", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = MagicMock()
    (root / "prefill-prompts.json").read_text.return_value = json.dumps({"extend": "Sonnet extension"})
    monkeypatch.setattr(module, "Path", lambda _: root)
    monkeypatch.setattr(module.sys, "argv", [str(script), "extend"])
    monkeypatch.setenv("SRT_FRONTEND_HOST", "localhost")
    monkeypatch.setenv("SRT_FRONTEND_PORT", "8000")
    monkeypatch.setenv("PROFILE_ISL", "131072")
    monkeypatch.setenv("PROFILE_MODEL_NAME", "test-model")

    def complete(request, timeout):
        payload = json.loads(request.data)
        assert payload["prompt"] == "Sonnet extension"
        assert payload["max_tokens"] == 1
        return io.BytesIO(json.dumps({"usage": {"prompt_tokens_details": {"cached_tokens": cached_tokens}}}).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", complete)
    if accepted:
        module.main()
    else:
        with pytest.raises(ValueError, match="Expected seeded-prefix reuse"):
            module.main()
