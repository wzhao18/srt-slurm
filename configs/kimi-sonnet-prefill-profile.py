"""Seed a Sonnet prefix, then extend it for a short prefill-only capture."""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def main() -> None:
    phase = sys.argv[1]
    root = Path("/logs/profile-benchmark")
    root.mkdir(parents=True, exist_ok=True)
    seed_len = int(os.environ.get("PROFILE_ISL", "131072"))
    extension_len = int(os.environ.get("PROFILE_EXTENSION_TOKENS", "49152"))
    if phase == "prepare":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.environ.get("PROFILE_TOKENIZER_PATH", "/model"))
        corpus = Path(os.environ["PROFILE_TEXT_CORPUS_PATH"]).read_text()
        tokens = tokenizer.encode(corpus, add_special_tokens=False)
        if not tokens:
            raise ValueError("Sonnet corpus is empty")
        target = seed_len + extension_len
        tokens = (tokens * ((target + len(tokens) - 1) // len(tokens)))[:target]
        prompts = {
            "seed": tokenizer.decode(tokens[:seed_len], skip_special_tokens=False),
            "extend": tokenizer.decode(tokens, skip_special_tokens=False),
        }
        seed_ids = tokenizer.encode(prompts["seed"], add_special_tokens=False)
        extend_ids = tokenizer.encode(prompts["extend"], add_special_tokens=False)
        if len(seed_ids) != seed_len or len(extend_ids) != target:
            raise ValueError("Tokenizer round trip changed the requested Sonnet lengths")
        shared = next((i for i, (a, b) in enumerate(zip(seed_ids, extend_ids)) if a != b), len(seed_ids))
        if shared < seed_len - 128:
            raise ValueError(f"Extension changes the seeded prefix: shared={shared}, seed={seed_len}")
        (root / "prefill-prompts.json").write_text(json.dumps(prompts))
        print(f"Prepared Sonnet: seed={len(seed_ids)}, extension={len(extend_ids) - len(seed_ids)}, shared={shared}")
        return
    if phase not in ("seed", "extend"):
        raise ValueError(f"Unknown phase: {phase}")
    prompts = json.loads((root / "prefill-prompts.json").read_text())
    payload = {
        "model": os.environ.get("PROFILE_MODEL_NAME", "moonshotai/Kimi-K3"),
        "prompt": prompts[phase],
        "max_tokens": 1,
        "temperature": 0,
        "ignore_eos": True,
    }
    endpoint = f"http://{os.environ['SRT_FRONTEND_HOST']}:{os.environ['SRT_FRONTEND_PORT']}/v1/completions"
    request = urllib.request.Request(endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=1800) as response:
        reply = json.load(response)
    (root / f"prefill-{phase}-response.json").write_text(json.dumps(reply, indent=2))
    usage = reply.get("usage", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    print(f"{phase}: seconds={time.monotonic() - start:.3f}, usage={usage}", flush=True)
    if phase == "extend" and cached is not None and cached < seed_len - 256:
        raise ValueError(f"Expected seeded-prefix reuse, got cached_tokens={cached}")
    if phase == "extend" and cached is None:
        print("Cache-hit details absent in frontend response; verify engine prefix-cache metrics before accepting trace.")


if __name__ == "__main__":
    main()
