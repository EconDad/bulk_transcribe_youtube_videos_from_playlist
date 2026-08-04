from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Mapping


def request_json(
    base_url: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}: {exc.reason}"
        ) from exc

    result = json.loads(raw)
    if not isinstance(result, dict):
        raise RuntimeError("Ollama returned a non-object response")
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    return result


def installed_models(base_url: str) -> set[str]:
    payload = request_json(base_url, "/api/tags")
    names: set[str] = set()

    for item in payload.get("models") or []:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name") or item.get("model")
        if name:
            names.add(str(name).removesuffix(":latest"))

    return names


def run_mode(
    *,
    base_url: str,
    model: str,
    think: bool,
    purpose: str,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "purpose": {"type": "string"},
            "mode": {"type": "string"},
        },
        "required": ["purpose", "mode"],
    }

    response = request_json(
        base_url,
        "/api/chat",
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only structured JSON matching the schema."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Set purpose to {purpose!r}. "
                        f"Set mode to {'thinking' if think else 'direct'}."
                    ),
                },
            ],
            "stream": False,
            "think": think,
            "format": schema,
            "keep_alive": "15m",
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": 8192,
                "num_predict": 300,
            },
        },
    )

    message = response.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("Response has no message object")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Response has no final message content")

    parsed = json.loads(content)
    if parsed.get("purpose") != purpose:
        raise RuntimeError(
            f"Unexpected purpose value: {parsed.get('purpose')!r}"
        )

    thinking = message.get("thinking")
    thinking_chars = (
        len(thinking)
        if isinstance(thinking, str)
        else 0
    )

    if think and thinking_chars == 0:
        print(
            "WARNING: think=true returned no separate thinking text; "
            "the installed Ollama/model build may not expose it.",
            file=sys.stderr,
        )

    return {
        "think": think,
        "thinking_chars": thinking_chars,
        "content": parsed,
        "done_reason": response.get("done_reason"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Qwen3 structured output in both v4 modes."
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--model",
        default="qwen3:8b",
    )
    args = parser.parse_args()

    names = installed_models(args.host)
    normalized_model = args.model.removesuffix(":latest")

    if normalized_model not in names:
        available = ", ".join(sorted(names)) or "(none)"
        print(
            f"ERROR: {args.model!r} is not installed. "
            f"Available models: {available}",
            file=sys.stderr,
        )
        return 2

    extraction = run_mode(
        base_url=args.host,
        model=args.model,
        think=True,
        purpose="extraction",
    )
    narrative = run_mode(
        base_url=args.host,
        model=args.model,
        think=False,
        purpose="narrative",
    )

    print("PASS: Qwen3 v4 mode routing")
    print(
        "Extraction:",
        json.dumps(extraction, indent=2, sort_keys=True),
    )
    print(
        "Narrative:",
        json.dumps(narrative, indent=2, sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
