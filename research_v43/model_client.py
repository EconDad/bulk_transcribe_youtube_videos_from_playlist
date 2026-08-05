"""Observable JSON-only model client for research pipeline v4.3."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Mapping, Protocol
from urllib import error, request


class ModelClientError(RuntimeError):
    """Raised when a model request or JSON response fails."""


class JsonTransport(Protocol):
    def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """POST JSON and return the decoded response object."""


class UrllibJsonTransport:
    """Standard-library HTTP transport."""

    def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(
                http_request,
                timeout=timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            raise ModelClientError(
                f"Model HTTP request failed: {exc}"
            ) from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelClientError(
                "Model endpoint returned invalid JSON"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ModelClientError(
                "Model endpoint response must be a JSON object"
            )
        return decoded


@dataclass(frozen=True, slots=True)
class ModelInvocation:
    model: str
    think: bool
    num_ctx: int
    prompt_chars: int
    response_chars: int
    stage: str = "model"
    num_predict: int = 0
    keep_alive: str = ""
    elapsed_seconds: float = 0.0
    done_reason: str = ""
    eval_count: int = 0
    thinking_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "think": self.think,
            "num_ctx": self.num_ctx,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "stage": self.stage,
            "num_predict": self.num_predict,
            "keep_alive": self.keep_alive,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "done_reason": self.done_reason,
            "eval_count": self.eval_count,
            "thinking_chars": self.thinking_chars,
        }


@dataclass(frozen=True, slots=True)
class JsonModelResponse:
    payload: Mapping[str, Any]
    invocation: ModelInvocation


class OllamaJsonClient:
    """JSON-only Ollama client with bounded, observable generation."""

    def __init__(
        self,
        *,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        think: bool = True,
        num_ctx: int = 8192,
        timeout_seconds: float = 300.0,
        num_predict: int = 1536,
        keep_alive: str = "30m",
        transport: JsonTransport | None = None,
    ) -> None:
        normalized_host = host.rstrip("/")
        if not normalized_host:
            raise ValueError("host cannot be empty")
        if not model.strip():
            raise ValueError("model cannot be empty")
        if num_ctx < 1024:
            raise ValueError("num_ctx must be at least 1024")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if num_predict < 64:
            raise ValueError("num_predict must be at least 64")
        if not keep_alive.strip():
            raise ValueError("keep_alive cannot be empty")

        self.host = normalized_host
        self.model = model.strip()
        self.think = bool(think)
        self.num_ctx = int(num_ctx)
        self.timeout_seconds = float(timeout_seconds)
        self.num_predict = int(num_predict)
        self.keep_alive = keep_alive.strip()
        self.transport = transport or UrllibJsonTransport()

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage: str = "model",
        num_predict: int | None = None,
        think: bool | None = None,
    ) -> JsonModelResponse:
        if not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        if not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")
        normalized_stage = stage.strip() or "model"
        resolved_num_predict = (
            self.num_predict
            if num_predict is None
            else int(num_predict)
        )
        if resolved_num_predict < 64:
            raise ValueError("num_predict must be at least 64")
        resolved_think = self.think if think is None else bool(think)

        request_payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "think": resolved_think,
            "keep_alive": self.keep_alive,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
                "num_predict": resolved_num_predict,
            },
        }

        started = time.monotonic()
        try:
            response = self.transport.post_json(
                url=f"{self.host}/api/chat",
                payload=request_payload,
                timeout_seconds=self.timeout_seconds,
            )
        except ModelClientError as exc:
            elapsed = time.monotonic() - started
            raise ModelClientError(
                f"{normalized_stage} failed after {elapsed:.1f}s: {exc}"
            ) from exc
        elapsed = time.monotonic() - started

        message = response.get("message")
        if not isinstance(message, Mapping):
            raise ModelClientError(
                f"{normalized_stage}: Ollama response is missing message"
            )
        content = message.get("content")
        thinking = message.get("thinking")
        thinking_text = thinking if isinstance(thinking, str) else ""
        done_reason_raw = response.get("done_reason")
        done_reason = (
            done_reason_raw
            if isinstance(done_reason_raw, str)
            else ""
        )
        eval_count_raw = response.get("eval_count")
        eval_count = (
            eval_count_raw
            if isinstance(eval_count_raw, int)
            else 0
        )
        if not isinstance(content, str) or not content.strip():
            raise ModelClientError(
                f"{normalized_stage}: Ollama response message content is empty "
                f"(done_reason={done_reason or 'unknown'}, "
                f"eval_count={eval_count}, "
                f"thinking_chars={len(thinking_text)}, "
                f"num_predict={resolved_num_predict}, "
                f"think={str(resolved_think).lower()})"
            )

        try:
            decoded_content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelClientError(
                f"{normalized_stage}: Ollama message content is not valid JSON"
            ) from exc
        if not isinstance(decoded_content, Mapping):
            raise ModelClientError(
                f"{normalized_stage}: Ollama JSON content must be an object"
            )

        invocation = ModelInvocation(
            model=self.model,
            think=resolved_think,
            num_ctx=self.num_ctx,
            prompt_chars=len(system_prompt) + len(user_prompt),
            response_chars=len(content),
            stage=normalized_stage,
            num_predict=resolved_num_predict,
            keep_alive=self.keep_alive,
            elapsed_seconds=elapsed,
            done_reason=done_reason,
            eval_count=eval_count,
            thinking_chars=len(thinking_text),
        )
        return JsonModelResponse(
            payload=dict(decoded_content),
            invocation=invocation,
        )
