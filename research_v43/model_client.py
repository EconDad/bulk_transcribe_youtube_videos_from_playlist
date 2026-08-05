"""Minimal JSON-only model client for research pipeline v4.3."""

from __future__ import annotations

from dataclasses import dataclass
import json
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "think": self.think,
            "num_ctx": self.num_ctx,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
        }


@dataclass(frozen=True, slots=True)
class JsonModelResponse:
    payload: Mapping[str, Any]
    invocation: ModelInvocation


class OllamaJsonClient:
    """JSON-only Ollama client with deterministic generation options."""

    def __init__(
        self,
        *,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        think: bool = True,
        num_ctx: int = 8192,
        timeout_seconds: float = 300.0,
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

        self.host = normalized_host
        self.model = model.strip()
        self.think = bool(think)
        self.num_ctx = int(num_ctx)
        self.timeout_seconds = float(timeout_seconds)
        self.transport = transport or UrllibJsonTransport()

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> JsonModelResponse:
        if not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        if not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")

        request_payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "think": self.think,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
            },
        }

        response = self.transport.post_json(
            url=f"{self.host}/api/chat",
            payload=request_payload,
            timeout_seconds=self.timeout_seconds,
        )

        message = response.get("message")
        if not isinstance(message, Mapping):
            raise ModelClientError("Ollama response is missing message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ModelClientError(
                "Ollama response message content is empty"
            )

        try:
            decoded_content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelClientError(
                "Ollama message content is not valid JSON"
            ) from exc
        if not isinstance(decoded_content, Mapping):
            raise ModelClientError(
                "Ollama JSON content must be an object"
            )

        invocation = ModelInvocation(
            model=self.model,
            think=self.think,
            num_ctx=self.num_ctx,
            prompt_chars=len(system_prompt) + len(user_prompt),
            response_chars=len(content),
        )
        return JsonModelResponse(
            payload=dict(decoded_content),
            invocation=invocation,
        )
