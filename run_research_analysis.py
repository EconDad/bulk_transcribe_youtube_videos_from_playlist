from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from youtube_research_analysis import (
    ResearchInputPackageWriter,
    ResearchManifestStore,
    ResearchPackageWriter,
    TranscriptSourcePackage,
)

PROMPT_VERSION = "phase4-ollama-qwen3-dual-mode-v4.1.1"
DEFAULT_EXTRACTION_MODEL = "qwen3:8b"
DEFAULT_NARRATIVE_MODEL = "qwen3:8b"
# Backward-compatible alias for callers that still import DEFAULT_MODEL.
DEFAULT_MODEL = DEFAULT_EXTRACTION_MODEL
DEFAULT_EXTRACTION_THINK = True
DEFAULT_NARRATIVE_THINK = False
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_NUM_CTX = 8192
DEFAULT_CHUNK_TOKEN_BUDGET = 1200
DEFAULT_TIMEOUT_SECONDS = 900
MAX_CITATION_SEGMENTS = 6
MAX_COPY_RUN_WORDS = 12
MAX_CHUNK_SPLIT_DEPTH = 3
MIN_SPLITTABLE_SEGMENTS = 4

CHUNK_SYSTEM_PROMPT = """You extract transcript-grounded concepts and reusable symbolic relationships from one transcript chunk.

Claim requirements:
- Write concise, self-contained English paraphrases.
- Explain why each concept matters within the lesson.
- Do not copy long transcript phrases.
- Do not add dates, amounts, entities, or conclusions absent from the cited segments.
- Cite one narrow supporting range of no more than six segments.
- Do not mention prompts, schemas, validation, evidence IDs, or citation IDs.

Formula requirements:
- Return only reusable symbolic equalities between named quantities.
- Each ASCII formula must use snake_case variables and an explicit arithmetic operator.
- Standalone values, dates, percentages, and monetary amounts are data, not formulas.
- Define every variable and describe the general relationship, not merely the video's numeric example.
- Mark a relationship as stated only when the speaker expresses the operation; use derived only when adjacent transcript statements clearly support it.
- Return an empty formulas array when the transcript chunk does not support a reusable relationship.
- Do not repeat examples or wording from these instructions.

Grounding requirements:
- Use only the supplied transcript.
- Preserve zero-based segment indexes.
- Return only JSON matching the supplied structured-output format.
- Return an empty caveats array.
"""

SYNTHESIS_SYSTEM_PROMPT = """You write a coherent research brief from transcript-grounded evidence notes.

Writing requirements:
- Synthesize the lesson rather than concatenating notes.
- Explain the video's purpose, progression, relationships, and practical conclusion.
- Use natural English with no foreign-language characters.
- Do not invent dates, amounts, entities, or facts.
- Do not mention prompts, schemas, evidence catalogs, validation, unavailable IDs, or machine-readable syntax.
- Do not use the same wording or citation set for multiple takeaways.
- Each takeaway must make one distinct point.
- Section headings must be short reader-facing topic labels.
- Use only the supplied evidence IDs, and attach only the one or two most relevant IDs to each takeaway and no more than three to each section.
- Return only JSON matching the supplied structured-output format.
"""




def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off"
    )


def _analysis_backend_label(
    extraction_model: str,
    narrative_model: str,
) -> str:
    if extraction_model == narrative_model:
        return f"ollama:{extraction_model}"
    return (
        f"ollama:extract={extraction_model};"
        f"narrative={narrative_model}"
    )


class OllamaError(RuntimeError):
    """Raised when the local Ollama service or model request fails."""


@dataclass(frozen=True)
class TranscriptChunk:
    chunk_index: int
    start_segment: int
    end_segment: int
    rendered_text: str
    estimated_tokens: int


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str
    start_segment: int
    end_segment: int


class OllamaClient:
    """Minimal stdlib client for Ollama's local HTTP API."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_HOST,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        normalized = base_url.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            normalized = "http://" + normalized
        self.base_url = normalized
        self.timeout_seconds = timeout_seconds

    def _request(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
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
                timeout=self.timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(
                f"Ollama HTTP {exc.code} for {path}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Start it with `ollama serve` or the system service. "
                f"Underlying error: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"Ollama request timed out after "
                f"{self.timeout_seconds} seconds"
            ) from exc

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(
                f"Ollama returned invalid JSON for {path}: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise OllamaError(
                f"Ollama returned a non-object response for {path}"
            )
        if result.get("error"):
            raise OllamaError(str(result["error"]))
        return result

    def list_models(self) -> list[str]:
        payload = self._request("/api/tags")
        models = payload.get("models") or []
        names: list[str] = []

        for item in models:
            if not isinstance(item, Mapping):
                continue
            name = item.get("name") or item.get("model")
            if name:
                names.append(str(name))

        return names

    def require_model(self, model: str) -> None:
        names = self.list_models()
        normalized = {name.removesuffix(":latest") for name in names}

        if (
            model not in names
            and model.removesuffix(":latest") not in normalized
        ):
            available = ", ".join(sorted(names)) or "(none)"
            raise OllamaError(
                f"Ollama model {model!r} is not installed. "
                f"Available models: {available}"
            )

    def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        num_ctx: int,
        num_predict: int,
        think: bool = False,
        keep_alive: str = "15m",
    ) -> dict[str, Any]:
        response = self._request(
            "/api/chat",
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": bool(think),
                "format": dict(schema),
                "keep_alive": keep_alive,
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                    "top_p": 0.9,
                    "repeat_penalty": 1.05,
                },
            },
        )

        message = response.get("message")
        if not isinstance(message, Mapping):
            raise OllamaError("Ollama response has no message object")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama response has no message content")

        return response


def _json_from_ollama_response(
    response: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    message = response.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None

    if not isinstance(content, str):
        raise OllamaError(f"{context} response has no JSON content")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        done_reason = response.get("done_reason")
        eval_count = response.get("eval_count")
        raise OllamaError(
            f"{context} response was not valid JSON: {exc}; "
            f"done_reason={done_reason!r}; eval_count={eval_count!r}; "
            f"content_chars={len(content)}"
        ) from exc

    if not isinstance(payload, dict):
        raise OllamaError(f"{context} response must be a JSON object")
    return payload


def _estimate_tokens(text: str) -> int:
    # Conservative approximation for English transcript text plus labels.
    return max(1, math.ceil(len(text) / 3.5))


def _format_timestamp(seconds: float) -> str:
    total_ms = round(float(seconds) * 1000)
    minutes, milliseconds = divmod(total_ms, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return (
            f"{hours:02d}:{minutes:02d}:"
            f"{whole_seconds:02d}.{milliseconds:03d}"
        )

    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _render_segment(index: int, segment: Mapping[str, Any]) -> str:
    return (
        f"[S{index:04d} | "
        f"{_format_timestamp(float(segment['start']))}-"
        f"{_format_timestamp(float(segment['end']))}] "
        f"{str(segment['text']).strip()}"
    )


def chunk_transcript(
    segments: Sequence[Mapping[str, Any]],
    *,
    token_budget: int = DEFAULT_CHUNK_TOKEN_BUDGET,
) -> list[TranscriptChunk]:
    if token_budget < 200:
        raise ValueError("token_budget must be at least 200")
    if not segments:
        raise ValueError("Transcript contains no segments")

    chunks: list[TranscriptChunk] = []
    current_lines: list[str] = []
    current_tokens = 0
    start_segment = 0

    def flush(end_segment: int) -> None:
        nonlocal current_lines, current_tokens, start_segment
        if not current_lines:
            return
        chunks.append(
            TranscriptChunk(
                chunk_index=len(chunks),
                start_segment=start_segment,
                end_segment=end_segment,
                rendered_text="\n".join(current_lines) + "\n",
                estimated_tokens=current_tokens,
            )
        )
        current_lines = []
        current_tokens = 0

    for index, segment in enumerate(segments):
        line = _render_segment(index, segment)
        tokens = _estimate_tokens(line)

        if current_lines and current_tokens + tokens > token_budget:
            flush(index - 1)
            start_segment = index

        if not current_lines:
            start_segment = index

        current_lines.append(line)
        current_tokens += tokens

    flush(len(segments) - 1)
    return chunks



def chunk_output_schema(
    *,
    min_segment: int,
    max_segment: int,
) -> dict[str, Any]:
    segment_index = {
        "type": "integer",
        "minimum": min_segment,
        "maximum": max_segment,
    }

    claim = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {
                "type": "string",
                "minLength": 1,
                "maxLength": 90,
            },
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": 260,
            },
            "explanation": {
                "type": "string",
                "minLength": 1,
                "maxLength": 360,
            },
            "start_segment": segment_index,
            "end_segment": segment_index,
        },
        "required": [
            "topic",
            "text",
            "explanation",
            "start_segment",
            "end_segment",
        ],
    }

    variable = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "symbol": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "meaning": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
            },
            "unit": {
                "type": "string",
                "maxLength": 80,
            },
        },
        "required": ["symbol", "meaning", "unit"],
    }

    formula = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "formula_id": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 120,
            },
            "description": {
                "type": "string",
                "minLength": 1,
                "maxLength": 300,
            },
            "ascii": {
                "type": "string",
                "minLength": 5,
                "maxLength": 300,
            },
            "latex": {
                "type": "string",
                "minLength": 5,
                "maxLength": 400,
            },
            "derivation_type": {
                "type": "string",
                "enum": ["stated", "derived"],
            },
            "variables": {
                "type": "array",
                "items": variable,
                "minItems": 2,
                "maxItems": 10,
            },
            "derivation_steps": {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 240,
                },
                "minItems": 1,
                "maxItems": 6,
            },
            "start_segment": segment_index,
            "end_segment": segment_index,
        },
        "required": [
            "formula_id",
            "name",
            "description",
            "ascii",
            "latex",
            "derivation_type",
            "variables",
            "derivation_steps",
            "start_segment",
            "end_segment",
        ],
    }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claims": {
                "type": "array",
                "items": claim,
                "minItems": 0,
                "maxItems": 5,
            },
            "formulas": {
                "type": "array",
                "items": formula,
                "maxItems": 5,
            },
            "caveats": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 1,
                },
                "maxItems": 0,
            },
        },
        "required": [
            "claims",
            "formulas",
            "caveats",
        ],
    }



def compact_chunk_output_schema(
    *,
    min_segment: int,
    max_segment: int,
) -> dict[str, Any]:
    """Smaller recovery schema for a constrained 4K context."""
    schema = json.loads(
        json.dumps(
            chunk_output_schema(
                min_segment=min_segment,
                max_segment=max_segment,
            )
        )
    )

    claims = schema["properties"]["claims"]
    claims["maxItems"] = 3
    claim = claims["items"]
    claim["properties"]["topic"]["maxLength"] = 60
    claim["properties"]["text"]["maxLength"] = 180
    claim["properties"]["explanation"]["maxLength"] = 220

    formulas = schema["properties"]["formulas"]
    formulas["maxItems"] = 2
    formula = formulas["items"]
    formula["properties"]["name"]["maxLength"] = 90
    formula["properties"]["description"]["maxLength"] = 180
    formula["properties"]["ascii"]["maxLength"] = 180
    formula["properties"]["latex"]["maxLength"] = 240
    formula["properties"]["variables"]["maxItems"] = 6
    formula["properties"]["variables"]["items"]["properties"]["meaning"]["maxLength"] = 120
    formula["properties"]["derivation_steps"]["maxItems"] = 3
    formula["properties"]["derivation_steps"]["items"]["maxLength"] = 160

    caveats = schema["properties"]["caveats"]
    caveats["maxItems"] = 0
    return schema


def safe_chunk_token_budget(
    requested: int,
    num_ctx: int,
) -> int:
    """Reserve room for instructions, schema, and structured output."""
    if requested < 200:
        raise ValueError("requested chunk budget must be at least 200")
    if num_ctx < 2048:
        raise ValueError("num_ctx must be at least 2048")

    context_cap = max(600, min(1600, num_ctx // 4))
    return min(requested, context_cap)



def final_output_schema(
    evidence_ids: Sequence[str],
) -> dict[str, Any]:
    if not evidence_ids:
        raise ValueError("At least one evidence ID is required")

    evidence_id = {
        "type": "string",
        "enum": list(evidence_ids),
    }
    evidence_list = {
        "type": "array",
        "items": evidence_id,
        "minItems": 1,
    }

    cited_text = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
            },
            "evidence_ids": evidence_list,
        },
        "required": ["text", "evidence_ids"],
    }

    section = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "heading": {
                "type": "string",
                "minLength": 1,
                "maxLength": 160,
            },
            "summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": 900,
            },
            "evidence_ids": evidence_list,
        },
        "required": ["heading", "summary", "evidence_ids"],
    }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "executive_summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1800,
            },
            "executive_summary_evidence_ids": evidence_list,
            "key_takeaways": {
                "type": "array",
                "items": cited_text,
                "minItems": 3,
                "maxItems": 10,
            },
            "sections": {
                "type": "array",
                "items": section,
                "minItems": 2,
                "maxItems": 10,
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string", "maxLength": 400},
                "maxItems": 10,
            },
        },
        "required": [
            "executive_summary",
            "executive_summary_evidence_ids",
            "key_takeaways",
            "sections",
            "caveats",
        ],
    }



def compact_final_output_schema(
    evidence_ids: Sequence[str],
) -> dict[str, Any]:
    """Compact, evidence-constrained synthesis schema."""
    if not evidence_ids:
        raise ValueError("At least one evidence ID is required")

    evidence_id = {
        "type": "string",
        "enum": list(evidence_ids),
    }

    takeaway = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {
                "type": "string",
                "minLength": 40,
                "maxLength": 320,
            },
            "evidence_ids": {
                "type": "array",
                "items": evidence_id,
                "minItems": 1,
                "maxItems": 2,
                "uniqueItems": True,
            },
        },
        "required": ["text", "evidence_ids"],
    }

    section = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "heading": {
                "type": "string",
                "minLength": 3,
                "maxLength": 80,
            },
            "summary": {
                "type": "string",
                "minLength": 80,
                "maxLength": 520,
            },
            "evidence_ids": {
                "type": "array",
                "items": evidence_id,
                "minItems": 1,
                "maxItems": 3,
                "uniqueItems": True,
            },
        },
        "required": ["heading", "summary", "evidence_ids"],
    }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "executive_summary": {
                "type": "string",
                "minLength": 180,
                "maxLength": 950,
            },
            "executive_summary_evidence_ids": {
                "type": "array",
                "items": evidence_id,
                "minItems": 2,
                "maxItems": 6,
                "uniqueItems": True,
            },
            "key_takeaways": {
                "type": "array",
                "items": takeaway,
                "minItems": 4,
                "maxItems": 7,
            },
            "sections": {
                "type": "array",
                "items": section,
                "minItems": 3,
                "maxItems": 6,
            },
        },
        "required": [
            "executive_summary",
            "executive_summary_evidence_ids",
            "key_takeaways",
            "sections",
        ],
    }




def _validate_range(
    *,
    start_segment: int,
    end_segment: int,
    minimum: int,
    maximum: int,
    context: str,
) -> None:
    if not (
        minimum <= start_segment <= end_segment <= maximum
    ):
        raise OllamaError(
            f"{context} returned invalid segment range "
            f"{start_segment}-{end_segment}; expected within "
            f"{minimum}-{maximum}"
        )



_BAD_BOILERPLATE = (
    "specified segment range",
    "not seasonally adjusted",
    "not normalized",
    "not adjusted for any changes",
    "based on the company's financial statements",
    "evidence identities",
    "evidence catalog",
    "json schema",
    "machine-readable relationship",
    "formula concept:",
    "not available in the catalog",
    "following concepts are not found",
    "never output",
    "provided transcript",
    "provided data",
    "validation problem",
    "quality retry",
    "new swrite",
)




def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _longest_common_word_run(left: str, right: str) -> int:
    """Return the longest contiguous shared word run."""
    a = _words(left)
    b = _words(right)
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)
    longest = 0
    for word_a in a:
        current = [0]
        for index, word_b in enumerate(b, start=1):
            if word_a == word_b:
                value = previous[index - 1] + 1
            else:
                value = 0
            current.append(value)
            longest = max(longest, value)
        previous = current
    return longest


def _range_text(
    source: TranscriptSourcePackage,
    start_segment: int,
    end_segment: int,
) -> str:
    return " ".join(
        str(source.segments[index]["text"]).strip()
        for index in range(start_segment, end_segment + 1)
    )


_CITATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been",
    "being", "by", "can", "company", "does", "for", "from", "had",
    "has", "have", "how", "if", "in", "into", "is", "it", "its",
    "lesson", "of", "on", "or", "that", "the", "their", "them",
    "then", "this", "to", "video", "was", "were", "what", "when",
    "which", "while", "with", "would",
}


def _support_words(text: str) -> list[str]:
    normalized = text.replace("_", " ").lower()
    return [
        word
        for word in re.findall(r"[a-z0-9]+", normalized)
        if word not in _CITATION_STOPWORDS
        and (len(word) >= 3 or word in {"pe", "eps"})
    ]


def _select_support_window(
    *,
    source: TranscriptSourcePackage,
    start_segment: int,
    end_segment: int,
    support_text: str,
    max_segments: int = MAX_CITATION_SEGMENTS,
) -> tuple[int, int, float]:
    """Choose the narrowest high-overlap support window in a model range."""
    if start_segment > end_segment:
        raise OllamaError(
            f"Cannot localize reversed citation range "
            f"{start_segment}-{end_segment}"
        )

    query_words = _support_words(support_text)
    query_set = set(query_words)
    if not query_set:
        raise OllamaError(
            "Cannot localize citation because the claim has no "
            "content-bearing words"
        )

    best: tuple[float, int, int, int] | None = None
    maximum_length = min(
        max_segments,
        end_segment - start_segment + 1,
    )

    for window_length in range(1, maximum_length + 1):
        last_start = end_segment - window_length + 1
        for candidate_start in range(start_segment, last_start + 1):
            candidate_end = candidate_start + window_length - 1
            window_text = _range_text(
                source,
                candidate_start,
                candidate_end,
            )
            window_words = _support_words(window_text)
            window_set = set(window_words)
            overlap = query_set & window_set

            if not overlap:
                continue

            coverage = len(overlap) / len(query_set)
            overlap_count = len(overlap)

            query_bigrams = {
                (query_words[index], query_words[index + 1])
                for index in range(len(query_words) - 1)
            }
            window_bigrams = {
                (window_words[index], window_words[index + 1])
                for index in range(len(window_words) - 1)
            }
            bigram_matches = len(query_bigrams & window_bigrams)

            # Coverage dominates; matched phrases help; shorter windows win
            # ties so citations remain readable.
            score = (
                coverage * 100.0
                + overlap_count * 8.0
                + bigram_matches * 6.0
                - window_length * 0.75
            )
            candidate = (
                score,
                -window_length,
                -candidate_start,
                candidate_end,
            )
            if best is None or candidate > best:
                best = candidate

    if best is None:
        raise OllamaError(
            "Unable to find transcript support inside the proposed "
            f"range {start_segment}-{end_segment}"
        )

    score, negative_length, negative_start, candidate_end = best
    candidate_start = -negative_start
    window_length = -negative_length
    window_words = set(
        _support_words(
            _range_text(source, candidate_start, candidate_end)
        )
    )
    overlap_count = len(query_set & window_words)
    coverage = overlap_count / len(query_set)

    if overlap_count < 2 and not (
        overlap_count == 1 and len(query_set) <= 2
    ):
        raise OllamaError(
            "Best citation window has insufficient lexical support "
            f"({overlap_count} matched content words)"
        )
    if coverage < 0.12:
        raise OllamaError(
            "Best citation window covers too little of the claim "
            f"({coverage:.1%})"
        )
    if window_length > max_segments:
        raise AssertionError("Citation localization exceeded window limit")

    return candidate_start, candidate_end, score


def _numeric_tokens(text: str) -> set[str]:
    return {
        token.replace(",", "")
        for token in re.findall(
            r"(?<![A-Za-z_])\d[\d,]*(?:\.\d+)?",
            str(text),
        )
    }


def _validate_numeric_grounding(
    text: str,
    support_text: str,
    *,
    context: str,
) -> None:
    claimed = _numeric_tokens(text)
    supported = _numeric_tokens(support_text)
    unsupported = sorted(claimed - supported)
    if unsupported:
        raise OllamaError(
            f"{context} introduces unsupported numeric values "
            f"{unsupported}"
        )


_FORMULA_ALIASES = {
    "earnings_per_share": (
        "earnings per share",
        "eps",
    ),
    "price_to_earnings_ratio": (
        "price to earnings ratio",
        "price earnings ratio",
        "p e ratio",
        "pe ratio",
        "pe",
    ),
    "common_shares_outstanding": (
        "common shares outstanding",
        "shares outstanding",
    ),
    "shares_outstanding": (
        "shares outstanding",
    ),
    "stock_price": (
        "stock price",
        "market price",
        "price",
    ),
    "market_price_per_share": (
        "market price",
        "price per share",
        "price",
    ),
    "book_value_per_share": (
        "book value per share",
        "book value",
    ),
    "total_equity": (
        "total equity",
        "equity",
    ),
    "total_assets": (
        "total assets",
        "assets",
    ),
    "total_liabilities": (
        "total liabilities",
        "liabilities",
    ),
    "total_revenue": (
        "total revenue",
        "revenue",
    ),
    "cost_of_revenue": (
        "cost of revenue",
    ),
    "gross_profit": (
        "gross profit",
    ),
    "net_income": (
        "net income",
    ),
    "margin_of_safety_gap": (
        "margin of safety",
        "margin of safety gap",
    ),
}


_OPERATOR_CUES = {
    "-": (
        " minus ",
        " subtract",
        " subtracted",
        " difference between ",
        " less ",
    ),
    "/": (
        " divide",
        " divided by ",
        " per ",
    ),
    "+": (
        " plus ",
        " add ",
        " added ",
        " sum of ",
    ),
    "*": (
        " multiply",
        " multiplied by ",
        " times ",
        " product of ",
    ),
}


def _formula_parts(
    ascii_formula: str,
) -> tuple[str, str, list[str], str]:
    if ascii_formula.count("=") != 1:
        raise OllamaError(
            "Formula ASCII must contain exactly one '=' relationship"
        )
    left, right = [
        part.strip()
        for part in ascii_formula.split("=", 1)
    ]
    operators = re.findall(r"[+\-*/]", right)
    if len(set(operators)) != 1:
        raise OllamaError(
            "Formula support localization requires one arithmetic "
            "operator type"
        )
    operator = operators[0]
    identifiers = re.findall(
        r"\b[a-z][a-z0-9_]*\b",
        ascii_formula,
    )
    return left, right, list(dict.fromkeys(identifiers)), operator


def _symbol_aliases(
    symbol: str,
    variables: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    aliases = list(_FORMULA_ALIASES.get(symbol, ()))
    phrase = symbol.replace("_", " ").lower().strip()
    if phrase:
        aliases.append(phrase)

    for variable in variables:
        if str(variable.get("symbol") or "").strip() != symbol:
            continue
        meaning = re.sub(
            r"\s+",
            " ",
            str(variable.get("meaning") or "")
            .replace("_", " ")
            .lower(),
        ).strip()
        if meaning:
            aliases.append(meaning)

    unique: list[str] = []
    for alias in aliases:
        normalized = re.sub(r"\s+", " ", alias).strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return tuple(unique)


def _alias_spans(
    normalized_text: str,
    aliases: Sequence[str],
) -> list[tuple[int, int, int]]:
    """Return unique whole-phrase spans, longest aliases first."""
    spans: set[tuple[int, int, int]] = set()

    for alias in sorted(
        aliases,
        key=lambda value: (-len(value.split()), -len(value), value),
    ):
        pattern = re.compile(
            r"(?<![a-z0-9])"
            + re.escape(alias)
            + r"(?![a-z0-9])"
        )
        for match in pattern.finditer(normalized_text):
            spans.add(
                (
                    match.start(),
                    match.end(),
                    len(alias.split()),
                )
            )

    return sorted(
        spans,
        key=lambda value: (-value[2], value[0], value[1]),
    )


def _assign_distinct_symbol_spans(
    candidates: Mapping[str, Sequence[tuple[int, int, int]]],
) -> dict[str, tuple[int, int, int]] | None:
    """Assign one non-overlapping transcript mention per variable."""
    ordered = sorted(
        candidates,
        key=lambda symbol: (
            len(candidates[symbol]),
            -max((span[2] for span in candidates[symbol]), default=0),
            symbol,
        ),
    )

    assigned: dict[str, tuple[int, int, int]] = {}

    def overlaps(
        left: tuple[int, int, int],
        right: tuple[int, int, int],
    ) -> bool:
        return left[0] < right[1] and right[0] < left[1]

    def search(index: int) -> bool:
        if index >= len(ordered):
            return True

        symbol = ordered[index]
        for span in candidates[symbol]:
            if any(overlaps(span, used) for used in assigned.values()):
                continue
            assigned[symbol] = span
            if search(index + 1):
                return True
            assigned.pop(symbol, None)

        return False

    if not search(0):
        return None
    return assigned


def _division_cue_score(
    *,
    normalized_text: str,
    left_symbol: str,
    identifiers: Sequence[str],
) -> float:
    """Require explicit division unless the output is genuinely per-share."""
    padded = f" {normalized_text} "

    explicit_cues = (
        " divide ",
        " divides ",
        " divided by ",
        " quotient ",
        " ratio of ",
    )
    if any(cue in padded for cue in explicit_cues):
        return 90.0

    denominator_symbols = identifiers[1:]
    is_per_share_output = left_symbol.endswith("_per_share")
    has_share_denominator = any(
        "share" in symbol
        for symbol in denominator_symbols
    )
    per_share_cues = (
        " per share ",
        " one share ",
        " each share ",
        " just one share ",
        " break it down per share ",
        " break it down to just one share ",
    )

    if (
        is_per_share_output
        and has_share_denominator
        and any(cue in padded for cue in per_share_cues)
    ):
        return 35.0

    return 0.0


def _window_supports_formula(
    *,
    window_text: str,
    left_symbol: str,
    identifiers: Sequence[str],
    operator: str,
    variables: Sequence[Mapping[str, Any]],
) -> tuple[bool, float, list[str]]:
    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        window_text.lower(),
    ).strip()

    candidates: dict[str, list[tuple[int, int, int]]] = {}
    missing: list[str] = []

    canonical_phrases = {
        symbol: re.sub(
            r"\s+",
            " ",
            symbol.replace("_", " ").lower(),
        ).strip()
        for symbol in identifiers
    }

    for symbol in identifiers:
        own_phrase = canonical_phrases[symbol]
        reserved_by_other_symbols = {
            phrase
            for other_symbol, phrase in canonical_phrases.items()
            if other_symbol != symbol
        }

        aliases = tuple(
            alias
            for alias in _symbol_aliases(symbol, variables)
            if (
                re.sub(r"\s+", " ", alias.lower()).strip()
                == own_phrase
                or re.sub(r"\s+", " ", alias.lower()).strip()
                not in reserved_by_other_symbols
            )
        )

        spans = _alias_spans(normalized, aliases)
        if not spans:
            missing.append(symbol)
        candidates[symbol] = spans

    if missing:
        return False, 0.0, missing

    if operator == "/":
        cue_score = _division_cue_score(
            normalized_text=normalized,
            left_symbol=left_symbol,
            identifiers=identifiers,
        )
        if cue_score <= 0:
            return False, 0.0, []
    else:
        cue_present = any(
            cue.strip() in f" {normalized} "
            for cue in _OPERATOR_CUES[operator]
        )
        if not cue_present:
            return False, 0.0, []
        cue_score = 30.0

    assigned = _assign_distinct_symbol_spans(candidates)
    if assigned is None:
        return False, 0.0, list(identifiers)

    exact_matches = sum(span[2] for span in assigned.values())
    score = (
        len(identifiers) * 100.0
        + exact_matches * 12.0
        + cue_score
    )
    return True, score, []


def _locate_formula_support(
    *,
    source: TranscriptSourcePackage,
    formula: Mapping[str, Any],
    max_segments: int = MAX_CITATION_SEGMENTS,
) -> tuple[int, int, float]:
    ascii_formula = str(formula.get("ascii") or "").strip()
    left_symbol, _, identifiers, operator = _formula_parts(ascii_formula)
    variables = [
        item
        for item in formula.get("variables") or []
        if isinstance(item, Mapping)
    ]

    best: tuple[float, int, int, int] | None = None
    segment_count = len(source.segments)

    for length in range(1, min(max_segments, segment_count) + 1):
        for start in range(0, segment_count - length + 1):
            end = start + length - 1
            window = _range_text(source, start, end)
            supported, score, _ = _window_supports_formula(
                window_text=window,
                left_symbol=left_symbol,
                identifiers=identifiers,
                operator=operator,
                variables=variables,
            )
            if not supported:
                continue

            # Prefer explicit compact support and then earlier ranges.
            candidate = (
                score - length * 1.5,
                -length,
                -start,
                end,
            )
            if best is None or candidate > best:
                best = candidate

    if best is None:
        raise OllamaError(
            "No transcript window contains every formula variable "
            "and the required arithmetic relationship"
        )

    score, negative_length, negative_start, end = best
    start = -negative_start
    return start, end, score


def _repair_chunk_citations(
    payload: Mapping[str, Any],
    *,
    source: TranscriptSourcePackage,
    chunk: TranscriptChunk,
) -> list[str]:
    """Localize claims and globally reground formulas to entailed text."""
    repairs: list[str] = []

    retained_claims: list[dict[str, Any]] = []
    for index, raw_item in enumerate(payload.get("claims") or []):
        context = f"chunk {chunk.chunk_index} claim {index}"
        if not isinstance(raw_item, dict):
            repairs.append(f"{context}: dropped non-object item")
            continue
        try:
            start = int(raw_item["start_segment"])
            end = int(raw_item["end_segment"])
            _validate_range(
                start_segment=start,
                end_segment=end,
                minimum=chunk.start_segment,
                maximum=chunk.end_segment,
                context=context,
            )
            if end - start + 1 > MAX_CITATION_SEGMENTS:
                support_text = " ".join(
                    [
                        str(raw_item.get("topic") or ""),
                        str(raw_item.get("text") or ""),
                        str(raw_item.get("explanation") or ""),
                    ]
                )
                new_start, new_end, score = _select_support_window(
                    source=source,
                    start_segment=start,
                    end_segment=end,
                    support_text=support_text,
                )
                raw_item["start_segment"] = new_start
                raw_item["end_segment"] = new_end
                repairs.append(
                    f"{context}: {start}-{end} -> "
                    f"{new_start}-{new_end} "
                    f"(support score {score:.1f})"
                )
            retained_claims.append(raw_item)
        except (OllamaError, KeyError, TypeError, ValueError) as exc:
            repairs.append(f"{context}: dropped ({exc})")

    retained_formulas: list[dict[str, Any]] = []
    for index, raw_item in enumerate(payload.get("formulas") or []):
        context = f"chunk {chunk.chunk_index} formula {index}"
        if not isinstance(raw_item, dict):
            repairs.append(f"{context}: dropped non-object item")
            continue
        try:
            old_start = int(raw_item["start_segment"])
            old_end = int(raw_item["end_segment"])
            new_start, new_end, score = _locate_formula_support(
                source=source,
                formula=raw_item,
            )
            raw_item["start_segment"] = new_start
            raw_item["end_segment"] = new_end
            if (old_start, old_end) != (new_start, new_end):
                repairs.append(
                    f"{context}: REGROUND {old_start}-{old_end} -> "
                    f"{new_start}-{new_end} "
                    f"(entailment score {score:.1f})"
                )
            retained_formulas.append(raw_item)
        except (OllamaError, KeyError, TypeError, ValueError) as exc:
            repairs.append(f"{context}: dropped ({exc})")

    if isinstance(payload, dict):
        payload["claims"] = retained_claims
        payload["formulas"] = retained_formulas

    return repairs


def _contains_cjk(text: str) -> bool:
    return bool(
        re.search(
            r"[\u3400-\u4dbf\u4e00-\u9fff]",
            str(text),
        )
    )


def _normalize_sentence_surface(text: str) -> str:
    """Normalize sentence mechanics without changing substantive wording."""
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    normalized = normalized.lstrip(" \t-–—•[](){}'\"")
    if not normalized:
        return normalized

    characters = list(normalized)
    for index, character in enumerate(characters):
        if "a" <= character <= "z":
            characters[index] = character.upper()
            break
        if "A" <= character <= "Z":
            break
    normalized = "".join(characters)

    if normalized[-1] not in ".?!":
        normalized = normalized.rstrip(" ,;:") + "."

    return normalized


def _normalize_chunk_surface(
    payload: Mapping[str, Any],
    *,
    chunk_index: int,
) -> list[str]:
    """Normalize useful fields and remove obvious model-language noise."""
    repairs: list[str] = []

    normalized_claims: list[dict[str, Any]] = []
    for index, raw_claim in enumerate(payload.get("claims") or []):
        if not isinstance(raw_claim, dict):
            repairs.append(
                f"chunk {chunk_index} claim {index}: "
                "dropped non-object item"
            )
            continue

        combined = " ".join(
            [
                str(raw_claim.get("topic") or ""),
                str(raw_claim.get("text") or ""),
                str(raw_claim.get("explanation") or ""),
            ]
        )
        if _contains_cjk(combined):
            repairs.append(
                f"chunk {chunk_index} claim {index}: "
                "dropped mixed-language output"
            )
            continue

        for field, suffix in (
            ("text", ""),
            ("explanation", " explanation"),
        ):
            original = str(raw_claim.get(field) or "")
            normalized = _normalize_sentence_surface(original)
            raw_claim[field] = normalized
            if normalized != original:
                repairs.append(
                    f"chunk {chunk_index} claim {index}{suffix}: "
                    "sentence mechanics normalized"
                )

        normalized_claims.append(raw_claim)

    normalized_formulas: list[dict[str, Any]] = []
    for index, raw_formula in enumerate(payload.get("formulas") or []):
        if not isinstance(raw_formula, dict):
            repairs.append(
                f"chunk {chunk_index} formula {index}: "
                "dropped non-object item"
            )
            continue

        structural_text = " ".join(
            [
                str(raw_formula.get("name") or ""),
                str(raw_formula.get("ascii") or ""),
                str(raw_formula.get("latex") or ""),
            ]
        )
        if _contains_cjk(structural_text):
            repairs.append(
                f"chunk {chunk_index} formula {index}: "
                "dropped mixed-language formula"
            )
            continue

        original_description = str(
            raw_formula.get("description") or ""
        )
        if _contains_cjk(original_description):
            original_description = re.sub(
                r"[\u3400-\u4dbf\u4e00-\u9fff]+",
                " ",
                original_description,
            )
            repairs.append(
                f"chunk {chunk_index} formula {index} description: "
                "removed foreign-language artifact"
            )
        raw_formula["description"] = _normalize_sentence_surface(
            original_description
        )

        normalized_steps: list[str] = []
        for step_index, step in enumerate(
            raw_formula.get("derivation_steps") or []
        ):
            original = str(step)
            if _contains_cjk(original):
                original = re.sub(
                    r"[\u3400-\u4dbf\u4e00-\u9fff]+",
                    " ",
                    original,
                )
                repairs.append(
                    f"chunk {chunk_index} formula {index} "
                    f"derivation step {step_index}: "
                    "removed foreign-language artifact"
                )
            normalized = _normalize_sentence_surface(original)
            if normalized:
                normalized_steps.append(normalized)
            if normalized != str(step):
                repairs.append(
                    f"chunk {chunk_index} formula {index} "
                    f"derivation step {step_index}: "
                    "sentence mechanics normalized"
                )
        raw_formula["derivation_steps"] = normalized_steps
        normalized_formulas.append(raw_formula)

    normalized_caveats: list[str] = []
    for index, caveat in enumerate(payload.get("caveats") or []):
        original = str(caveat)
        lowered = original.lower()

        if (
            _contains_cjk(original)
            or any(phrase in lowered for phrase in _BAD_BOILERPLATE)
        ):
            repairs.append(
                f"chunk {chunk_index} caveat {index}: "
                "dropped low-quality caveat"
            )
            continue

        normalized = _normalize_sentence_surface(original)
        alpha_words = re.findall(r"[A-Za-z]{2,}", normalized)
        if len(alpha_words) < 4:
            repairs.append(
                f"chunk {chunk_index} caveat {index}: "
                "dropped non-substantive caveat"
            )
            continue

        normalized_caveats.append(normalized)
        if normalized != original:
            repairs.append(
                f"chunk {chunk_index} caveat {index}: "
                "sentence mechanics normalized"
            )

    if isinstance(payload, dict):
        payload["claims"] = normalized_claims
        payload["formulas"] = normalized_formulas
        if payload.get("caveats"):
            repairs.append(
                f"chunk {chunk_index}: discarded model-generated caveats"
            )
        payload["caveats"] = []

    return repairs


def _validate_plain_english(
    text: str,
    *,
    context: str,
    source_text: str | None = None,
    require_sentence: bool = True,
) -> None:
    stripped = text.strip()
    if not stripped:
        raise OllamaError(f"{context} is empty")
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", stripped):
        raise OllamaError(f"{context} contains CJK characters")
    if re.search(r"\b[CE][0-9]{1,4}\b", stripped):
        raise OllamaError(
            f"{context} contains an evidence or citation ID in prose"
        )
    lowered = stripped.lower()
    for phrase in _BAD_BOILERPLATE:
        if phrase in lowered:
            raise OllamaError(
                f"{context} contains unsupported boilerplate: {phrase!r}"
            )
    if require_sentence:
        if not re.match(r"^[A-Z0-9]", stripped):
            raise OllamaError(
                f"{context} must begin as a complete sentence"
            )
        if stripped[-1] not in ".?!":
            raise OllamaError(
                f"{context} must end as a complete sentence"
            )
    if source_text is not None:
        copied = _longest_common_word_run(stripped, source_text)
        if copied >= MAX_COPY_RUN_WORDS:
            raise OllamaError(
                f"{context} copies {copied} consecutive transcript words"
            )


def _variable_meaning(symbol: str) -> str:
    phrase = symbol.replace("_", " ").strip()
    if not phrase:
        return "Quantity used in the relationship"
    return phrase[0].upper() + phrase[1:]


def _variable_unit(symbol: str) -> str:
    lowered = symbol.lower()
    if "ratio" in lowered or lowered in {"pe", "p_e"}:
        return "ratio"
    if (
        "per_share" in lowered
        or lowered.endswith("_per_share")
        or "price" in lowered
    ):
        return "currency per share"
    if "shares" in lowered:
        return "shares"
    if any(
        word in lowered
        for word in (
            "assets",
            "liabilities",
            "equity",
            "revenue",
            "income",
            "profit",
            "cost",
            "price",
            "value",
            "earnings",
        )
    ):
        return "currency"
    return ""



def _normalize_formula_semantics(
    payload: Mapping[str, Any],
    *,
    chunk_index: int,
) -> list[str]:
    """Normalize transcript-specific labels without changing the equation."""
    repairs: list[str] = []

    for index, formula in enumerate(payload.get("formulas") or []):
        if not isinstance(formula, dict):
            continue

        ascii_formula = re.sub(
            r"\s+",
            " ",
            str(formula.get("ascii") or "").strip(),
        )
        if ascii_formula.count("=") != 1:
            continue

        left, right = [
            part.strip()
            for part in ascii_formula.split("=", 1)
        ]
        right_identifiers = set(
            re.findall(r"\b[a-z][a-z0-9_]*\b", right)
        )

        is_absolute_price_book_gap = (
            left == "margin_of_safety"
            and "-" in right
            and any(
                symbol in right_identifiers
                for symbol in (
                    "market_price",
                    "stock_price",
                    "price",
                )
            )
            and any(
                symbol in right_identifiers
                for symbol in (
                    "book_value_per_share",
                    "book_value",
                )
            )
        )

        if not is_absolute_price_book_gap:
            continue

        normalized_ascii = (
            "margin_of_safety_gap = " + right
        )
        formula["ascii"] = normalized_ascii
        formula["formula_id"] = "margin_of_safety_gap"
        formula["name"] = "Margin of safety gap"
        formula["description"] = (
            "The absolute price-to-book gap described by the speaker."
        )

        latex = str(formula.get("latex") or "")
        if "=" in latex:
            _, latex_right = latex.split("=", 1)
            formula["latex"] = (
                r"\text{Margin of Safety Gap}=" + latex_right
            )
        else:
            formula["latex"] = (
                r"\text{Margin of Safety Gap}="
                + right.replace("_", r"\_")
            )

        normalized_variables: list[dict[str, Any]] = []
        for variable in formula.get("variables") or []:
            if not isinstance(variable, Mapping):
                continue
            item = dict(variable)
            if str(item.get("symbol") or "") == "margin_of_safety":
                item["symbol"] = "margin_of_safety_gap"
                item["meaning"] = (
                    "Absolute difference between market price and "
                    "book value per share"
                )
                if not str(item.get("unit") or "").strip():
                    item["unit"] = "currency per share"
            normalized_variables.append(item)
        formula["variables"] = normalized_variables

        repairs.append(
            f"chunk {chunk_index} formula {index}: "
            "renamed margin_of_safety to margin_of_safety_gap"
        )

    return repairs


def _repair_formula_variables(
    payload: Mapping[str, Any],
    *,
    chunk_index: int,
) -> list[str]:
    """Complete missing variable metadata from the symbolic equation."""
    repairs: list[str] = []

    for index, formula in enumerate(payload.get("formulas") or []):
        if not isinstance(formula, dict):
            continue

        ascii_formula = str(formula.get("ascii") or "")
        identifiers = set(
            re.findall(r"\b[a-z][a-z0-9_]*\b", ascii_formula)
        )
        if not identifiers:
            continue

        existing: dict[str, dict[str, Any]] = {}
        for variable in formula.get("variables") or []:
            if not isinstance(variable, Mapping):
                continue
            symbol = str(variable.get("symbol") or "").strip()
            if symbol in identifiers and symbol not in existing:
                raw_meaning = str(
                    variable.get("meaning") or ""
                ).strip()
                normalized_meaning = re.sub(
                    r"\s+",
                    " ",
                    raw_meaning.replace("_", " "),
                ).strip()
                symbol_phrase = symbol.replace("_", " ")
                if (
                    not normalized_meaning
                    or normalized_meaning.lower()
                    == symbol_phrase.lower()
                ):
                    normalized_meaning = _variable_meaning(symbol)
                    repairs.append(
                        f"chunk {chunk_index} formula {index}: "
                        f"normalized meaning for {symbol}"
                    )

                existing[symbol] = {
                    "symbol": symbol,
                    "meaning": normalized_meaning,
                    "unit": str(
                        variable.get("unit")
                        or _variable_unit(symbol)
                    ).strip(),
                }

        for symbol in sorted(identifiers):
            if symbol not in existing:
                existing[symbol] = {
                    "symbol": symbol,
                    "meaning": _variable_meaning(symbol),
                    "unit": _variable_unit(symbol),
                }
                repairs.append(
                    f"chunk {chunk_index} formula {index}: "
                    f"added definition for {symbol}"
                )

        original_symbols = {
            str(variable.get("symbol") or "").strip()
            for variable in formula.get("variables") or []
            if isinstance(variable, Mapping)
        }
        removed = sorted(original_symbols - identifiers - {""})
        if removed:
            repairs.append(
                f"chunk {chunk_index} formula {index}: "
                f"removed unused definitions {removed}"
            )

        formula["variables"] = [
            existing[symbol] for symbol in sorted(identifiers)
        ]

    return repairs


def _prune_chunk_payload(
    payload: Mapping[str, Any],
    *,
    source: TranscriptSourcePackage,
    chunk: TranscriptChunk,
) -> list[str]:
    """Drop isolated bad items while preserving valid grounded content."""
    repairs: list[str] = []

    valid_claims: list[dict[str, Any]] = []
    for index, claim in enumerate(payload.get("claims") or []):
        try:
            start = int(claim["start_segment"])
            end = int(claim["end_segment"])
            _validate_range(
                start_segment=start,
                end_segment=end,
                minimum=chunk.start_segment,
                maximum=chunk.end_segment,
                context=f"chunk {chunk.chunk_index} claim {index}",
            )
            if end - start + 1 > MAX_CITATION_SEGMENTS:
                raise OllamaError(
                    "citation exceeds the six-segment limit"
                )
            cited_text = _range_text(source, start, end)
            _validate_plain_english(
                str(claim["text"]),
                context=f"chunk {chunk.chunk_index} claim {index}",
                source_text=cited_text,
                require_sentence=True,
            )
            _validate_plain_english(
                str(claim["explanation"]),
                context=(
                    f"chunk {chunk.chunk_index} claim "
                    f"{index} explanation"
                ),
                source_text=cited_text,
                require_sentence=True,
            )
            _validate_numeric_grounding(
                " ".join(
                    [
                        str(claim["text"]),
                        str(claim["explanation"]),
                    ]
                ),
                cited_text,
                context=f"chunk {chunk.chunk_index} claim {index}",
            )
            valid_claims.append(claim)
        except (OllamaError, KeyError, TypeError, ValueError) as exc:
            repairs.append(
                f"chunk {chunk.chunk_index} claim {index}: "
                f"dropped ({exc})"
            )

    valid_formulas: list[dict[str, Any]] = []
    for index, formula in enumerate(payload.get("formulas") or []):
        try:
            _validate_formula_candidate(
                formula,
                source=source,
                context=f"chunk {chunk.chunk_index} formula {index}",
                minimum_segment=0,
                maximum_segment=len(source.segments) - 1,
            )
            valid_formulas.append(formula)
        except (OllamaError, KeyError, TypeError, ValueError) as exc:
            repairs.append(
                f"chunk {chunk.chunk_index} formula {index}: "
                f"dropped ({exc})"
            )

    valid_caveats: list[str] = []
    for index, caveat in enumerate(payload.get("caveats") or []):
        try:
            _validate_plain_english(
                str(caveat),
                context=f"chunk {chunk.chunk_index} caveat {index}",
                require_sentence=True,
            )
            valid_caveats.append(str(caveat))
        except OllamaError as exc:
            repairs.append(
                f"chunk {chunk.chunk_index} caveat {index}: "
                f"dropped ({exc})"
            )

    if isinstance(payload, dict):
        payload["claims"] = valid_claims
        payload["formulas"] = valid_formulas
        payload["caveats"] = valid_caveats

    return repairs


def _validate_formula_candidate(
    formula: Mapping[str, Any],
    *,
    source: TranscriptSourcePackage,
    context: str,
    minimum_segment: int,
    maximum_segment: int,
) -> None:
    start = int(formula["start_segment"])
    end = int(formula["end_segment"])
    _validate_range(
        start_segment=start,
        end_segment=end,
        minimum=minimum_segment,
        maximum=maximum_segment,
        context=context,
    )
    if end - start + 1 > MAX_CITATION_SEGMENTS:
        raise OllamaError(
            f"{context} citation spans more than "
            f"{MAX_CITATION_SEGMENTS} segments"
        )

    ascii_formula = str(formula.get("ascii") or "").strip()
    latex_formula = str(formula.get("latex") or "").strip()
    description = str(formula.get("description") or "").strip()

    if ascii_formula.count("=") != 1:
        raise OllamaError(
            f"{context} ASCII must contain exactly one '=' relationship"
        )
    if "=" not in latex_formula:
        raise OllamaError(
            f"{context} LaTeX must contain an '=' relationship"
        )

    left, right = [part.strip() for part in ascii_formula.split("=", 1)]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", left):
        raise OllamaError(
            f"{context} left side must be one snake_case variable"
        )
    if not re.search(r"[+\-*/]", right):
        raise OllamaError(
            f"{context} right side must contain an explicit operator"
        )

    identifiers = set(re.findall(r"\b[a-z][a-z0-9_]*\b", ascii_formula))
    numeric_only = not identifiers
    if numeric_only or len(identifiers) < 3:
        raise OllamaError(
            f"{context} must relate at least three named quantities"
        )

    variables = formula.get("variables") or []
    symbols = {
        str(item.get("symbol") or "").strip()
        for item in variables
        if isinstance(item, Mapping)
    }
    if "" in symbols:
        symbols.remove("")
    missing_definitions = identifiers - symbols
    extra_definitions = symbols - identifiers
    if missing_definitions:
        raise OllamaError(
            f"{context} lacks variable definitions for "
            f"{sorted(missing_definitions)}"
        )
    if extra_definitions:
        raise OllamaError(
            f"{context} defines unused variables "
            f"{sorted(extra_definitions)}"
        )

    _validate_plain_english(
        description,
        context=f"{context} description",
        require_sentence=True,
    )

    steps = [
        str(step).strip()
        for step in formula.get("derivation_steps") or []
    ]
    if not steps:
        raise OllamaError(
            f"{context} must include at least one derivation step"
        )
    for index, step in enumerate(steps):
        _validate_plain_english(
            step,
            context=f"{context} derivation step {index}",
            require_sentence=True,
        )

    source_text = _range_text(source, start, end)
    left_symbol, _, ordered_identifiers, operator = _formula_parts(
        ascii_formula
    )
    supported, _, missing = _window_supports_formula(
        window_text=source_text,
        left_symbol=left_symbol,
        identifiers=ordered_identifiers,
        operator=operator,
        variables=[
            item
            for item in variables
            if isinstance(item, Mapping)
        ],
    )
    if not supported:
        raise OllamaError(
            f"{context} is not entailed by its transcript range; "
            f"missing variables={missing}"
        )


def _validate_chunk_payload(
    payload: Mapping[str, Any],
    *,
    source: TranscriptSourcePackage,
    chunk: TranscriptChunk,
) -> None:
    expected = {
        "claims",
        "formulas",
        "caveats",
    }
    if set(payload) != expected:
        raise OllamaError(
            f"chunk {chunk.chunk_index} returned unexpected keys"
        )

    for index, claim in enumerate(payload["claims"]):
        start = int(claim["start_segment"])
        end = int(claim["end_segment"])
        _validate_range(
            start_segment=start,
            end_segment=end,
            minimum=chunk.start_segment,
            maximum=chunk.end_segment,
            context=f"chunk {chunk.chunk_index} claim {index}",
        )
        if end - start + 1 > MAX_CITATION_SEGMENTS:
            raise OllamaError(
                f"chunk {chunk.chunk_index} claim {index} citation "
                f"spans more than {MAX_CITATION_SEGMENTS} segments"
            )
        cited_text = _range_text(source, start, end)
        _validate_plain_english(
            str(claim["text"]),
            context=f"chunk {chunk.chunk_index} claim {index}",
            source_text=cited_text,
            require_sentence=True,
        )
        _validate_plain_english(
            str(claim["explanation"]),
            context=(
                f"chunk {chunk.chunk_index} claim {index} explanation"
            ),
            source_text=cited_text,
            require_sentence=True,
        )
        _validate_numeric_grounding(
            " ".join(
                [
                    str(claim["text"]),
                    str(claim["explanation"]),
                ]
            ),
            cited_text,
            context=f"chunk {chunk.chunk_index} claim {index}",
        )

    for index, formula in enumerate(payload["formulas"]):
        _validate_formula_candidate(
            formula,
            source=source,
            context=f"chunk {chunk.chunk_index} formula {index}",
            minimum_segment=0,
            maximum_segment=len(source.segments) - 1,
        )

    for index, caveat in enumerate(payload["caveats"]):
        _validate_plain_english(
            str(caveat),
            context=f"chunk {chunk.chunk_index} caveat {index}",
            require_sentence=True,
        )


def _text_similarity(left: str, right: str) -> float:
    a = set(_support_words(left))
    b = set(_support_words(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _validate_output_against_evidence(
    text: str,
    ids: Sequence[str],
    *,
    context: str,
    source: TranscriptSourcePackage,
    by_id: Mapping[str, Evidence],
    max_ids: int,
) -> None:
    normalized_ids = [str(value) for value in ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise OllamaError(f"{context} repeats evidence IDs")
    if not 1 <= len(normalized_ids) <= max_ids:
        raise OllamaError(
            f"{context} must use 1-{max_ids} relevant evidence IDs"
        )

    support = " ".join(
        part
        for value in normalized_ids
        for part in (
            by_id[value].text,
            _range_text(
                source,
                by_id[value].start_segment,
                by_id[value].end_segment,
            ),
        )
    )

    _validate_numeric_grounding(
        text,
        support,
        context=context,
    )

    output_words = set(_support_words(text))
    support_words = set(_support_words(support))
    overlap = output_words & support_words
    if len(overlap) < 2:
        raise OllamaError(
            f"{context} is not meaningfully connected to its evidence"
        )


def _validate_narrative_quality(
    payload: Mapping[str, Any],
    *,
    source: TranscriptSourcePackage,
    evidence: Sequence[Evidence],
) -> None:
    by_id = {item.evidence_id: item for item in evidence}
    forbidden = (
        "evidence",
        "catalog",
        "schema",
        "machine-readable",
        "machine_readable",
        "formula concept",
        "validation",
        "unavailable",
        "provided transcript",
        "provided data",
        "cannot be determined accurately",
    )

    def clean_text(value: str, context: str) -> str:
        text_value = str(value).strip()
        _validate_plain_english(
            text_value,
            context=context,
            source_text=source.transcript_text,
            require_sentence=True,
        )
        lowered = text_value.lower()
        for phrase in forbidden:
            if phrase in lowered:
                raise OllamaError(
                    f"{context} contains internal/debug wording "
                    f"{phrase!r}"
                )
        if "_" in text_value:
            raise OllamaError(
                f"{context} contains machine-style variable names"
            )
        return text_value

    summary = clean_text(
        str(payload["executive_summary"]),
        "executive_summary",
    )
    sentence_count = len(
        [
            part
            for part in re.split(r"(?<=[.!?])\s+", summary)
            if part
        ]
    )
    if not 3 <= sentence_count <= 5:
        raise OllamaError(
            "executive_summary must contain 3-5 complete sentences"
        )

    _validate_output_against_evidence(
        summary,
        payload["executive_summary_evidence_ids"],
        context="executive_summary",
        source=source,
        by_id=by_id,
        max_ids=6,
    )

    takeaway_texts: list[str] = []
    takeaway_id_sets: list[tuple[str, ...]] = []
    for index, item in enumerate(payload["key_takeaways"]):
        value = clean_text(
            str(item["text"]),
            f"key_takeaways[{index}]",
        )
        _validate_output_against_evidence(
            value,
            item["evidence_ids"],
            context=f"key_takeaways[{index}]",
            source=source,
            by_id=by_id,
            max_ids=2,
        )
        for prior in takeaway_texts:
            if _text_similarity(value, prior) >= 0.68:
                raise OllamaError(
                    f"key_takeaways[{index}] duplicates another takeaway"
                )
        ids_tuple = tuple(sorted(str(v) for v in item["evidence_ids"]))
        if ids_tuple in takeaway_id_sets:
            raise OllamaError(
                f"key_takeaways[{index}] reuses an earlier citation set"
            )
        takeaway_texts.append(value)
        takeaway_id_sets.append(ids_tuple)

    previous_start = -1
    for index, item in enumerate(payload["sections"]):
        heading = str(item["heading"]).strip()
        _validate_plain_english(
            heading,
            context=f"sections[{index}] heading",
            require_sentence=False,
        )
        if any(phrase in heading.lower() for phrase in forbidden):
            raise OllamaError(
                f"sections[{index}] heading contains internal wording"
            )
        if "_" in heading or len(heading.split()) > 10:
            raise OllamaError(
                f"sections[{index}] heading is not reader-facing"
            )

        summary_text = clean_text(
            str(item["summary"]),
            f"sections[{index}] summary",
        )
        _validate_output_against_evidence(
            summary_text,
            item["evidence_ids"],
            context=f"sections[{index}]",
            source=source,
            by_id=by_id,
            max_ids=3,
        )

        ids = [str(value) for value in item["evidence_ids"]]
        section_start = min(
            by_id[value].start_segment
            for value in ids
        )
        if section_start < previous_start:
            raise OllamaError(
                "sections are not ordered by transcript progression"
            )
        previous_start = section_start

    used_ids = {
        str(value)
        for value in payload["executive_summary_evidence_ids"]
    }
    for item in payload["key_takeaways"]:
        used_ids.update(str(value) for value in item["evidence_ids"])
    for item in payload["sections"]:
        used_ids.update(str(value) for value in item["evidence_ids"])

    represented_segments = {
        by_id[value].start_segment
        for value in used_ids
        if value in by_id
    }
    if len(source.segments) >= 50 and len(represented_segments) < 4:
        raise OllamaError(
            "final synthesis does not cover enough transcript areas"
        )


def _chunk_from_range(
    source: TranscriptSourcePackage,
    *,
    chunk_index: int,
    start_segment: int,
    end_segment: int,
) -> TranscriptChunk:
    lines = [
        _render_segment(index, source.segments[index])
        for index in range(start_segment, end_segment + 1)
    ]
    rendered = "\n".join(lines) + "\n"
    return TranscriptChunk(
        chunk_index=chunk_index,
        start_segment=start_segment,
        end_segment=end_segment,
        rendered_text=rendered,
        estimated_tokens=_estimate_tokens(rendered),
    )


def _aggregate_ollama_responses(
    responses: Sequence[Mapping[str, Any]],
    *,
    split_count: int = 0,
) -> dict[str, Any]:
    numeric_fields = (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )
    aggregate: dict[str, Any] = {
        "done": True,
        "done_reason": "stop",
        "request_count": 0,
        "split_count": split_count,
    }

    for response in responses:
        aggregate["request_count"] += int(
            response.get("request_count", 1)
        )
        aggregate["split_count"] += int(
            response.get("split_count", 0)
        )
        for field in numeric_fields:
            value = response.get(field)
            if isinstance(value, (int, float)):
                aggregate[field] = aggregate.get(field, 0) + value

    return aggregate


def _merge_chunk_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    formulas: list[dict[str, Any]] = []
    caveats: list[str] = []
    claim_keys: set[tuple[str, int, int]] = set()
    formula_keys: set[tuple[str, int, int]] = set()

    for payload in payloads:
        for claim in payload.get("claims") or []:
            key = (
                str(claim.get("text") or "").strip().lower(),
                int(claim["start_segment"]),
                int(claim["end_segment"]),
            )
            if key not in claim_keys:
                claim_keys.add(key)
                claims.append(dict(claim))

        for formula in payload.get("formulas") or []:
            key = (
                str(formula.get("ascii") or "").replace(" ", "").lower(),
                int(formula["start_segment"]),
                int(formula["end_segment"]),
            )
            if key not in formula_keys:
                formula_keys.add(key)
                formulas.append(dict(formula))

        for caveat in payload.get("caveats") or []:
            normalized = str(caveat).strip()
            if normalized and normalized not in caveats:
                caveats.append(normalized)

    return {
        "claims": claims,
        "formulas": formulas,
        "caveats": caveats,
    }


def _build_chunk_prompt(
    *,
    source: TranscriptSourcePackage,
    chunk: TranscriptChunk,
    schema: Mapping[str, Any],
    compact: bool,
    prior_error: Exception | None = None,
) -> str:
    title = str(source.metadata.get("title") or source.video_id)

    if compact:
        task = (
            "Return a compact replacement extraction with zero to three "
            "claims and at most two formulas. Keep each claim and "
            "explanation to one concise sentence. Return empty claims and "
            "formulas when the passage has no standalone evidence. Return "
            "no caveats. Do not mention the earlier response or validation."
        )
    else:
        task = (
            "Extract zero to five durable concepts and any reusable "
            "symbolic relationships actually expressed in this chunk. "
            "Paraphrase rather than copying. Numeric examples belong in "
            "claims, not the formulas array. If this passage is only a "
            "transition, continuation, navigation instruction, or contains "
            "no standalone concept, return empty claims and formulas. "
            "Return no caveats."
        )

    return (
        f"Video title: {title}\n"
        f"Chunk: {chunk.chunk_index + 1}\n"
        f"Segment range: {chunk.start_segment}-{chunk.end_segment}\n\n"
        f"{task}\n\n"
        f"Transcript:\n{chunk.rendered_text}"
    )


def extract_chunk_evidence(
    *,
    client: OllamaClient,
    model: str,
    source: TranscriptSourcePackage,
    chunk: TranscriptChunk,
    num_ctx: int,
    think: bool = DEFAULT_EXTRACTION_THINK,
    _split_depth: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    errors: list[Exception] = []

    for compact in (False, True):
        schema = (
            compact_chunk_output_schema(
                min_segment=chunk.start_segment,
                max_segment=chunk.end_segment,
            )
            if compact
            else chunk_output_schema(
                min_segment=chunk.start_segment,
                max_segment=chunk.end_segment,
            )
        )
        prompt = _build_chunk_prompt(
            source=source,
            chunk=chunk,
            schema=schema,
            compact=compact,
            prior_error=errors[-1] if errors else None,
        )
        response = client.chat(
            model=model,
            system=CHUNK_SYSTEM_PROMPT,
            user=prompt,
            schema=schema,
            num_ctx=num_ctx,
            num_predict=1800 if compact else 2400,
            think=think,
        )
        responses.append(response)

        try:
            payload = _json_from_ollama_response(
                response,
                context=(
                    f"chunk {chunk.chunk_index} "
                    f"{'compact recovery' if compact else 'initial'}"
                ),
            )
            citation_repairs = _repair_chunk_citations(
                payload,
                source=source,
                chunk=chunk,
            )
            surface_repairs = _normalize_chunk_surface(
                payload,
                chunk_index=chunk.chunk_index,
            )
            semantic_repairs = _normalize_formula_semantics(
                payload,
                chunk_index=chunk.chunk_index,
            )
            variable_repairs = _repair_formula_variables(
                payload,
                chunk_index=chunk.chunk_index,
            )
            prune_repairs = _prune_chunk_payload(
                payload,
                source=source,
                chunk=chunk,
            )
            _validate_chunk_payload(
                payload,
                source=source,
                chunk=chunk,
            )
            for repair in [
                *citation_repairs,
                *surface_repairs,
                *semantic_repairs,
                *variable_repairs,
                *prune_repairs,
            ]:
                print(f"REPAIR: {repair}")

            has_evidence = bool(
                payload["claims"] or payload["formulas"]
            )
            if not has_evidence and not compact:
                errors.append(
                    OllamaError(
                        f"chunk {chunk.chunk_index} produced no "
                        "standalone evidence; trying compact recovery"
                    )
                )
                continue

            if not has_evidence:
                print(
                    f"SKIP: chunk {chunk.chunk_index} segments "
                    f"{chunk.start_segment}-{chunk.end_segment} "
                    "contain no standalone evidence"
                )

            return payload, _aggregate_ollama_responses(responses)
        except (OllamaError, KeyError, TypeError, ValueError) as exc:
            errors.append(exc)

    segment_count = chunk.end_segment - chunk.start_segment + 1
    if (
        segment_count >= MIN_SPLITTABLE_SEGMENTS
        and _split_depth < MAX_CHUNK_SPLIT_DEPTH
    ):
        midpoint = (chunk.start_segment + chunk.end_segment) // 2
        left = _chunk_from_range(
            source,
            chunk_index=chunk.chunk_index,
            start_segment=chunk.start_segment,
            end_segment=midpoint,
        )
        right = _chunk_from_range(
            source,
            chunk_index=chunk.chunk_index,
            start_segment=midpoint + 1,
            end_segment=chunk.end_segment,
        )
        print(
            f"SPLIT: chunk {chunk.chunk_index} segments "
            f"{chunk.start_segment}-{chunk.end_segment} after "
            f"{errors[-1]}; retrying {left.start_segment}-"
            f"{left.end_segment} and {right.start_segment}-"
            f"{right.end_segment}"
        )

        left_payload, left_response = extract_chunk_evidence(
            client=client,
            model=model,
            source=source,
            chunk=left,
            num_ctx=num_ctx,
            think=think,
            _split_depth=_split_depth + 1,
        )
        right_payload, right_response = extract_chunk_evidence(
            client=client,
            model=model,
            source=source,
            chunk=right,
            num_ctx=num_ctx,
            think=think,
            _split_depth=_split_depth + 1,
        )
        merged = _merge_chunk_payloads(
            [left_payload, right_payload]
        )
        return merged, _aggregate_ollama_responses(
            [*responses, left_response, right_response],
            split_count=1,
        )

    joined_errors = "; ".join(str(error) for error in errors)
    raise OllamaError(
        f"chunk {chunk.chunk_index} failed compact recovery and "
        f"cannot be split further: {joined_errors}"
    )




def build_evidence_catalog(
    chunk_payloads: Sequence[Mapping[str, Any]],
) -> tuple[list[Evidence], list[dict[str, Any]], list[str]]:
    evidence: list[Evidence] = []
    formulas: list[dict[str, Any]] = []
    next_evidence = 1

    for payload in chunk_payloads:
        for claim in payload["claims"]:
            topic = str(claim["topic"]).strip()
            statement = str(claim["text"]).strip()
            explanation = str(claim["explanation"]).strip()
            evidence.append(
                Evidence(
                    evidence_id=f"E{next_evidence:04d}",
                    text=(
                        f"{topic}. {statement} {explanation}"
                    ),
                    start_segment=int(claim["start_segment"]),
                    end_segment=int(claim["end_segment"]),
                )
            )
            next_evidence += 1

        for formula in payload["formulas"]:
            normalized = dict(formula)
            formula_evidence = Evidence(
                evidence_id=f"E{next_evidence:04d}",
                text=(
                    f"{formula['name']}. "
                    f"{formula['description']}"
                ),
                start_segment=int(formula["start_segment"]),
                end_segment=int(formula["end_segment"]),
            )
            normalized["_evidence"] = formula_evidence
            evidence.append(formula_evidence)
            next_evidence += 1
            formulas.append(normalized)

    if not evidence:
        raise OllamaError("No grounded claims or formulas were extracted")

    return evidence, formulas, []


def _render_evidence_catalog(evidence: Sequence[Evidence]) -> str:
    return "\n".join(
        (
            f"[{item.evidence_id} | "
            f"S{item.start_segment:04d}-S{item.end_segment:04d}] "
            f"{item.text}"
        )
        for item in evidence
    )



def synthesize_narrative(
    *,
    client: OllamaClient,
    model: str,
    source: TranscriptSourcePackage,
    evidence: Sequence[Evidence],
    num_ctx: int,
    think: bool = DEFAULT_NARRATIVE_THINK,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_ids = [item.evidence_id for item in evidence]
    schema = compact_final_output_schema(evidence_ids)
    title = str(source.metadata.get("title") or source.video_id)

    compact_evidence = "\n".join(
        (
            f"[{item.evidence_id}|S{item.start_segment:04d}-"
            f"S{item.end_segment:04d}] NOTE: {item.text[:260]} "
            f"SOURCE: {_range_text(source, item.start_segment, item.end_segment)[:300]}"
        )
        for item in evidence
    )

    base_prompt = (
        f"Video title: {title}\n\n"
        "Write a reader-ready brief covering the lesson from beginning "
        "to end. Use four to seven distinct takeaways and three to six "
        "ordered sections. Write the executive summary as exactly four "
        "complete sentences: purpose, progression, key relationships, and "
        "practical conclusion. Paraphrase and connect ideas; do not copy notes "
        "or source excerpts. Do not mention the evidence system. Use only "
        "numbers and dates present in the evidence attached to that item. "
        "Choose only the most relevant evidence IDs for each item.\n\n"
        f"Grounded evidence notes:\n{compact_evidence}\n"
    )

    last_error: Exception | None = None

    for attempt in range(3):
        prompt = base_prompt
        if attempt:
            prompt += (
                "\nRewrite the entire response from scratch. The previous "
                "response failed quality checks. Do not mention those checks, "
                "the prior response, schemas, evidence catalogs, or missing "
                "information."
            )

        response = client.chat(
            model=model,
            system=SYNTHESIS_SYSTEM_PROMPT,
            user=prompt,
            schema=schema,
            num_ctx=num_ctx,
            num_predict=1900,
            think=think,
        )

        try:
            payload = _json_from_ollama_response(
                response,
                context=f"final synthesis attempt {attempt + 1}",
            )

            expected = {
                "executive_summary",
                "executive_summary_evidence_ids",
                "key_takeaways",
                "sections",
            }
            if set(payload) != expected:
                raise OllamaError(
                    "final synthesis returned unexpected keys"
                )

            valid_ids = set(evidence_ids)

            def validate_ids(
                values: Iterable[str],
                context: str,
            ) -> list[str]:
                normalized = [str(value) for value in values]
                unknown = [
                    value
                    for value in normalized
                    if value not in valid_ids
                ]
                if unknown:
                    raise OllamaError(
                        f"{context} referenced unknown evidence IDs: "
                        f"{unknown}"
                    )
                if not normalized:
                    raise OllamaError(
                        f"{context} requires evidence IDs"
                    )
                return normalized

            payload["executive_summary_evidence_ids"] = validate_ids(
                payload["executive_summary_evidence_ids"],
                "executive_summary",
            )
            for index, item in enumerate(payload["key_takeaways"]):
                item["evidence_ids"] = validate_ids(
                    item["evidence_ids"],
                    f"key_takeaways[{index}]",
                )
            for index, item in enumerate(payload["sections"]):
                item["evidence_ids"] = validate_ids(
                    item["evidence_ids"],
                    f"sections[{index}]",
                )

            payload["caveats"] = []
            _validate_narrative_quality(
                payload,
                source=source,
                evidence=evidence,
            )
            return payload, response

        except (OllamaError, KeyError, TypeError, ValueError) as exc:
            last_error = exc

    raise OllamaError(
        "Final synthesis failed quality validation three times: "
        f"{last_error}"
    )


def _formula_key(formula: Mapping[str, Any]) -> str:
    ascii_formula = re.sub(
        r"\s+",
        "",
        str(formula.get("ascii") or "").lower(),
    )
    if ascii_formula:
        return ascii_formula
    return str(formula.get("formula_id") or "").lower()


def _unique_formula_id(
    requested: str,
    used: set[str],
) -> str:
    base = re.sub(r"[^a-z0-9_]+", "_", requested.lower()).strip("_")
    if not base or not base[0].isalpha():
        base = f"formula_{base}" if base else "formula"

    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate



def assemble_research_payload(
    *,
    narrative: Mapping[str, Any],
    evidence: Sequence[Evidence],
    extracted_formulas: Sequence[Mapping[str, Any]],
    extraction_caveats: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {item.evidence_id: item for item in evidence}
    citation_by_range: dict[tuple[int, int], str] = {}
    citations: list[dict[str, Any]] = []

    def citation_for_range(start: int, end: int) -> str:
        key = (start, end)
        citation_id = citation_by_range.get(key)
        if citation_id is None:
            citation_id = f"C{len(citations) + 1}"
            citation_by_range[key] = citation_id
            citations.append(
                {
                    "citation_id": citation_id,
                    "start_segment": start,
                    "end_segment": end,
                }
            )
        return citation_id

    def map_evidence_ids(values: Sequence[str]) -> list[str]:
        mapped: list[str] = []
        for evidence_id in values:
            item = by_id[evidence_id]
            citation_id = citation_for_range(
                item.start_segment,
                item.end_segment,
            )
            if citation_id not in mapped:
                mapped.append(citation_id)
        return mapped

    formulas: list[dict[str, Any]] = []
    formulas_by_key: dict[str, dict[str, Any]] = {}
    used_formula_ids: set[str] = set()

    for extracted in extracted_formulas:
        formula_evidence = extracted["_evidence"]
        citation_id = citation_for_range(
            formula_evidence.start_segment,
            formula_evidence.end_segment,
        )
        key = _formula_key(extracted)

        if key in formulas_by_key:
            existing = formulas_by_key[key]
            if citation_id not in existing["citation_ids"]:
                existing["citation_ids"].append(citation_id)
            continue

        description = str(
            extracted.get("description") or ""
        ).strip()
        derivation_steps = [
            f"Meaning: {description}",
            *[
                str(step).strip()
                for step in extracted.get("derivation_steps") or []
                if str(step).strip()
            ],
        ]

        formula = {
            "formula_id": _unique_formula_id(
                str(extracted.get("formula_id") or "formula"),
                used_formula_ids,
            ),
            "name": str(extracted.get("name") or "").strip(),
            "ascii": str(extracted.get("ascii") or "").strip(),
            "latex": str(extracted.get("latex") or "").strip(),
            "derivation_type": str(
                extracted.get("derivation_type") or ""
            ),
            "variables": list(extracted.get("variables") or []),
            "derivation_steps": derivation_steps,
            "citation_ids": [citation_id],
        }
        formulas.append(formula)
        formulas_by_key[key] = formula

    caveats: list[str] = []

    research = {
        "executive_summary": str(
            narrative["executive_summary"]
        ).strip(),
        "executive_summary_citation_ids": map_evidence_ids(
            narrative["executive_summary_evidence_ids"]
        ),
        "key_takeaways": [
            {
                "text": str(item["text"]).strip(),
                "citation_ids": map_evidence_ids(
                    item["evidence_ids"]
                ),
            }
            for item in narrative["key_takeaways"]
        ],
        "sections": [
            {
                "heading": str(item["heading"]).strip(),
                "summary": str(item["summary"]).strip(),
                "citation_ids": map_evidence_ids(
                    item["evidence_ids"]
                ),
            }
            for item in narrative["sections"]
        ],
        "formulas": formulas,
        "caveats": caveats,
    }

    return citations, research



def _usage_from_response(response: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
        "request_count",
        "split_count",
    )
    usage = {
        field: response[field]
        for field in fields
        if field in response
    }

    message = response.get("message")
    if isinstance(message, Mapping):
        thinking = message.get("thinking")
        if isinstance(thinking, str):
            usage["thinking_chars"] = len(thinking)

    return usage


def analyze_video(
    *,
    video_id: str,
    extraction_model: str = DEFAULT_EXTRACTION_MODEL,
    narrative_model: str = DEFAULT_NARRATIVE_MODEL,
    extraction_think: bool = DEFAULT_EXTRACTION_THINK,
    narrative_think: bool = DEFAULT_NARRATIVE_THINK,
    model: str | None = None,
    raw_root: str | Path = "Raw Transcripts",
    inputs_root: str | Path = "Research Inputs",
    processed_root: str | Path = "Processed Research",
    manifest_path: str | Path = "manifests/research.jsonl",
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    num_ctx: int = DEFAULT_NUM_CTX,
    chunk_token_budget: int = DEFAULT_CHUNK_TOKEN_BUDGET,
    client: OllamaClient | None = None,
    reasoning_effort: str | None = None,
) -> Path:
    if model is not None:
        extraction_model = model
        narrative_model = model

    source = TranscriptSourcePackage.load(raw_root, video_id)
    ResearchInputPackageWriter(inputs_root).write(source)
    manifest = ResearchManifestStore(manifest_path)
    queued = manifest.queue(source)

    if (
        queued.get("status") == "research_ready"
        and queued.get("source_package_sha256")
        == source.package_sha256
        and queued.get("prompt_version") == PROMPT_VERSION
        and queued.get("extraction_model") == extraction_model
        and queued.get("narrative_model") == narrative_model
        and queued.get("extraction_think") is extraction_think
        and queued.get("narrative_think") is narrative_think
        and queued.get("analysis_backend") == "ollama"
    ):
        ready_path = Path(processed_root) / video_id / "_READY"
        if ready_path.is_file():
            print(
                f"SKIP: {video_id} already has a current research package"
            )
            return ready_path.parent

    api_client = client or OllamaClient(ollama_host)
    for required_model in dict.fromkeys(
        [extraction_model, narrative_model]
    ):
        api_client.require_model(required_model)

    manifest.transition(
        video_id=video_id,
        new_status="analyzing",
        title=str(source.metadata.get("title") or ""),
        url=str(source.metadata.get("source_url") or ""),
        updates={
            "source_package_sha256": source.package_sha256,
            "analysis_model": extraction_model,
            "extraction_model": extraction_model,
            "narrative_model": narrative_model,
            "extraction_think": extraction_think,
            "narrative_think": narrative_think,
            "analysis_backend": "ollama",
            "ollama_host": ollama_host,
            "prompt_version": PROMPT_VERSION,
            "num_ctx": num_ctx,
            "chunk_token_budget": chunk_token_budget,
            "reasoning_effort_compat": reasoning_effort,
        },
    )

    try:
        effective_chunk_budget = safe_chunk_token_budget(
            chunk_token_budget,
            num_ctx,
        )
        if effective_chunk_budget != chunk_token_budget:
            print(
                f"CAP: requested chunk budget {chunk_token_budget} -> "
                f"{effective_chunk_budget} for num_ctx={num_ctx}"
            )
        chunks = chunk_transcript(
            source.segments,
            token_budget=effective_chunk_budget,
        )
        print(
            f"Research plan: {len(source.segments)} segments, "
            f"{len(chunks)} analysis chunks, "
            f"extraction={extraction_model} think={str(extraction_think).lower()}, "
            f"narrative={narrative_model} think={str(narrative_think).lower()}"
        )

        chunk_payloads: list[dict[str, Any]] = []
        usage: dict[str, Any] = {
            "routes": {
                "extraction": {
                    "model": extraction_model,
                    "think": extraction_think,
                },
                "narrative": {
                    "model": narrative_model,
                    "think": narrative_think,
                },
            },
            "chunk_count": len(chunks),
            "requested_chunk_token_budget": chunk_token_budget,
            "effective_chunk_token_budget": effective_chunk_budget,
            "chunks": [],
        }

        for chunk in chunks:
            payload, response = extract_chunk_evidence(
                client=api_client,
                model=extraction_model,
                source=source,
                chunk=chunk,
                num_ctx=num_ctx,
                think=extraction_think,
            )
            chunk_payloads.append(payload)
            usage["chunks"].append(
                {
                    "chunk_index": chunk.chunk_index,
                    "start_segment": chunk.start_segment,
                    "end_segment": chunk.end_segment,
                    "estimated_input_tokens": (
                        chunk.estimated_tokens
                    ),
                    **_usage_from_response(response),
                }
            )
            chunk_status = (
                "PASS"
                if payload["claims"] or payload["formulas"]
                else "SKIP"
            )
            print(
                f"{chunk_status}: chunk {chunk.chunk_index + 1}/"
                f"{len(chunks)} "
                f"(segments {chunk.start_segment}-"
                f"{chunk.end_segment})"
            )

        evidence, formulas, extraction_caveats = (
            build_evidence_catalog(chunk_payloads)
        )
        if not evidence:
            raise OllamaError(
                "No grounded claims or formulas were extracted from "
                "the complete transcript"
            )
        narrative, final_response = synthesize_narrative(
            client=api_client,
            model=narrative_model,
            source=source,
            evidence=evidence,
            num_ctx=num_ctx,
            think=narrative_think,
        )
        usage["final_synthesis"] = _usage_from_response(
            final_response
        )
        usage["evidence_count"] = len(evidence)
        usage["formula_candidates"] = len(formulas)

        citations, research = assemble_research_payload(
            narrative=narrative,
            evidence=evidence,
            extracted_formulas=formulas,
            extraction_caveats=extraction_caveats,
        )

        result = ResearchPackageWriter(processed_root).write(
            source=source,
            research=research,
            citations=citations,
            analysis_backend=_analysis_backend_label(
                extraction_model,
                narrative_model,
            ),
            prompt_version=PROMPT_VERSION,
            analysis_details={
                "provider": "ollama",
                "extraction": {
                    "model": extraction_model,
                    "think": extraction_think,
                    "num_ctx": num_ctx,
                },
                "narrative": {
                    "model": narrative_model,
                    "think": narrative_think,
                    "num_ctx": num_ctx,
                },
            },
        )

        manifest.transition(
            video_id=video_id,
            new_status="research_ready",
            updates={
                "source_package_sha256": source.package_sha256,
                "research_directory": str(result.directory),
                "research_package_sha256": result.package_sha256,
                "analysis_model": extraction_model,
                "extraction_model": extraction_model,
                "narrative_model": narrative_model,
                "extraction_think": extraction_think,
                "narrative_think": narrative_think,
                "analysis_backend": "ollama",
                "ollama_host": ollama_host,
                "prompt_version": PROMPT_VERSION,
                "usage": usage,
            },
        )

        print(
            f"PASS: research package written to {result.directory}"
        )
        print(f"Package SHA-256: {result.package_sha256}")
        return result.directory

    except Exception as exc:
        current = manifest.get(video_id)
        if current and current.get("status") in {
            "analysis_queued",
            "analyzing",
        }:
            manifest.transition(
                video_id=video_id,
                new_status="analysis_failed",
                error=f"{type(exc).__name__}: {exc}",
                updates={
                    "source_package_sha256": source.package_sha256,
                    "analysis_model": extraction_model,
                    "extraction_model": extraction_model,
                    "narrative_model": narrative_model,
                    "extraction_think": extraction_think,
                    "narrative_think": narrative_think,
                    "analysis_backend": "ollama",
                    "ollama_host": ollama_host,
                    "prompt_version": PROMPT_VERSION,
                },
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one local transcript-grounded research package "
            "through Ollama."
        )
    )
    parser.add_argument("video_id")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Backward-compatible override that sets both extraction and "
            "narrative models."
        ),
    )
    parser.add_argument(
        "--extraction-model",
        default=os.environ.get(
            "OLLAMA_EXTRACTION_MODEL",
            os.environ.get(
                "OLLAMA_RESEARCH_MODEL",
                DEFAULT_EXTRACTION_MODEL,
            ),
        ),
    )
    parser.add_argument(
        "--narrative-model",
        default=os.environ.get(
            "OLLAMA_NARRATIVE_MODEL",
            os.environ.get(
                "OLLAMA_RESEARCH_MODEL",
                DEFAULT_NARRATIVE_MODEL,
            ),
        ),
    )
    parser.add_argument(
        "--extraction-think",
        action=argparse.BooleanOptionalAction,
        default=_env_bool(
            "OLLAMA_EXTRACTION_THINK",
            DEFAULT_EXTRACTION_THINK,
        ),
    )
    parser.add_argument(
        "--narrative-think",
        action=argparse.BooleanOptionalAction,
        default=_env_bool(
            "OLLAMA_NARRATIVE_THINK",
            DEFAULT_NARRATIVE_THINK,
        ),
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get(
            "OLLAMA_HOST",
            DEFAULT_OLLAMA_HOST,
        ),
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=int(
            os.environ.get(
                "OLLAMA_RESEARCH_NUM_CTX",
                DEFAULT_NUM_CTX,
            )
        ),
    )
    parser.add_argument(
        "--chunk-token-budget",
        type=int,
        default=int(
            os.environ.get(
                "OLLAMA_RESEARCH_CHUNK_TOKENS",
                DEFAULT_CHUNK_TOKEN_BUDGET,
            )
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("OPENAI_REASONING_EFFORT"),
        help="Accepted for backward compatibility; Ollama ignores it.",
    )
    parser.add_argument("--raw-root", default="Raw Transcripts")
    parser.add_argument("--inputs-root", default="Research Inputs")
    parser.add_argument(
        "--processed-root",
        default="Processed Research",
    )
    parser.add_argument(
        "--manifest",
        default="manifests/research.jsonl",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        analyze_video(
            video_id=args.video_id,
            extraction_model=(
                args.model or args.extraction_model
            ),
            narrative_model=(
                args.model or args.narrative_model
            ),
            extraction_think=args.extraction_think,
            narrative_think=args.narrative_think,
            raw_root=args.raw_root,
            inputs_root=args.inputs_root,
            processed_root=args.processed_root,
            manifest_path=args.manifest,
            ollama_host=args.ollama_host,
            num_ctx=args.num_ctx,
            chunk_token_budget=args.chunk_token_budget,
            reasoning_effort=args.reasoning_effort,
        )
    except (OllamaError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
