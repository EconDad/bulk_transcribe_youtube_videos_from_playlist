"""Autonomous visual-equation recovery for research pipeline v4.3.

The vision model performs literal source-frame transcription only. Formula
retention is deterministic: parser-safe normalization, the shared safe AST
parser, and cross-frame structural agreement decide acceptance.
"""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error, request

from .calculation_inventory import CalculationItem
from .expression_ast import (
    ExpressionValidationError,
    FormulaCandidate,
    parse_formula,
)


class VisualEvidenceError(RuntimeError):
    """Raised for deterministic Stage E visual-evidence failures."""


VISION_PROMPT = r"""
Perform literal visual transcription of the principal mathematical equation
visible in this image.

If no mathematical equation is visibly present, set equation_present to false
and leave equation_ascii and equation_latex empty.

Transcribe only what is visibly present. Do not derive, simplify, rearrange,
complete, correct, identify, or infer mathematical content from outside
knowledge. Grouping is critical.

For equation_ascii use / for division and ^(...) for exponents. Preserve
fraction-bar grouping, numerator/denominator grouping, brackets, parentheses,
variable names, and subscripts as closely as possible. Use * for multiplication
only when explicitly visible.

For equation_latex preserve the same mathematical tree. For visible_variables,
include a meaning only when that meaning is visibly printed; otherwise use an
empty string. List genuinely ambiguous glyphs/operators/exponents/subscripts or
grouping in uncertain_tokens. Confidence is confidence in literal visual
transcription only.

Return only the requested structured object.
""".strip()

VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "equation_present": {"type": "boolean"},
        "equation_ascii": {"type": "string"},
        "equation_latex": {"type": "string"},
        "visible_variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "visible_meaning": {"type": "string"},
                },
                "required": ["symbol", "visible_meaning"],
                "additionalProperties": False,
            },
        },
        "uncertain_tokens": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "equation_present",
        "equation_ascii",
        "equation_latex",
        "visible_variables",
        "uncertain_tokens",
        "confidence",
    ],
    "additionalProperties": False,
}

_SUBSCRIPT_DIGITS = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
}
_ALLOWED_FUNCTIONS = {"sum", "sqrt", "log", "exp", "abs", "min", "max"}


@dataclass(frozen=True, slots=True)
class VisualRecoveryConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3-vl:8b-instruct"
    num_ctx: int = 8192
    num_predict: int = 1536
    timeout_seconds: float = 600.0
    keep_alive: str = "10m"
    yt_dlp_path: str = "yt-dlp"
    deno_path: str = ""
    ffmpeg_path: str = "ffmpeg"
    frame_count: int = 7
    min_consensus: int = 3
    max_height: int = 1080

    def __post_init__(self) -> None:
        if not self.host.strip() or not self.model.strip():
            raise ValueError("visual host/model cannot be empty")
        if self.num_ctx < 1024 or self.num_predict < 64:
            raise ValueError("visual model limits are invalid")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.frame_count < 3:
            raise ValueError("frame_count must be at least 3")
        if not 2 <= self.min_consensus <= self.frame_count:
            raise ValueError("min_consensus is invalid")
        if self.max_height <= 0:
            raise ValueError("max_height must be positive")


@dataclass(frozen=True, slots=True)
class VisualRecoveryResult:
    calculation_id: str
    state: str
    reason: str
    candidate: FormulaCandidate | None
    evidence: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_unicode_subscripts(text: str) -> str:
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9_]*)([₀₁₂₃₄₅₆₇₈₉]+)")

    def replace(match: re.Match[str]) -> str:
        digits = "".join(_SUBSCRIPT_DIGITS[ch] for ch in match.group(2))
        return f"{match.group(1)}_{digits}"

    return pattern.sub(replace, text)


def normalize_visual_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        raise VisualEvidenceError("visual symbol must be nonempty")
    normalized = _replace_unicode_subscripts(symbol.strip()).lower()
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise VisualEvidenceError(f"cannot normalize visual symbol: {symbol!r}")
    if not normalized[0].isalpha():
        normalized = f"v_{normalized}"
    if not re.fullmatch(r"[a-z][a-z0-9_]*", normalized):
        raise VisualEvidenceError(f"visual symbol is not parser-safe: {symbol!r}")
    return normalized


def _normalize_visual_formula_ascii(raw_ascii: str) -> tuple[str, dict[str, str]]:
    if not isinstance(raw_ascii, str) or not raw_ascii.strip():
        raise VisualEvidenceError("visual equation ASCII is empty")
    text = raw_ascii.strip()
    for old, new in {
        "−": "-", "–": "-", "—": "-", "×": "*", "·": "*", "⋅": "*",
        "÷": "/", "／": "/", "［": "[", "］": "]", "（": "(", "）": ")",
    }.items():
        text = text.replace(old, new)
    text = _replace_unicode_subscripts(text).replace("[", "(").replace("]", ")")
    if text.count("=") != 1:
        raise VisualEvidenceError("visual equation must contain one assignment")

    symbol_map: dict[str, str] = {}

    def replace_identifier(match: re.Match[str]) -> str:
        raw = match.group(0)
        normalized = normalize_visual_symbol(raw)
        symbol_map.setdefault(raw, normalized)
        return normalized

    text = re.sub(r"[A-Za-z][A-Za-z0-9_]*", replace_identifier, text)
    text = re.sub(r"(?<![a-z0-9_])(\d+(?:\.\d+)?)(?=[a-z_])", r"\1*", text)
    text = re.sub(r"(?<=\))(?=\()", "*", text)
    text = re.sub(r"(?<=\))(?=[a-z_])", "*", text)
    text = re.sub(r"(?<=\))(?=\d)", "*", text)
    text = re.sub(r"(?<=\d)(?=\()", "*", text)
    text = re.sub(r"\s+", " ", text).strip()
    for function_name in _ALLOWED_FUNCTIONS:
        text = re.sub(
            rf"\b{re.escape(function_name)}\s*\*\s*\(",
            f"{function_name}(",
            text,
        )
    return text, symbol_map


def normalize_visual_formula_ascii(raw_ascii: str) -> str:
    normalized, _ = _normalize_visual_formula_ascii(raw_ascii)
    return normalized


def select_visual_consensus(
    frame_records: Sequence[Mapping[str, Any]], *, min_consensus: int
) -> dict[str, Any]:
    eligible: list[Mapping[str, Any]] = []
    for record in frame_records:
        if record.get("status") != "ok":
            continue
        result = record.get("result")
        if not isinstance(result, Mapping) or result.get("equation_present") is not True:
            continue
        uncertain = result.get("uncertain_tokens")
        if (
            isinstance(uncertain, (str, bytes))
            or not isinstance(uncertain, Sequence)
            or uncertain
        ):
            continue
        canonical = record.get("parsed_canonical_ascii")
        if isinstance(canonical, str) and canonical:
            eligible.append(record)

    counts = Counter(str(r["parsed_canonical_ascii"]) for r in eligible)
    groups = [
        {"canonical_ascii": canonical, "count": count}
        for canonical, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    if not eligible:
        return {
            "passed": False,
            "reason": "No clean parser-valid equation frames were available.",
            "eligible_frames": 0,
            "required_count": min_consensus,
            "groups": groups,
            "winner_canonical_ascii": "",
            "winner_count": 0,
        }
    winner_canonical, winner_count = counts.most_common(1)[0]
    second_count = counts.most_common(2)[1][1] if len(counts) > 1 else 0
    passed = (
        winner_count >= min_consensus
        and winner_count > second_count
        and winner_count * 2 > len(eligible)
    )
    if winner_count < min_consensus:
        reason = f"Consensus count {winner_count} is below required {min_consensus}."
    elif winner_count <= second_count:
        reason = "Visual equation consensus is tied."
    elif winner_count * 2 <= len(eligible):
        reason = "Visual equation consensus lacks a strict majority."
    else:
        reason = (
            f"{winner_count}/{len(eligible)} clean equation frames agree on the "
            "same shared-parser AST."
        )
    return {
        "passed": passed,
        "reason": reason,
        "eligible_frames": len(eligible),
        "required_count": min_consensus,
        "groups": groups,
        "winner_canonical_ascii": winner_canonical,
        "winner_count": winner_count,
    }


def _cue_window(
    item: CalculationItem, segments: Sequence[Mapping[str, Any]]
) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    for index in range(item.start_segment, item.end_segment + 1):
        if index >= len(segments):
            raise VisualEvidenceError(f"{item.calculation_id} exceeds transcript")
        start = segments[index].get("start")
        end = segments[index].get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise VisualEvidenceError(f"segment {index} lacks numeric timing")
        starts.append(float(start))
        ends.append(float(end))
    low, high = min(starts), max(ends)
    if high <= low:
        raise VisualEvidenceError("visual cue has non-positive duration")
    return low, high


def _frame_timestamps(start: float, end: float, count: int) -> tuple[float, ...]:
    span = end - start
    margin = min(0.5, span * 0.08)
    low, high = start + margin, end - margin
    if high <= low:
        return tuple((start + end) / 2 for _ in range(count))
    step = (high - low) / (count - 1)
    return tuple(round(low + step * i, 3) for i in range(count))


def _source_url(video_id: str, metadata: Mapping[str, Any]) -> str:
    for key in ("webpage_url", "source_url", "video_url", "original_url", "url"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
            return value.strip()
    return f"https://www.youtube.com/watch?v={video_id}"


def _read_info_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _safe_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "--version"], check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip().splitlines()[0] if completed.returncode == 0 else ""


def _validate_vision_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = set(VISION_SCHEMA["required"])
    if set(payload) != required:
        raise VisualEvidenceError("vision response schema mismatch")
    if not isinstance(payload["equation_present"], bool):
        raise VisualEvidenceError("equation_present must be boolean")
    for field in ("equation_ascii", "equation_latex"):
        if not isinstance(payload[field], str):
            raise VisualEvidenceError(f"{field} must be a string")
    variables = payload["visible_variables"]
    if isinstance(variables, (str, bytes)) or not isinstance(variables, Sequence):
        raise VisualEvidenceError("visible_variables must be an array")
    normalized_variables: list[dict[str, str]] = []
    for item in variables:
        if not isinstance(item, Mapping) or set(item) != {"symbol", "visible_meaning"}:
            raise VisualEvidenceError("visible variable is invalid")
        if not isinstance(item["symbol"], str) or not isinstance(item["visible_meaning"], str):
            raise VisualEvidenceError("visible variable values must be strings")
        normalized_variables.append(dict(item))
    uncertain = payload["uncertain_tokens"]
    if (
        isinstance(uncertain, (str, bytes))
        or not isinstance(uncertain, Sequence)
        or not all(isinstance(item, str) for item in uncertain)
    ):
        raise VisualEvidenceError("uncertain_tokens must be a string array")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise VisualEvidenceError("confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise VisualEvidenceError("confidence must be between 0 and 1")
    if payload["equation_present"] and not payload["equation_ascii"].strip():
        raise VisualEvidenceError("equation_present=true requires equation_ascii")
    return {
        "equation_present": payload["equation_present"],
        "equation_ascii": payload["equation_ascii"].strip(),
        "equation_latex": payload["equation_latex"].strip(),
        "visible_variables": normalized_variables,
        "uncertain_tokens": list(uncertain),
        "confidence": float(confidence),
    }


class VisualEquationRecoverer:
    def __init__(
        self,
        config: VisualRecoveryConfig,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.logger = logger

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)

    def _fail(
        self,
        *,
        item: CalculationItem,
        evidence: dict[str, Any],
        stage: str,
        reason: str,
    ) -> VisualRecoveryResult:
        evidence.update(
            {"status": "visual_review_required", "failure_stage": stage, "reason": reason}
        )
        self._log(f"VISUAL REVIEW {item.calculation_id}: {stage}: {reason}")
        return VisualRecoveryResult(
            calculation_id=item.calculation_id,
            state="visual_review_required",
            reason=reason,
            candidate=None,
            evidence=evidence,
        )

    def _vision_request(
        self, frame_path: Path, *, unload_after: bool
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        image = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.config.model,
            "stream": False,
            "format": VISION_SCHEMA,
            "messages": [{"role": "user", "content": VISION_PROMPT, "images": [image]}],
            "options": {
                "temperature": 0,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
            },
            "keep_alive": 0 if unload_after else self.config.keep_alive,
        }
        req = request.Request(
            f"{self.config.host.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise VisualEvidenceError(f"vision request failed: {exc}") from exc
        elapsed = time.monotonic() - started
        if not isinstance(raw, Mapping) or not isinstance(raw.get("message"), Mapping):
            raise VisualEvidenceError("vision endpoint response is invalid")
        message = raw["message"]
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            raise VisualEvidenceError("vision instruct model returned thinking content")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise VisualEvidenceError("vision response content is empty")
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise VisualEvidenceError("vision content is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise VisualEvidenceError("vision structured content must be an object")
        return _validate_vision_payload(decoded), {
            "model": self.config.model,
            "elapsed_seconds": round(elapsed, 3),
            "done_reason": raw.get("done_reason") if isinstance(raw.get("done_reason"), str) else "",
            "prompt_eval_count": raw.get("prompt_eval_count") if isinstance(raw.get("prompt_eval_count"), int) else 0,
            "eval_count": raw.get("eval_count") if isinstance(raw.get("eval_count"), int) else 0,
            "thinking_chars": len(thinking) if isinstance(thinking, str) else 0,
        }

    def recover(
        self,
        *,
        video_id: str,
        item: CalculationItem,
        segments: Sequence[Mapping[str, Any]],
        source_metadata: Mapping[str, Any],
    ) -> VisualRecoveryResult:
        if not item.visual_equation_cue:
            raise VisualEvidenceError("visual recovery requires visual_equation_cue=true")
        start, end = _cue_window(item, segments)
        timestamps = _frame_timestamps(start, end, self.config.frame_count)
        source_url = _source_url(video_id, source_metadata)
        evidence: dict[str, Any] = {
            "schema_version": "1.0",
            "calculation_id": item.calculation_id,
            "source_mode": "visual",
            "cue": {
                "start_segment": item.start_segment,
                "end_segment": item.end_segment,
                "start_seconds": start,
                "end_seconds": end,
            },
            "acquisition": {
                "source_url": source_url,
                "client": "web_embedded",
                "downloader": "yt-dlp-native-http",
                "max_height": self.config.max_height,
                "yt_dlp_version": _safe_version(self.config.yt_dlp_path),
            },
            "vision": {
                "model": self.config.model,
                "temperature": 0,
                "num_ctx": self.config.num_ctx,
                "frame_count": self.config.frame_count,
                "min_consensus": self.config.min_consensus,
            },
            "frames": [],
            "consensus": {},
        }
        deno_path = os.path.expanduser(self.config.deno_path)
        if not deno_path or not Path(deno_path).is_file():
            return self._fail(
                item=item,
                evidence=evidence,
                stage="acquisition_preflight",
                reason="Deno runtime required for web_embedded acquisition is unavailable.",
            )

        with tempfile.TemporaryDirectory(prefix=f"research-v43-visual-{video_id}-") as tmp:
            temp_root = Path(tmp)
            media_root = temp_root / "media"
            frame_root = temp_root / "frames"
            media_root.mkdir()
            frame_root.mkdir()
            output_template = str(media_root / f"{video_id}.%(ext)s")
            command = [
                self.config.yt_dlp_path,
                "--no-playlist",
                "--js-runtimes",
                f"deno:{deno_path}",
                "--remote-components",
                "ejs:npm",
                "--extractor-args",
                "youtube:player_client=web_embedded",
                "--write-info-json",
                "-f",
                f"bestvideo[height<={self.config.max_height}]/best[height<={self.config.max_height}]",
                "-o",
                output_template,
                source_url,
            ]
            self._log(
                f"VISUAL ACQUIRE {item.calculation_id}: web_embedded video <= "
                f"{self.config.max_height}p; yt-dlp progress follows"
            )
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    timeout=max(self.config.timeout_seconds, 600.0),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return self._fail(
                    item=item,
                    evidence=evidence,
                    stage="media_acquisition",
                    reason=f"web_embedded acquisition failed: {exc}",
                )
            evidence["acquisition"]["exit_code"] = completed.returncode
            if completed.returncode != 0:
                return self._fail(
                    item=item,
                    evidence=evidence,
                    stage="media_acquisition",
                    reason=(
                        "web_embedded native media acquisition failed; no rejected "
                        "client fallback was attempted."
                    ),
                )
            info_path = media_root / f"{video_id}.info.json"
            info = _read_info_json(info_path)
            media_candidates = [
                path
                for path in media_root.glob(f"{video_id}.*")
                if path.is_file()
                and not path.name.endswith((".json", ".part", ".ytdl"))
            ]
            if len(media_candidates) != 1:
                return self._fail(
                    item=item,
                    evidence=evidence,
                    stage="media_acquisition",
                    reason="acquisition did not produce exactly one source-media file.",
                )
            media_path = media_candidates[0]
            media_sha = _sha256_file(media_path)
            evidence["acquisition"].update(
                {
                    "format_id": str(info.get("format_id") or ""),
                    "protocol": str(info.get("protocol") or ""),
                    "vcodec": str(info.get("vcodec") or ""),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "duration": info.get("duration"),
                    "filesize": media_path.stat().st_size,
                    "source_media_sha256": media_sha,
                }
            )
            self._log(
                f"VISUAL ACQUIRE PASS {item.calculation_id}: "
                f"{media_path.stat().st_size} bytes; sha256={media_sha[:16]}…"
            )

            frame_records: list[dict[str, Any]] = []
            for frame_index, timestamp in enumerate(timestamps, start=1):
                frame_path = frame_root / f"frame_{frame_index:02d}_{timestamp:.3f}.png"
                self._log(
                    f"VISUAL FRAME {frame_index}/{len(timestamps)}: extract t={timestamp:.3f}s"
                )
                process = subprocess.run(
                    [
                        self.config.ffmpeg_path,
                        "-nostdin",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{timestamp:.3f}",
                        "-i",
                        str(media_path),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(frame_path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if process.returncode != 0 or not frame_path.is_file():
                    frame_records.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp,
                            "status": "error",
                            "stage": "frame_extraction",
                            "error": "local ffmpeg frame extraction failed",
                        }
                    )
                    continue
                record: dict[str, Any] = {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "frame_sha256": _sha256_file(frame_path),
                    "status": "ok",
                }
                self._log(
                    f"VISUAL VISION {frame_index}/{len(timestamps)} START: "
                    f"{self.config.model}"
                )
                try:
                    result, invocation = self._vision_request(
                        frame_path, unload_after=frame_index == len(timestamps)
                    )
                    record["result"] = result
                    record["invocation"] = invocation
                    if result["equation_present"]:
                        normalized, symbol_map = _normalize_visual_formula_ascii(
                            result["equation_ascii"]
                        )
                        parsed = parse_formula(normalized)
                        record["normalized_ascii"] = normalized
                        record["symbol_map"] = symbol_map
                        record["parsed_canonical_ascii"] = parsed.canonical_ascii
                        record["parsed"] = parsed.to_dict()
                        self._log(
                            f"VISUAL VISION {frame_index}/{len(timestamps)} PASS: "
                            f"{parsed.canonical_ascii}"
                        )
                    else:
                        self._log(
                            f"VISUAL VISION {frame_index}/{len(timestamps)}: no equation"
                        )
                except (VisualEvidenceError, ExpressionValidationError) as exc:
                    record.update(
                        {"status": "error", "stage": "vision_validation", "error": str(exc)}
                    )
                    self._log(
                        f"VISUAL VISION {frame_index}/{len(timestamps)} REJECT: {exc}"
                    )
                frame_records.append(record)

            evidence["frames"] = frame_records
            consensus = select_visual_consensus(
                frame_records, min_consensus=self.config.min_consensus
            )
            evidence["consensus"] = consensus
            self._log(
                f"VISUAL CONSENSUS {item.calculation_id}: "
                f"winner={consensus['winner_count']}/{consensus['eligible_frames']}; "
                f"required={self.config.min_consensus}; passed={consensus['passed']}"
            )
            if not consensus["passed"]:
                return self._fail(
                    item=item,
                    evidence=evidence,
                    stage="visual_consensus",
                    reason=str(consensus["reason"]),
                )

            winner_canonical = str(consensus["winner_canonical_ascii"])
            winners = [
                record
                for record in frame_records
                if record.get("status") == "ok"
                and record.get("parsed_canonical_ascii") == winner_canonical
                and isinstance(record.get("result"), Mapping)
                and not record["result"].get("uncertain_tokens")
            ]
            parsed = parse_formula(winner_canonical)
            meaning_votes: dict[str, Counter[str]] = {
                identifier: Counter() for identifier in parsed.identifiers
            }
            for record in winners:
                for variable in record["result"]["visible_variables"]:
                    raw_symbol = variable["symbol"]
                    meaning = variable["visible_meaning"].strip()
                    if not raw_symbol.strip():
                        continue
                    try:
                        normalized_symbol = normalize_visual_symbol(raw_symbol)
                    except VisualEvidenceError:
                        continue
                    if normalized_symbol in meaning_votes and meaning:
                        meaning_votes[normalized_symbol][meaning] += 1
            variables = []
            for identifier in sorted(parsed.identifiers):
                votes = meaning_votes[identifier]
                meaning = (
                    votes.most_common(1)[0][0]
                    if votes
                    else f"Visible mathematical symbol {identifier}"
                )
                variables.append({"symbol": identifier, "meaning": meaning, "unit": ""})
            latex = winners[0]["result"]["equation_latex"].strip()
            if not latex:
                return self._fail(
                    item=item,
                    evidence=evidence,
                    stage="candidate_construction",
                    reason="consensus equation lacks literal LaTeX transcription.",
                )
            visual_source = {
                "source_url": source_url,
                "cue_start_segment": item.start_segment,
                "cue_end_segment": item.end_segment,
                "cue_start_seconds": start,
                "cue_end_seconds": end,
                "source_media_sha256": media_sha,
                "client": "web_embedded",
                "format_id": str(info.get("format_id") or ""),
                "vision_model": self.config.model,
                "consensus_canonical_ascii": winner_canonical,
                "consensus_count": consensus["winner_count"],
                "required_consensus_count": self.config.min_consensus,
                "frame_evidence": [
                    {
                        "timestamp_seconds": record["timestamp_seconds"],
                        "frame_sha256": record["frame_sha256"],
                    }
                    for record in winners
                ],
            }
            try:
                candidate = FormulaCandidate.from_mapping(
                    {
                        "calculation_id": item.calculation_id,
                        "formula_id": "visual_equation",
                        "name": item.name,
                        "ascii": winner_canonical,
                        "latex": latex,
                        "derivation_type": "stated_visual",
                        "variables": variables,
                        "derivation_steps": [
                            "Transcribed literally from independently agreeing source-video frames."
                        ],
                        "source_claims": [
                            {
                                "start_segment": item.start_segment,
                                "end_segment": item.end_segment,
                                "relationship": (
                                    "The transcript announces a displayed equation; the "
                                    "mathematical structure is recovered from independently "
                                    "agreeing source-video frames."
                                ),
                            }
                        ],
                        "visual_source": visual_source,
                    }
                )
            except ExpressionValidationError as exc:
                return self._fail(
                    item=item,
                    evidence=evidence,
                    stage="candidate_construction",
                    reason=f"visual candidate failed shared validation: {exc}",
                )
            evidence.update(
                {
                    "status": "formula_retained",
                    "reason": str(consensus["reason"]),
                    "formula_id": candidate.formula_id,
                    "candidate_canonical_ascii": candidate.parsed.canonical_ascii,
                }
            )
            return VisualRecoveryResult(
                calculation_id=item.calculation_id,
                state="formula_retained",
                reason=str(consensus["reason"]),
                candidate=candidate,
                evidence=evidence,
            )
