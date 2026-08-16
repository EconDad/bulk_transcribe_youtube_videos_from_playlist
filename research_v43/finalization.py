"""Stage F narrative synthesis and final package promotion for research v4.3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .artifacts import verify_diagnostic_package


FINAL_PROMPT_VERSION = "phase4-qwen3-v4.3-stage-f.1"
DIAGNOSTIC_PROMPT_VERSION = "phase4-qwen3-v4.3-stage-e.3"
MAX_CITATION_SEGMENTS = 6


class FinalizationError(RuntimeError):
    """Raised when a diagnostic package cannot be promoted safely."""


@dataclass(frozen=True, slots=True)
class NarrativeEvidence:
    evidence_id: str
    topic: str
    text: str
    explanation: str
    start_segment: int
    end_segment: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "topic": self.topic,
            "text": self.text,
            "explanation": self.explanation,
            "start_segment": self.start_segment,
            "end_segment": self.end_segment,
        }


@dataclass(frozen=True, slots=True)
class FinalPackageResult:
    package_dir: Path
    package_sha256: str
    artifact_sha256: Mapping[str, str]


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FinalizationError(f"Missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FinalizationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FinalizationError(f"Artifact must contain a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_digest(hashes: Mapping[str, str]) -> str:
    material = "\n".join(
        f"{name}:{hashes[name]}" for name in sorted(hashes)
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _segment_text(
    segments: Sequence[Mapping[str, Any]], start: int, end: int
) -> str:
    return " ".join(
        str(segments[index].get("text") or "").strip()
        for index in range(start, end + 1)
    ).strip()


def _numeric_tokens(text: str) -> set[str]:
    return {
        token.replace(",", "")
        for token in re.findall(
            r"(?<![A-Za-z_])\d[\d,]*(?:\.\d+)?%?",
            str(text),
        )
    }


def _validate_numeric_grounding(
    text: str, support_text: str, *, context: str
) -> None:
    claimed = _numeric_tokens(text)
    supported = _numeric_tokens(support_text)
    unsupported = sorted(claimed - supported)
    if unsupported:
        raise FinalizationError(
            f"{context} introduces unsupported numeric values: {unsupported}"
        )


def _validate_reader_prose(text: str, *, context: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        raise FinalizationError(f"{context} cannot be empty")
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", normalized):
        raise FinalizationError(f"{context} contains non-English script artifacts")
    if re.search(r"\b(?:CALC|NODE|EVIDENCE|PROMPT)_?\d+\b", normalized, re.I):
        raise FinalizationError(f"{context} leaks internal identifiers")
    if re.search(r"\b[a-z]+_[a-z0-9_]+\b", normalized):
        raise FinalizationError(f"{context} contains machine-style identifiers")
    lowered = normalized.lower()
    for phrase in (
        "system prompt",
        "validation error",
        "json schema",
        "model response",
        "evidence id",
        "citation id",
    ):
        if phrase in lowered:
            raise FinalizationError(f"{context} leaks pipeline/debug language")
    return normalized


def _validate_range(
    start: int,
    end: int,
    *,
    segment_count: int,
    minimum: int = 0,
    maximum: int | None = None,
    context: str,
) -> None:
    max_allowed = segment_count - 1 if maximum is None else maximum
    if start < minimum or end < start or end > max_allowed:
        raise FinalizationError(
            f"{context} has invalid segment range {start}-{end}"
        )
    if end - start + 1 > MAX_CITATION_SEGMENTS:
        raise FinalizationError(
            f"{context} citation exceeds {MAX_CITATION_SEGMENTS} segments"
        )


def validate_diagnostic_for_finalization(
    diagnostic_dir: str | Path,
    *,
    source_package_sha256: str,
) -> dict[str, Mapping[str, Any]]:
    package = Path(diagnostic_dir)
    issues = verify_diagnostic_package(
        package,
        source_package_sha256=source_package_sha256,
        prompt_version=DIAGNOSTIC_PROMPT_VERSION,
    )
    if issues:
        raise FinalizationError(
            "Diagnostic package failed freshness/integrity validation: "
            + "; ".join(issues)
        )

    required = {
        "calculation_inventory.json",
        "formulas.json",
        "formula_entailment.json",
        "formula_coverage.json",
        "rejected_formulas.json",
        "visual_evidence.json",
    }
    payloads = {name: _load_json(package / name) for name in sorted(required)}
    coverage = payloads["formula_coverage.json"]
    if coverage.get("passed") is not True:
        raise FinalizationError(
            "Formula coverage is incomplete; final research promotion is blocked"
        )
    if int(coverage.get("unresolved", 0)) != 0:
        raise FinalizationError(
            "Formula coverage reports unresolved calculations"
        )

    formulas = payloads["formulas.json"].get("formulas")
    if not isinstance(formulas, Sequence) or isinstance(formulas, (str, bytes)):
        raise FinalizationError("formulas.json formulas must be an array")

    visual_records = payloads["visual_evidence.json"].get("records")
    if not isinstance(visual_records, Sequence) or isinstance(
        visual_records, (str, bytes)
    ):
        raise FinalizationError("visual_evidence.json records must be an array")
    visual_by_calculation = {
        str(record.get("calculation_id")): record
        for record in visual_records
        if isinstance(record, Mapping)
        and isinstance(record.get("calculation_id"), str)
    }
    for index, formula in enumerate(formulas):
        if not isinstance(formula, Mapping):
            raise FinalizationError(f"formulas[{index}] must be an object")
        if formula.get("derivation_type") != "stated_visual":
            continue
        calculation_id = str(formula.get("calculation_id") or "")
        record = visual_by_calculation.get(calculation_id)
        if not record or record.get("status") != "formula_retained":
            raise FinalizationError(
                f"stated_visual formula {calculation_id} lacks accepted visual provenance"
            )
        consensus = record.get("consensus")
        if not isinstance(consensus, Mapping) or consensus.get("passed") is not True:
            raise FinalizationError(
                f"stated_visual formula {calculation_id} lacks passing consensus"
            )

    return payloads


def build_narrative_chunks(
    segments: Sequence[Mapping[str, Any]],
    *,
    chunk_segments: int = 40,
    overlap_segments: int = 6,
) -> tuple[tuple[int, int], ...]:
    if chunk_segments < 8:
        raise ValueError("chunk_segments must be at least 8")
    if overlap_segments < 0 or overlap_segments >= chunk_segments:
        raise ValueError("overlap_segments is invalid")
    if not segments:
        raise FinalizationError("Transcript has no segments")
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < len(segments):
        end = min(len(segments) - 1, start + chunk_segments - 1)
        chunks.append((start, end))
        if end == len(segments) - 1:
            break
        start = end - overlap_segments + 1
    return tuple(chunks)


def build_narrative_extraction_prompt(
    *,
    video_id: str,
    title: str,
    segments: Sequence[Mapping[str, Any]],
    start_segment: int,
    end_segment: int,
) -> str:
    lines = [
        f"Video ID: {video_id}",
        f"Title: {title}",
        f"Segment range: {start_segment}-{end_segment}",
        "",
        "Identify zero to five durable reader-facing ideas from this bounded transcript chunk.",
        "Use only this transcript. Do not add outside facts.",
        "Each item must use a citation range of at most six segments.",
        "Numeric values and dates must appear in the cited range.",
        "Do not discuss formulas as machine expressions; formula handling occurs separately.",
        "Do not mention prompts, schemas, validators, evidence IDs, or internal pipeline state.",
        "Return JSON with exactly one key, evidence, containing an array of objects with keys:",
        "topic, text, explanation, start_segment, end_segment.",
        "",
        "Transcript:",
    ]
    for index in range(start_segment, end_segment + 1):
        lines.append(f"[S{index}] {str(segments[index].get('text') or '').strip()}")
    return "\n".join(lines)


def parse_narrative_extraction(
    payload: Mapping[str, Any],
    *,
    segments: Sequence[Mapping[str, Any]],
    minimum_segment: int,
    maximum_segment: int,
) -> tuple[dict[str, Any], ...]:
    if set(payload) != {"evidence"}:
        raise FinalizationError("Narrative extraction returned unexpected keys")
    raw_items = payload.get("evidence")
    if isinstance(raw_items, (str, bytes)) or not isinstance(raw_items, Sequence):
        raise FinalizationError("Narrative evidence must be an array")
    if len(raw_items) > 5:
        raise FinalizationError("Narrative extraction returned more than five items")

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            raise FinalizationError(f"evidence[{index}] must be an object")
        required = {"topic", "text", "explanation", "start_segment", "end_segment"}
        if set(raw) != required:
            raise FinalizationError(f"evidence[{index}] has invalid keys")
        start = int(raw["start_segment"])
        end = int(raw["end_segment"])
        _validate_range(
            start,
            end,
            segment_count=len(segments),
            minimum=minimum_segment,
            maximum=maximum_segment,
            context=f"evidence[{index}]",
        )
        topic = _validate_reader_prose(raw["topic"], context=f"evidence[{index}].topic")
        text = _validate_reader_prose(raw["text"], context=f"evidence[{index}].text")
        explanation = _validate_reader_prose(
            raw["explanation"], context=f"evidence[{index}].explanation"
        )
        support = _segment_text(segments, start, end)
        _validate_numeric_grounding(
            f"{topic} {text} {explanation}",
            support,
            context=f"evidence[{index}]",
        )
        normalized.append(
            {
                "topic": topic,
                "text": text,
                "explanation": explanation,
                "start_segment": start,
                "end_segment": end,
            }
        )
    return tuple(normalized)


def merge_narrative_evidence(
    items: Sequence[Mapping[str, Any]],
) -> tuple[NarrativeEvidence, ...]:
    ordered = sorted(
        items,
        key=lambda item: (
            int(item["start_segment"]),
            int(item["end_segment"]),
            str(item["text"]).lower(),
        ),
    )
    seen: set[tuple[str, int, int]] = set()
    result: list[NarrativeEvidence] = []
    for item in ordered:
        key = (
            re.sub(r"\s+", " ", str(item["text"]).lower()).strip(),
            int(item["start_segment"]),
            int(item["end_segment"]),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(
            NarrativeEvidence(
                evidence_id=f"N{len(result) + 1:04d}",
                topic=str(item["topic"]),
                text=str(item["text"]),
                explanation=str(item["explanation"]),
                start_segment=int(item["start_segment"]),
                end_segment=int(item["end_segment"]),
            )
        )
    if not result:
        raise FinalizationError("No narrative evidence survived validation")
    return tuple(result)


def build_synthesis_prompt(
    *,
    title: str,
    evidence: Sequence[NarrativeEvidence],
    formulas: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"Video title: {title}",
        "",
        "Write a concise reader-facing research brief using only the validated evidence below.",
        "The executive summary must be exactly four complete sentences.",
        "Return four to seven distinct key takeaways and three to six sections ordered by transcript progression.",
        "Use only the supplied narrative evidence IDs. Attach one or two evidence IDs to each takeaway and no more than three to each section.",
        "Do not mention internal IDs in prose, prompts, validation, schemas, or machine variable names.",
        "Do not introduce numbers or dates absent from the cited narrative evidence.",
        "Validated formulas may inform wording but may not be converted into new claims beyond the evidence.",
        "Return JSON with exactly: executive_summary, executive_summary_evidence_ids, key_takeaways, sections.",
        "Each key_takeaway object has text and evidence_ids. Each section has heading, summary, evidence_ids.",
        "",
        "Narrative evidence:",
    ]
    for item in evidence:
        lines.append(
            f"[{item.evidence_id} | S{item.start_segment}-S{item.end_segment}] "
            f"{item.topic}. {item.text} {item.explanation}"
        )
    if formulas:
        lines.append("")
        lines.append("Validated formula context (do not cite these IDs directly in prose):")
        for formula in formulas:
            name = str(formula.get("name") or formula.get("formula_id") or "Formula")
            derivation = str(formula.get("derivation_type") or "")
            meanings = [
                str(variable.get("meaning") or "").strip()
                for variable in formula.get("variables") or []
                if isinstance(variable, Mapping)
                and str(variable.get("meaning") or "").strip()
            ]
            lines.append(f"- {name} ({derivation}); quantities: {', '.join(meanings)}")
    return "\n".join(lines)


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[^.!?]+[.!?](?:\s|$)", text.strip()))


def validate_synthesis(
    payload: Mapping[str, Any],
    *,
    evidence: Sequence[NarrativeEvidence],
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "executive_summary",
        "executive_summary_evidence_ids",
        "key_takeaways",
        "sections",
    }
    if set(payload) != required:
        raise FinalizationError("Narrative synthesis returned unexpected keys")
    by_id = {item.evidence_id: item for item in evidence}

    def validate_ids(values: Any, *, context: str, maximum: int) -> list[str]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise FinalizationError(f"{context} evidence_ids must be an array")
        ids = [str(value) for value in values]
        if not ids or len(ids) > maximum:
            raise FinalizationError(f"{context} has invalid evidence count")
        unknown = [value for value in ids if value not in by_id]
        if unknown:
            raise FinalizationError(f"{context} references unknown evidence: {unknown}")
        return ids

    def support_text(ids: Sequence[str]) -> str:
        ranges = [(by_id[value].start_segment, by_id[value].end_segment) for value in ids]
        return " ".join(_segment_text(segments, start, end) for start, end in ranges)

    summary = _validate_reader_prose(payload["executive_summary"], context="executive_summary")
    if _sentence_count(summary) != 4:
        raise FinalizationError("executive_summary must contain exactly four sentences")
    summary_ids = validate_ids(
        payload["executive_summary_evidence_ids"], context="executive_summary", maximum=6
    )
    _validate_numeric_grounding(summary, support_text(summary_ids), context="executive_summary")

    raw_takeaways = payload["key_takeaways"]
    if isinstance(raw_takeaways, (str, bytes)) or not isinstance(raw_takeaways, Sequence):
        raise FinalizationError("key_takeaways must be an array")
    if not 4 <= len(raw_takeaways) <= 7:
        raise FinalizationError("key_takeaways must contain four to seven items")
    takeaways: list[dict[str, Any]] = []
    seen_takeaways: set[str] = set()
    for index, item in enumerate(raw_takeaways):
        if not isinstance(item, Mapping) or set(item) != {"text", "evidence_ids"}:
            raise FinalizationError(f"key_takeaways[{index}] is invalid")
        text = _validate_reader_prose(item["text"], context=f"key_takeaways[{index}]")
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if normalized in seen_takeaways:
            raise FinalizationError("key_takeaways contain duplicates")
        seen_takeaways.add(normalized)
        ids = validate_ids(item["evidence_ids"], context=f"key_takeaways[{index}]", maximum=2)
        _validate_numeric_grounding(text, support_text(ids), context=f"key_takeaways[{index}]")
        takeaways.append({"text": text, "evidence_ids": ids})

    raw_sections = payload["sections"]
    if isinstance(raw_sections, (str, bytes)) or not isinstance(raw_sections, Sequence):
        raise FinalizationError("sections must be an array")
    if not 3 <= len(raw_sections) <= 6:
        raise FinalizationError("sections must contain three to six items")
    sections: list[dict[str, Any]] = []
    previous_start = -1
    for index, item in enumerate(raw_sections):
        if not isinstance(item, Mapping) or set(item) != {"heading", "summary", "evidence_ids"}:
            raise FinalizationError(f"sections[{index}] is invalid")
        heading = _validate_reader_prose(item["heading"], context=f"sections[{index}].heading")
        section_summary = _validate_reader_prose(item["summary"], context=f"sections[{index}].summary")
        ids = validate_ids(item["evidence_ids"], context=f"sections[{index}]", maximum=3)
        _validate_numeric_grounding(section_summary, support_text(ids), context=f"sections[{index}]")
        section_start = min(by_id[value].start_segment for value in ids)
        if section_start < previous_start:
            raise FinalizationError("sections are not ordered by transcript progression")
        previous_start = section_start
        sections.append({"heading": heading, "summary": section_summary, "evidence_ids": ids})

    return {
        "executive_summary": summary,
        "executive_summary_evidence_ids": summary_ids,
        "key_takeaways": takeaways,
        "sections": sections,
    }


def build_citations_and_research(
    *,
    narrative: Mapping[str, Any],
    evidence: Sequence[NarrativeEvidence],
    formulas: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_id = {item.evidence_id: item for item in evidence}
    ranges: dict[tuple[int, int], str] = {}
    citations: list[dict[str, Any]] = []

    def citation_for_range(start: int, end: int) -> str:
        key = (start, end)
        if key not in ranges:
            citation_id = f"C{len(citations) + 1}"
            ranges[key] = citation_id
            citations.append(
                {
                    "citation_id": citation_id,
                    "start_segment": start,
                    "end_segment": end,
                    "start_seconds": float(segments[start].get("start", 0.0)),
                    "end_seconds": float(segments[end].get("end", 0.0)),
                    "segment_count": end - start + 1,
                }
            )
        return ranges[key]

    def map_evidence(ids: Sequence[str]) -> list[str]:
        result: list[str] = []
        for evidence_id in ids:
            item = by_id[evidence_id]
            citation_id = citation_for_range(item.start_segment, item.end_segment)
            if citation_id not in result:
                result.append(citation_id)
        return result

    final_formulas: list[dict[str, Any]] = []
    for formula in formulas:
        source_claims = formula.get("source_claims") or []
        formula_citations: list[str] = []
        for claim in source_claims:
            if not isinstance(claim, Mapping):
                continue
            start = int(claim.get("start_segment", -1))
            end = int(claim.get("end_segment", start))
            _validate_range(
                start,
                end,
                segment_count=len(segments),
                context=f"formula {formula.get('formula_id')} source claim",
            )
            citation_id = citation_for_range(start, end)
            if citation_id not in formula_citations:
                formula_citations.append(citation_id)
        if not formula_citations:
            raise FinalizationError(
                f"Retained formula {formula.get('formula_id')} lacks source claims"
            )
        normalized = dict(formula)
        normalized["citation_ids"] = formula_citations
        final_formulas.append(normalized)

    research = {
        "schema_version": "1.0",
        "executive_summary": narrative["executive_summary"],
        "executive_summary_citation_ids": map_evidence(
            narrative["executive_summary_evidence_ids"]
        ),
        "key_takeaways": [
            {"text": item["text"], "citation_ids": map_evidence(item["evidence_ids"])}
            for item in narrative["key_takeaways"]
        ],
        "sections": [
            {
                "heading": item["heading"],
                "summary": item["summary"],
                "citation_ids": map_evidence(item["evidence_ids"]),
            }
            for item in narrative["sections"]
        ],
        "formulas": final_formulas,
        "caveats": [],
    }
    source_map = {"schema_version": "1.0", "citations": citations}
    formulas_payload = {"schema_version": "1.0", "formulas": final_formulas}
    return source_map, research, formulas_payload


def _labels(ids: Sequence[str]) -> str:
    return " ".join(f"[{value}]" for value in ids)


def render_markdown(
    *,
    video_id: str,
    title: str,
    source_url: str,
    source_sha: str,
    research: Mapping[str, Any],
    source_map: Mapping[str, Any],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- **Video ID:** `{video_id}`",
        f"- **Source:** {source_url}",
        f"- **Transcript package SHA-256:** `{source_sha}`",
        "",
        "## Executive Summary",
        "",
        f"{research['executive_summary']} {_labels(research['executive_summary_citation_ids'])}",
        "",
        "## Key Takeaways",
        "",
    ]
    for item in research["key_takeaways"]:
        lines.append(f"- {item['text']} {_labels(item['citation_ids'])}")
    lines += ["", "## Sections", ""]
    for item in research["sections"]:
        lines += [
            f"### {item['heading']}",
            "",
            f"{item['summary']} {_labels(item['citation_ids'])}",
            "",
        ]
    lines += ["## Formulas", ""]
    if not research["formulas"]:
        lines += ["No reusable formulas were retained from the source.", ""]
    for formula in research["formulas"]:
        lines += [
            f"### {formula.get('name') or formula.get('formula_id')}",
            "",
            f"- **Formula ID:** `{formula.get('formula_id')}`",
            f"- **Type:** {formula.get('derivation_type')}",
            f"- **Machine-readable:** `{formula.get('ascii')}`",
            f"- **LaTeX:** `${formula.get('latex')}$`",
            f"- **Sources:** {_labels(formula.get('citation_ids') or [])}",
        ]
        variables = formula.get("variables") or []
        if variables:
            lines += ["- **Variables:**", ""]
            for variable in variables:
                if not isinstance(variable, Mapping):
                    continue
                symbol = str(variable.get("symbol") or "")
                meaning = str(variable.get("meaning") or "")
                unit = str(variable.get("unit") or "")
                suffix = f" — unit: {unit}" if unit else ""
                lines.append(f"  - `{symbol}`: {meaning}{suffix}")
        steps = formula.get("derivation_steps") or []
        if steps:
            lines += ["- **Derivation:**", ""]
            for index, step in enumerate(steps, 1):
                lines.append(f"  {index}. {step}")
        lines.append("")
    lines += ["## Source Map", ""]
    for item in source_map["citations"]:
        lines.append(
            f"- **[{item['citation_id']}]** segments {item['start_segment']}-{item['end_segment']}; "
            f"{item['start_seconds']:.2f}s-{item['end_seconds']:.2f}s"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_final_package(
    *,
    output_root: str | Path,
    video_id: str,
    title: str,
    source_url: str,
    source_package_sha256: str,
    research: Mapping[str, Any],
    source_map: Mapping[str, Any],
    formulas_payload: Mapping[str, Any],
    diagnostic_payloads: Mapping[str, Mapping[str, Any]],
    analysis_details: Mapping[str, Any],
) -> FinalPackageResult:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / video_id
    temp_parent = Path(tempfile.mkdtemp(prefix=f".{video_id}.f1.", dir=root))
    staging = temp_parent / video_id
    staging.mkdir()
    backup = root / f".{video_id}.backup"
    if backup.exists():
        shutil.rmtree(backup)
    try:
        research_payload = {
            "schema_version": "1.0",
            "video_id": video_id,
            "title": title,
            "source_package_sha256": source_package_sha256,
            **dict(research),
        }
        source_map_payload = {
            "schema_version": "1.0",
            "video_id": video_id,
            "source_package_sha256": source_package_sha256,
            **dict(source_map),
        }
        formula_payload = {
            "schema_version": "1.0",
            "video_id": video_id,
            "source_package_sha256": source_package_sha256,
            **dict(formulas_payload),
        }
        payloads: dict[str, Any] = {
            "research.json": research_payload,
            "formulas.json": formula_payload,
            "source_map.json": source_map_payload,
            "calculation_inventory.json": diagnostic_payloads["calculation_inventory.json"],
            "formula_entailment.json": diagnostic_payloads["formula_entailment.json"],
            "formula_coverage.json": diagnostic_payloads["formula_coverage.json"],
            "rejected_formulas.json": diagnostic_payloads["rejected_formulas.json"],
            "visual_evidence.json": diagnostic_payloads["visual_evidence.json"],
        }
        for name, payload in payloads.items():
            (staging / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        markdown = render_markdown(
            video_id=video_id,
            title=title,
            source_url=source_url,
            source_sha=source_package_sha256,
            research=research,
            source_map=source_map,
        )
        (staging / "research.md").write_text(markdown, encoding="utf-8")
        artifact_names = sorted([*payloads, "research.md"])
        hashes = {name: _sha256_file(staging / name) for name in artifact_names}
        metadata = {
            "schema_version": "1.0",
            "video_id": video_id,
            "title": title,
            "source_url": source_url,
            "source_package_sha256": source_package_sha256,
            "prompt_version": FINAL_PROMPT_VERSION,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": hashes,
            "analysis_details": dict(analysis_details),
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        hashes["metadata.json"] = _sha256_file(staging / "metadata.json")
        package_sha = _package_digest(hashes)
        ready = {
            "schema_version": "1.0",
            "video_id": video_id,
            "status": "research_ready",
            "source_package_sha256": source_package_sha256,
            "prompt_version": FINAL_PROMPT_VERSION,
            "package_sha256": package_sha,
            "artifact_sha256": hashes,
            "ready_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "_READY").write_text(
            json.dumps(ready, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return FinalPackageResult(target, package_sha, hashes)
    except Exception as exc:
        if isinstance(exc, FinalizationError):
            raise
        raise FinalizationError(f"Failed to write final research package: {exc}") from exc
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


def verify_final_package(
    package_dir: str | Path,
    *,
    source_package_sha256: str,
    prompt_version: str = FINAL_PROMPT_VERSION,
) -> list[str]:
    package = Path(package_dir)
    required = {
        "_READY",
        "metadata.json",
        "research.md",
        "research.json",
        "formulas.json",
        "source_map.json",
        "calculation_inventory.json",
        "formula_entailment.json",
        "formula_coverage.json",
        "rejected_formulas.json",
        "visual_evidence.json",
    }
    issues = [f"Missing artifact: {name}" for name in sorted(required) if not (package / name).is_file()]
    if issues:
        return issues
    try:
        metadata = _load_json(package / "metadata.json")
        ready = _load_json(package / "_READY")
    except FinalizationError as exc:
        return [str(exc)]
    if metadata.get("source_package_sha256") != source_package_sha256:
        issues.append("metadata source SHA is stale")
    if ready.get("source_package_sha256") != source_package_sha256:
        issues.append("_READY source SHA is stale")
    if metadata.get("prompt_version") != prompt_version:
        issues.append("metadata prompt version is stale")
    if ready.get("prompt_version") != prompt_version:
        issues.append("_READY prompt version is stale")
    hashes = ready.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        issues.append("_READY artifact hashes are missing")
        return issues
    actual_hashes: dict[str, str] = {}
    for name, expected in hashes.items():
        path = package / str(name)
        if not path.is_file():
            issues.append(f"Missing hashed artifact: {name}")
            continue
        actual = _sha256_file(path)
        actual_hashes[str(name)] = actual
        if actual != expected:
            issues.append(f"Artifact hash mismatch: {name}")
    if ready.get("package_sha256") != _package_digest(actual_hashes):
        issues.append("Package SHA mismatch")
    coverage = _load_json(package / "formula_coverage.json")
    if coverage.get("passed") is not True or int(coverage.get("unresolved", 0)) != 0:
        issues.append("Final package contains incomplete formula coverage")
    return issues
