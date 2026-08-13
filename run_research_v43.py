#!/usr/bin/env python3
"""Isolated v4.3 Stage C-D.3 diagnostic runner.

This runner does not replace or import the production v4.1.1 runner.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from research_v43.artifacts import write_diagnostic_package
from research_v43.calculation_inventory import (
    CalculationInventory,
    SourceMode,
    audit_visual_equation_cues,
    build_inventory_chunks,
    build_inventory_prompt,
    merge_inventories,
    parse_inventory_response,
)
from research_v43.coverage import reconcile_coverage
from research_v43.entailment import (
    FormulaEntailmentReport,
    NodeStatus,
    build_entailment_prompt,
    build_entailment_repair_prompt,
    validate_entailment_response,
)
from research_v43.formula_extraction import (
    ExtractionDisposition,
    FormulaExtractionResult,
    build_formula_extraction_prompt,
    build_formula_extraction_repair_prompt,
    parse_formula_extraction_response,
)
from research_v43.expression_ast import (
    DerivationType,
    FormulaCandidate,
)
from research_v43.inventory_evidence_audit import (
    AuditAction,
    apply_inventory_audit_decision,
    build_inventory_evidence_audit_prompt,
    build_inventory_evidence_repair_prompt,
    decision_evidence_records,
    find_deterministic_expansion,
    item_needs_evidence_audit,
    parse_inventory_evidence_audit_response,
)
from research_v43.model_client import ModelClientError, OllamaJsonClient


PROMPT_VERSION = "phase4-qwen3-v4.3-stage-cd.1"
INVENTORY_AUDIT_VERSION = "phase4-qwen3-v4.3-inventory-audit-cd.4b3.1"
EVIDENCE_AUDIT_PROMPT_VERSION = (
    "phase4-qwen3-v4.3-inventory-evidence-audit-cd.4b3.1"
)
EXTRACTION_PROMPT_VERSION = "phase4-qwen3-v4.3-extraction-cd.4a"
ENTAILMENT_PROMPT_VERSION = "phase4-qwen3-v4.3-entailment-cd.4b3.1"
PACKAGE_VERSION = "phase4-qwen3-v4.3-stage-cd.4c"
ENTAILMENT_INFERENCE_MODE = "direct-json-no-thinking-v1"
INVENTORY_SYSTEM_PROMPT = (
    "You identify source-grounded calculation events. Return strict JSON. "
    "Do not inject outside formulas or subject-matter knowledge."
)
INVENTORY_AUDIT_SYSTEM_PROMPT = (
    "You audit one bounded calculation inventory item against exact transcript "
    "evidence. Return strict JSON. Do not invent formulas or outside facts."
)
EXTRACTION_SYSTEM_PROMPT = (
    "You normalize formulas from one bounded calculation event. Return strict "
    "JSON and use only the supplied source evidence."
)
ENTAILMENT_SYSTEM_PROMPT = (
    "You map every formula expression node to exact source evidence or to a "
    "validated algebraic dependency. Return strict JSON only."
)


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_stamp()}] {message}", flush=True)


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    for key, value in expected.items():
        if payload.get(key) != value:
            return None
    response_payload = payload.get("response_payload")
    if not isinstance(response_payload, Mapping):
        return None
    return payload


def load_transcript_source(
    *,
    raw_root: str | Path,
    video_id: str,
) -> tuple[list[dict[str, Any]], str, Mapping[str, Any]]:
    package = Path(raw_root) / video_id
    transcript_path = package / "transcript.json"
    if not transcript_path.is_file():
        raise FileNotFoundError(
            f"Transcript package is missing: {transcript_path}"
        )

    raw = json.loads(transcript_path.read_text(encoding="utf-8"))
    raw_segments = raw.get("segments") if isinstance(raw, Mapping) else raw
    if not isinstance(raw_segments, Sequence) or isinstance(
        raw_segments, (str, bytes)
    ):
        raise ValueError("transcript.json must contain a segment array")

    segments: list[dict[str, Any]] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            raise ValueError(f"Transcript segment {index} is not an object")
        text = raw_segment.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Transcript segment {index} has no usable text"
            )
        segment = dict(raw_segment)
        segment["segment_id"] = index
        segment["text"] = text.strip()
        segments.append(segment)

    ready_path = package / "_READY"
    ready: Mapping[str, Any] = {}
    if ready_path.is_file():
        parsed_ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if isinstance(parsed_ready, Mapping):
            ready = parsed_ready

    source_sha = str(
        ready.get("package_sha256")
        or ready.get("source_package_sha256")
        or ""
    ).strip()
    if not source_sha:
        source_sha = hashlib.sha256(transcript_path.read_bytes()).hexdigest()

    metadata_path = package / "metadata.json"
    metadata: Mapping[str, Any] = {}
    if metadata_path.is_file():
        parsed_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if isinstance(parsed_metadata, Mapping):
            metadata = parsed_metadata

    return segments, source_sha, metadata


def _run_inventory(
    *,
    video_id: str,
    segments: Sequence[Mapping[str, Any]],
    source_sha: str,
    client: Any,
    progress_dir: Path,
    chunk_segments: int,
    overlap_segments: int,
    resume: bool,
    inventory_num_predict: int,
    invocations: list[dict[str, Any]],
) -> CalculationInventory:
    chunks = build_inventory_chunks(
        segments,
        chunk_segments=chunk_segments,
        overlap_segments=overlap_segments,
    )
    _log(
        "INVENTORY PLAN: "
        f"{len(segments)} segments -> {len(chunks)} chunks "
        f"(size={chunk_segments}, overlap={overlap_segments})"
    )

    inventories: list[CalculationInventory] = []
    for position, chunk in enumerate(chunks, start=1):
        segment_payload = [
            {
                "segment_id": item.get("segment_id"),
                "text": item.get("text"),
            }
            for item in chunk.segments
        ]
        chunk_sha = _canonical_sha(segment_payload)
        expected = {
            "schema_version": "1.0",
            "video_id": video_id,
            "source_package_sha256": source_sha,
            "prompt_version": PROMPT_VERSION,
            "stage": "calculation_inventory",
            "chunk_index": chunk.chunk_index,
            "chunk_sha256": chunk_sha,
        }
        checkpoint = (
            progress_dir
            / "inventory"
            / f"chunk_{chunk.chunk_index:04d}.json"
        )
        cached = (
            _load_checkpoint(checkpoint, expected=expected)
            if resume
            else None
        )
        stage = (
            f"calculation_inventory chunk {position}/{len(chunks)} "
            f"segments {chunk.start_segment}-{chunk.end_segment}"
        )

        if cached is not None:
            _log(f"RESUME {stage}")
            response_payload = cached["response_payload"]
            invocations.append(
                {
                    "stage": "calculation_inventory",
                    "chunk_index": chunk.chunk_index,
                    "start_segment": chunk.start_segment,
                    "end_segment": chunk.end_segment,
                    "cache_hit": True,
                }
            )
        else:
            prompt = build_inventory_prompt(
                video_id=video_id,
                segments=chunk.segments,
            )
            _log(
                f"START {stage}; prompt_chars={len(prompt)}; "
                f"num_predict={inventory_num_predict}"
            )
            started = time.monotonic()
            response = client.complete_json(
                system_prompt=INVENTORY_SYSTEM_PROMPT,
                user_prompt=prompt,
                stage=stage,
                num_predict=inventory_num_predict,
            )
            elapsed = time.monotonic() - started
            response_payload = response.payload
            invocations.append(
                {
                    "stage": "calculation_inventory",
                    "chunk_index": chunk.chunk_index,
                    "start_segment": chunk.start_segment,
                    "end_segment": chunk.end_segment,
                    "cache_hit": False,
                    **response.invocation.to_dict(),
                }
            )

        try:
            inventory = parse_inventory_response(
                json.dumps(response_payload),
                expected_video_id=video_id,
                maximum_segment=len(segments) - 1,
            )
        except Exception:
            checkpoint.unlink(missing_ok=True)
            raise

        if cached is None:
            _atomic_write_json(
                checkpoint,
                {
                    **expected,
                    "start_segment": chunk.start_segment,
                    "end_segment": chunk.end_segment,
                    "response_payload": response_payload,
                    "invocation": response.invocation.to_dict(),
                },
            )
            _log(f"PASS {stage}; elapsed={elapsed:.1f}s")
        for item in inventory.calculations:
            if (
                item.start_segment < chunk.start_segment
                or item.end_segment > chunk.end_segment
            ):
                raise ValueError(
                    f"{stage}: {item.calculation_id} falls outside "
                    "the supplied chunk"
                )
        inventories.append(inventory)

    merged = merge_inventories(
        video_id=video_id,
        inventories=inventories,
    )
    _log(
        "INVENTORY MERGE: "
        f"{sum(len(item.calculations) for item in inventories)} raw -> "
        f"{len(merged.calculations)} unique calculations"
    )
    return merged


def _checkpointed_model_call(
    *,
    client: Any,
    system_prompt: str,
    user_prompt: str,
    stage: str,
    checkpoint: Path,
    expected: Mapping[str, Any],
    resume: bool,
    num_predict: int,
    invocations: list[dict[str, Any]],
    think: bool | None = None,
) -> Mapping[str, Any]:
    cached = (
        _load_checkpoint(checkpoint, expected=expected)
        if resume
        else None
    )
    if cached is not None:
        _log(f"RESUME {stage}")
        invocations.append({"stage": stage, "cache_hit": True})
        return cached["response_payload"]

    resolved_think = client.think if think is None else bool(think)
    _log(
        f"START {stage}; prompt_chars={len(user_prompt)}; "
        f"num_predict={num_predict}; "
        f"think={str(resolved_think).lower()}"
    )
    try:
        response = client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage=stage,
            num_predict=num_predict,
            think=resolved_think,
        )
    except ModelClientError as exc:
        if "response message content is empty" not in str(exc):
            raise
        retry_num_predict = max(num_predict, 2048)
        _log(
            f"RETRY {stage}; empty model content; "
            f"num_predict={retry_num_predict}; think=false"
        )
        response = client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage=stage,
            num_predict=retry_num_predict,
            think=False,
        )
    _atomic_write_json(
        checkpoint,
        {
            **expected,
            "response_payload": response.payload,
            "invocation": response.invocation.to_dict(),
        },
    )
    invocations.append(
        {"stage": stage, "cache_hit": False, **response.invocation.to_dict()}
    )
    _log(
        f"PASS {stage}; elapsed={response.invocation.elapsed_seconds:.1f}s"
    )
    return response.payload


def _extract_with_one_repair(
    *,
    item: Any,
    segments: Sequence[Mapping[str, Any]],
    client: Any,
    checkpoint: Path,
    expected: Mapping[str, Any],
    resume: bool,
    num_predict: int,
    invocations: list[dict[str, Any]],
) -> tuple[
    FormulaExtractionResult | None,
    Mapping[str, Any],
    str | None,
]:
    stage = f"formula_extraction {item.calculation_id}"
    payload = _checkpointed_model_call(
        client=client,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=build_formula_extraction_prompt(
            item=item,
            segments=segments,
        ),
        stage=stage,
        checkpoint=checkpoint,
        expected=expected,
        resume=resume,
        num_predict=num_predict,
        invocations=invocations,
        think=True,
    )

    try:
        return (
            parse_formula_extraction_response(payload, item=item),
            payload,
            None,
        )
    except Exception as first_error:
        checkpoint.unlink(missing_ok=True)
        first_reason = f"{type(first_error).__name__}: {first_error}"
        repair_stage = f"formula_extraction_repair {item.calculation_id}"
        _log(f"RETRY {repair_stage}; validation_error={first_reason}")
        repaired_payload = _checkpointed_model_call(
            client=client,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_formula_extraction_repair_prompt(
                item=item,
                segments=segments,
                invalid_payload=payload,
                validation_error=first_reason,
            ),
            stage=repair_stage,
            checkpoint=checkpoint,
            expected=expected,
            resume=False,
            num_predict=max(num_predict, 2048),
            invocations=invocations,
            think=False,
        )

        try:
            return (
                parse_formula_extraction_response(
                    repaired_payload,
                    item=item,
                ),
                repaired_payload,
                None,
            )
        except Exception as repair_error:
            checkpoint.unlink(missing_ok=True)
            reason = (
                f"initial={first_reason}; "
                f"repair={type(repair_error).__name__}: {repair_error}"
            )
            return None, repaired_payload, reason


def _entail_with_one_repair(
    *,
    item: Any,
    candidate: FormulaCandidate,
    segments: Sequence[Mapping[str, Any]],
    client: Any,
    checkpoint: Path,
    expected: Mapping[str, Any],
    resume: bool,
    num_predict: int,
    invocations: list[dict[str, Any]],
) -> tuple[
    FormulaEntailmentReport | None,
    Mapping[str, Any],
    str | None,
]:
    stage = (
        f"formula_entailment {item.calculation_id}/"
        f"{candidate.formula_id}"
    )
    payload = _checkpointed_model_call(
        client=client,
        system_prompt=ENTAILMENT_SYSTEM_PROMPT,
        user_prompt=build_entailment_prompt(
            item=item,
            candidate=candidate,
            segments=segments,
        ),
        stage=stage,
        checkpoint=checkpoint,
        expected=expected,
        resume=resume,
        num_predict=num_predict,
        invocations=invocations,
        think=False,
    )

    validation_issues: list[str]
    try:
        report = validate_entailment_response(
            payload,
            item=item,
            candidate=candidate,
            segments=segments,
        )
        validation_issues = [
            issue
            for issue in report.issues
            if (
                "expression does not match AST" in issue
                or "operation does not match AST" in issue
            )
        ]
        if not validation_issues:
            return report, payload, None
    except Exception as first_error:
        checkpoint.unlink(missing_ok=True)
        reason = f"{type(first_error).__name__}: {first_error}"
        return None, payload, reason

    checkpoint.unlink(missing_ok=True)
    repair_stage = (
        f"formula_entailment_repair {item.calculation_id}/"
        f"{candidate.formula_id}"
    )
    _log(
        f"RETRY {repair_stage}; "
        f"validation_issues={validation_issues}"
    )
    repaired_payload = _checkpointed_model_call(
        client=client,
        system_prompt=ENTAILMENT_SYSTEM_PROMPT,
        user_prompt=build_entailment_repair_prompt(
            item=item,
            candidate=candidate,
            segments=segments,
            invalid_payload=payload,
            validation_issues=validation_issues,
        ),
        stage=repair_stage,
        checkpoint=checkpoint,
        expected=expected,
        resume=False,
        num_predict=max(num_predict, 2048),
        invocations=invocations,
        think=False,
    )

    try:
        repaired_report = validate_entailment_response(
            repaired_payload,
            item=item,
            candidate=candidate,
            segments=segments,
        )
        return repaired_report, repaired_payload, None
    except Exception as repair_error:
        checkpoint.unlink(missing_ok=True)
        reason = f"{type(repair_error).__name__}: {repair_error}"
        return None, repaired_payload, reason


def _normalize_derivation_classification(
    candidate: FormulaCandidate,
    report: FormulaEntailmentReport,
) -> FormulaCandidate:
    """Promote stated formulas when validated nodes require derivation."""

    has_derived_node = any(
        node.status is NodeStatus.DERIVED
        for node in report.nodes
    )
    if (
        candidate.derivation_type is DerivationType.STATED
        and has_derived_node
    ):
        return replace(
            candidate,
            derivation_type=DerivationType.DERIVED,
        )
    return candidate



def _run_inventory_evidence_audit(
    *,
    inventory: CalculationInventory,
    segments: Sequence[Mapping[str, Any]],
    video_id: str,
    source_sha: str,
    client: Any,
    progress_dir: Path,
    resume: bool,
    num_predict: int,
    invocations: list[dict[str, Any]],
    radius: int = 8,
) -> tuple[CalculationInventory, tuple[dict[str, Any], ...]]:
    """Run deterministic-first bounded inventory evidence auditing."""

    audited_items = []
    audit_records: list[dict[str, Any]] = []

    for item in inventory.calculations:
        needs_audit, selection_reasons = item_needs_evidence_audit(
            item=item,
            segments=segments,
        )
        if not needs_audit:
            audited_items.append(item)
            continue

        neighborhood_start = max(0, item.start_segment - radius)
        neighborhood_end = min(
            len(segments) - 1,
            item.end_segment + radius,
        )

        deterministic = find_deterministic_expansion(
            item=item,
            segments=segments,
            neighborhood_start=neighborhood_start,
            neighborhood_end=neighborhood_end,
        )
        if deterministic is not None:
            updated = apply_inventory_audit_decision(
                item=item,
                decision=deterministic,
            )
            audited_items.append(updated)
            audit_records.append(
                {
                    "calculation_id": item.calculation_id,
                    "action": "expand",
                    "decision_source": "deterministic",
                    "selection_reasons": list(selection_reasons),
                    "before": {
                        "start_segment": item.start_segment,
                        "end_segment": item.end_segment,
                        "formula_expected": item.formula_expected,
                    "variables_mentioned": list(item.variables_mentioned),
                    "operations_mentioned": list(item.operations_mentioned),
                    },
                    "after": {
                        "start_segment": updated.start_segment,
                        "end_segment": updated.end_segment,
                        "formula_expected": updated.formula_expected,
                    "variables_mentioned": list(updated.variables_mentioned),
                    "operations_mentioned": list(updated.operations_mentioned),
                    },
                    "reason": deterministic.reason,
                    "evidence": list(
                        decision_evidence_records(
                            decision=deterministic,
                            segments=segments,
                        )
                    ),
                }
            )
            _log(
                f"INVENTORY AUTO-EXPAND {item.calculation_id}: "
                f"S{item.start_segment}-S{item.end_segment} -> "
                f"S{updated.start_segment}-S{updated.end_segment}"
            )
            continue

        neighborhood = tuple(
            {
                "segment_id": index,
                "text": segments[index].get("text"),
            }
            for index in range(neighborhood_start, neighborhood_end + 1)
        )
        prompt = build_inventory_evidence_audit_prompt(
            item=item,
            neighborhood_segments=neighborhood,
            selection_reasons=selection_reasons,
        )
        item_sha = _canonical_sha(item.to_dict())
        neighborhood_sha = _canonical_sha(neighborhood)
        checkpoint = (
            progress_dir
            / "inventory_evidence_audit"
            / f"{item.calculation_id}.json"
        )
        expected = {
            "schema_version": "1.0",
            "video_id": video_id,
            "source_package_sha256": source_sha,
            "prompt_version": EVIDENCE_AUDIT_PROMPT_VERSION,
            "stage": "inventory_evidence_audit",
            "calculation_id": item.calculation_id,
            "calculation_sha256": item_sha,
            "neighborhood_sha256": neighborhood_sha,
            "radius": radius,
        }
        stage = f"inventory_evidence_audit {item.calculation_id}"

        payload = _checkpointed_model_call(
            client=client,
            system_prompt=INVENTORY_AUDIT_SYSTEM_PROMPT,
            user_prompt=prompt,
            stage=stage,
            checkpoint=checkpoint,
            expected=expected,
            resume=resume,
            num_predict=min(max(num_predict, 1024), 2048),
            invocations=invocations,
            think=False,
        )

        try:
            decision = parse_inventory_evidence_audit_response(
                json.dumps(payload),
                item=item,
                segments=segments,
                neighborhood_start=neighborhood_start,
                neighborhood_end=neighborhood_end,
            )
        except Exception as first_error:
            checkpoint.unlink(missing_ok=True)
            repair_prompt = build_inventory_evidence_repair_prompt(
                original_prompt=prompt,
                previous_response=payload,
                validation_error=(
                    f"{type(first_error).__name__}: {first_error}"
                ),
            )
            repair_checkpoint = (
                progress_dir
                / "inventory_evidence_audit"
                / f"{item.calculation_id}.repair.json"
            )
            repair_expected = {
                **expected,
                "stage": "inventory_evidence_audit_repair",
            }
            _log(
                f"RETRY inventory_evidence_audit_repair "
                f"{item.calculation_id}; validation_error="
                f"{type(first_error).__name__}: {first_error}"
            )
            repaired_payload = _checkpointed_model_call(
                client=client,
                system_prompt=INVENTORY_AUDIT_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                stage=(
                    "inventory_evidence_audit_repair "
                    f"{item.calculation_id}"
                ),
                checkpoint=repair_checkpoint,
                expected=repair_expected,
                resume=False,
                num_predict=2048,
                invocations=invocations,
                think=False,
            )
            try:
                decision = parse_inventory_evidence_audit_response(
                    json.dumps(repaired_payload),
                    item=item,
                    segments=segments,
                    neighborhood_start=neighborhood_start,
                    neighborhood_end=neighborhood_end,
                )
            except Exception as repair_error:
                _log(
                    f"REJECT inventory_evidence_audit "
                    f"{item.calculation_id}: "
                    f"{type(repair_error).__name__}: {repair_error}"
                )
                audited_items.append(item)
                audit_records.append(
                    {
                        "calculation_id": item.calculation_id,
                        "action": "audit_failed",
                        "decision_source": "model",
                        "selection_reasons": list(selection_reasons),
                        "reason": (
                            "Segment-ID audit remained invalid after one "
                            "repair; original inventory item preserved."
                        ),
                        "validation_error": (
                            f"{type(repair_error).__name__}: "
                            f"{repair_error}"
                        ),
                    }
                )
                continue

        updated = apply_inventory_audit_decision(
            item=item,
            decision=decision,
        )
        audited_items.append(updated)
        audit_records.append(
            {
                "calculation_id": item.calculation_id,
                "action": decision.action.value,
                "decision_source": "model",
                "selection_reasons": list(selection_reasons),
                "before": {
                    "start_segment": item.start_segment,
                    "end_segment": item.end_segment,
                    "formula_expected": item.formula_expected,
                    "variables_mentioned": list(item.variables_mentioned),
                    "operations_mentioned": list(item.operations_mentioned),
                },
                "after": {
                    "start_segment": updated.start_segment,
                    "end_segment": updated.end_segment,
                    "formula_expected": updated.formula_expected,
                    "variables_mentioned": list(updated.variables_mentioned),
                    "operations_mentioned": list(updated.operations_mentioned),
                },
                "reason": decision.reason,
                "evidence": list(
                    decision_evidence_records(
                        decision=decision,
                        segments=segments,
                    )
                ),
            }
        )

        if decision.action is AuditAction.EXPAND:
            _log(
                f"INVENTORY MODEL-EXPAND {item.calculation_id}: "
                f"S{item.start_segment}-S{item.end_segment} -> "
                f"S{updated.start_segment}-S{updated.end_segment}"
            )
        elif decision.action is AuditAction.RECONCILE:
            _log(
                f"INVENTORY RECONCILE {item.calculation_id}: "
                f"S{item.start_segment}-S{item.end_segment} -> "
                f"S{updated.start_segment}-S{updated.end_segment}; "
                f"variables={list(updated.variables_mentioned)}; "
                f"operations={list(updated.operations_mentioned)}"
            )
        else:
            _log(
                f"INVENTORY DOWNGRADE {item.calculation_id}: "
                "formula_expected=false"
            )

    return (
        CalculationInventory(
            schema_version=inventory.schema_version,
            video_id=inventory.video_id,
            calculations=tuple(audited_items),
        ),
        tuple(audit_records),
    )

def _audit_failures_by_id(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    # Index terminal inventory-audit failures for fail-closed routing.
    failures: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("action") != "audit_failed":
            continue
        calculation_id = record.get("calculation_id")
        if not isinstance(calculation_id, str) or not calculation_id:
            continue
        failures[calculation_id] = record
    return failures


def run_pipeline(
    *,
    video_id: str,
    client: Any,
    raw_root: str | Path = "Raw Transcripts",
    output_root: str | Path = "Research v43 Diagnostics",
    progress_root: str | Path = "Research v43 Progress",
    inventory_chunk_segments: int = 40,
    inventory_overlap_segments: int = 6,
    inventory_num_predict: int = 1536,
    detail_num_predict: int = 1536,
    resume: bool = True,
) -> tuple[int, Path]:
    segments, source_sha, source_metadata = load_transcript_source(
        raw_root=raw_root,
        video_id=video_id,
    )
    progress_dir = Path(progress_root) / video_id
    invocations: list[dict[str, Any]] = []

    raw_inventory = _run_inventory(
        video_id=video_id,
        segments=segments,
        source_sha=source_sha,
        client=client,
        progress_dir=progress_dir,
        chunk_segments=inventory_chunk_segments,
        overlap_segments=inventory_overlap_segments,
        resume=resume,
        inventory_num_predict=inventory_num_predict,
        invocations=invocations,
    )

    inventory, visual_audit_records = audit_visual_equation_cues(
        inventory=raw_inventory,
        segments=segments,
    )
    _log(
        "INVENTORY AUDIT: "
        f"{len(visual_audit_records)} visual cue promotion(s)"
    )

    inventory, evidence_audit_records = _run_inventory_evidence_audit(
        inventory=inventory,
        segments=segments,
        video_id=video_id,
        source_sha=source_sha,
        client=client,
        progress_dir=progress_dir,
        resume=resume,
        num_predict=detail_num_predict,
        invocations=invocations,
    )
    inventory_audit_records = (
        *visual_audit_records,
        *evidence_audit_records,
    )
    _log(
        "INVENTORY EVIDENCE AUDIT: "
        f"{len(evidence_audit_records)} selected item(s)"
    )
    audit_failures = _audit_failures_by_id(evidence_audit_records)

    retained_formulas: list[dict[str, Any]] = []
    entailment_reports: list[dict[str, Any]] = []
    rejected_formulas: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []

    for item in inventory.calculations:
        audit_failure = audit_failures.get(item.calculation_id)
        if audit_failure is not None:
            validation_error = audit_failure.get("validation_error")
            reason = (
                "Inventory evidence audit remained unresolved after one "
                "bounded repair; downstream formula extraction was skipped."
            )
            if isinstance(validation_error, str) and validation_error:
                reason += f" {validation_error}"
            resolutions.append(
                {
                    "calculation_id": item.calculation_id,
                    "state": "insufficient_source_detail",
                    "formula_ids": [],
                    "reason": reason,
                }
            )
            _log(
                f"SKIP formula_extraction {item.calculation_id}: "
                "inventory audit unresolved"
            )
            continue

        if not item.formula_expected and not item.visual_equation_cue:
            resolutions.append(
                {
                    "calculation_id": item.calculation_id,
                    "state": "non_symbolic_calculation",
                    "formula_ids": [],
                    "reason": (
                        "Inventory marked this event as not requiring a "
                        "reusable symbolic formula."
                    ),
                }
            )
            continue

        if item.visual_equation_cue:
            resolutions.append(
                {
                    "calculation_id": item.calculation_id,
                    "state": "visual_review_required",
                    "formula_ids": [],
                    "reason": (
                        "The source announces a visual equation; Stage C-D.4C "
                        "does not yet perform frame recovery."
                    ),
                }
            )
            continue

        item_payload = item.to_dict()
        item_sha = _canonical_sha(item_payload)
        extraction_stage = f"formula_extraction {item.calculation_id}"
        extraction_checkpoint = (
            progress_dir
            / "extraction"
            / f"{item.calculation_id}.json"
        )
        extraction_expected = {
            "schema_version": "1.0",
            "video_id": video_id,
            "source_package_sha256": source_sha,
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "stage": "formula_extraction",
            "calculation_id": item.calculation_id,
            "calculation_sha256": item_sha,
        }
        extraction, extraction_payload, extraction_error = (
            _extract_with_one_repair(
                item=item,
                segments=segments,
                client=client,
                checkpoint=extraction_checkpoint,
                expected=extraction_expected,
                resume=resume,
                num_predict=detail_num_predict,
                invocations=invocations,
            )
        )
        if extraction is None:
            reason = (
                "Invalid formula extraction response after one repair: "
                f"{extraction_error}"
            )
            _log(
                f"REJECT formula_extraction "
                f"{item.calculation_id}: {reason}"
            )
            rejected_formulas.append(
                {
                    "calculation_id": item.calculation_id,
                    "stage": "formula_extraction_validation",
                    "reason": reason,
                    "model_response": dict(extraction_payload),
                }
            )
            resolutions.append(
                {
                    "calculation_id": item.calculation_id,
                    "state": "formula_rejected",
                    "formula_ids": [],
                    "reason": reason,
                }
            )
            continue

        if extraction.disposition is not ExtractionDisposition.CANDIDATES_PROPOSED:
            state_map = {
                ExtractionDisposition.NON_SYMBOLIC_CALCULATION: (
                    "non_symbolic_calculation"
                ),
                ExtractionDisposition.INSUFFICIENT_SOURCE_DETAIL: (
                    "insufficient_source_detail"
                ),
                ExtractionDisposition.VISUAL_REVIEW_REQUIRED: (
                    "visual_review_required"
                ),
            }
            resolutions.append(
                {
                    "calculation_id": item.calculation_id,
                    "state": state_map[extraction.disposition],
                    "formula_ids": [],
                    "reason": extraction.reason,
                }
            )
            continue

        accepted_ids: list[str] = []
        for candidate in extraction.candidates:
            candidate_sha = _canonical_sha(candidate.to_dict())
            entailment_stage = (
                f"formula_entailment {item.calculation_id}/"
                f"{candidate.formula_id}"
            )
            entailment_checkpoint = (
                progress_dir
                / "entailment"
                / item.calculation_id
                / f"{candidate.formula_id}.json"
            )
            entailment_expected = {
                "schema_version": "1.0",
                "video_id": video_id,
                "source_package_sha256": source_sha,
                "prompt_version": ENTAILMENT_PROMPT_VERSION,
                "stage": "formula_entailment",
                "calculation_id": item.calculation_id,
                "formula_id": candidate.formula_id,
                "candidate_sha256": candidate_sha,
                "inference_mode": ENTAILMENT_INFERENCE_MODE,
            }
            report, entailment_payload, entailment_error = (
                _entail_with_one_repair(
                    item=item,
                    candidate=candidate,
                    segments=segments,
                    client=client,
                    checkpoint=entailment_checkpoint,
                    expected=entailment_expected,
                    resume=resume,
                    num_predict=detail_num_predict,
                    invocations=invocations,
                )
            )
            if report is None:
                reason = (
                    "Invalid entailment response after one repair: "
                    f"{entailment_error}"
                )
                _log(
                    f"REJECT {entailment_stage}: {reason}"
                )
                entailment_reports.append(
                    {
                        "calculation_id": item.calculation_id,
                        "formula_id": candidate.formula_id,
                        "passed": False,
                        "issues": [reason],
                        "nodes": [],
                    }
                )
                rejected_formulas.append(
                    {
                        "calculation_id": item.calculation_id,
                        "formula_id": candidate.formula_id,
                        "stage": "entailment_validation",
                        "reason": reason,
                        "candidate": candidate.to_dict(),
                        "model_response": dict(entailment_payload),
                    }
                )
                continue

            entailment_reports.append(report.to_dict())

            if report.passed:
                retained_candidate = _normalize_derivation_classification(
                    candidate,
                    report,
                )
                if retained_candidate is not candidate:
                    _log(
                        "NORMALIZE derivation_type "
                        f"{item.calculation_id}/{candidate.formula_id}: "
                        "stated -> derived"
                    )
                retained_formulas.append(retained_candidate.to_dict())
                accepted_ids.append(candidate.formula_id)
            else:
                rejected_formulas.append(
                    {
                        "calculation_id": item.calculation_id,
                        "formula_id": candidate.formula_id,
                        "stage": "entailment",
                        "reason": "; ".join(report.issues),
                        "candidate": candidate.to_dict(),
                    }
                )

        if accepted_ids:
            resolutions.append(
                {
                    "calculation_id": item.calculation_id,
                    "state": "formula_retained",
                    "formula_ids": accepted_ids,
                    "reason": (
                        "All retained formulas passed AST-node entailment."
                    ),
                }
            )
        else:
            resolutions.append(
                {
                    "calculation_id": item.calculation_id,
                    "state": "formula_rejected",
                    "formula_ids": [],
                    "reason": (
                        "Formula candidates were proposed, but none passed "
                        "expression-node entailment."
                    ),
                }
            )

    coverage = reconcile_coverage(
        inventory=inventory,
        resolutions=resolutions,
        formulas=retained_formulas,
    )

    payloads = {
        "calculation_inventory.json": inventory.to_dict(),
        "inventory_audit.json": {
            "schema_version": "1.0",
            "video_id": video_id,
            "audit_version": INVENTORY_AUDIT_VERSION,
            "records": list(inventory_audit_records),
        },
        "formulas.json": {
            "schema_version": "1.0",
            "video_id": video_id,
            "formulas": retained_formulas,
        },
        "formula_entailment.json": {
            "schema_version": "1.0",
            "video_id": video_id,
            "reports": entailment_reports,
        },
        "formula_coverage.json": {
            "schema_version": "1.0",
            "video_id": video_id,
            **coverage.to_dict(),
        },
        "rejected_formulas.json": {
            "schema_version": "1.0",
            "video_id": video_id,
            "rejected_formulas": rejected_formulas,
        },
        "model_invocations.json": {
            "schema_version": "1.0",
            "video_id": video_id,
            "source_title": source_metadata.get("title"),
            "invocations": invocations,
        },
    }

    result = write_diagnostic_package(
        output_root=output_root,
        video_id=video_id,
        source_package_sha256=source_sha,
        prompt_version=PACKAGE_VERSION,
        payloads=payloads,
    )
    return (0 if coverage.passed else 2), result.package_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated research pipeline v4.3 Stage C-D.4C diagnostics."
        )
    )
    parser.add_argument("video_id")
    parser.add_argument("--raw-root", default="Raw Transcripts")
    parser.add_argument(
        "--output-root",
        default="Research v43 Diagnostics",
    )
    parser.add_argument(
        "--progress-root",
        default="Research v43 Progress",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get(
            "OLLAMA_HOST",
            "http://127.0.0.1:11434",
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "OLLAMA_EXTRACTION_MODEL",
            "qwen3:8b",
        ),
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=int(os.environ.get("OLLAMA_RESEARCH_NUM_CTX", "8192")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(
            os.environ.get("OLLAMA_RESEARCH_TIMEOUT_SECONDS", "300")
        ),
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=int(
            os.environ.get("OLLAMA_RESEARCH_NUM_PREDICT", "1536")
        ),
    )
    parser.add_argument(
        "--keep-alive",
        default=os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
    )
    parser.add_argument(
        "--inventory-chunk-segments",
        type=int,
        default=int(
            os.environ.get("V43_INVENTORY_CHUNK_SEGMENTS", "40")
        ),
    )
    parser.add_argument(
        "--inventory-overlap-segments",
        type=int,
        default=int(
            os.environ.get("V43_INVENTORY_OVERLAP_SEGMENTS", "6")
        ),
    )
    parser.add_argument(
        "--inventory-num-predict",
        type=int,
        default=int(
            os.environ.get("V43_INVENTORY_NUM_PREDICT", "1536")
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = OllamaJsonClient(
        host=args.host,
        model=args.model,
        think=True,
        num_ctx=args.num_ctx,
        timeout_seconds=args.timeout,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive,
    )
    try:
        exit_code, package = run_pipeline(
            video_id=args.video_id,
            client=client,
            raw_root=args.raw_root,
            output_root=args.output_root,
            progress_root=args.progress_root,
            inventory_chunk_segments=args.inventory_chunk_segments,
            inventory_overlap_segments=args.inventory_overlap_segments,
            inventory_num_predict=args.inventory_num_predict,
            detail_num_predict=args.num_predict,
            resume=not args.no_resume,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Diagnostic package: {package}")
    if exit_code == 0:
        print("PASS: formula coverage is complete")
    else:
        print("REVIEW REQUIRED: formula coverage contains unresolved items")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
