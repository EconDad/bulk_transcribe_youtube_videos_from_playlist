#!/usr/bin/env python3
"""Isolated v4.3 Stage C-D.1 diagnostic runner.

This runner does not replace or import the production v4.1.1 runner.
"""

from __future__ import annotations

import argparse
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
    build_inventory_chunks,
    build_inventory_prompt,
    merge_inventories,
    parse_inventory_response,
)
from research_v43.coverage import reconcile_coverage
from research_v43.entailment import (
    build_entailment_prompt,
    validate_entailment_response,
)
from research_v43.formula_extraction import (
    ExtractionDisposition,
    build_formula_extraction_prompt,
    parse_formula_extraction_response,
)
from research_v43.model_client import OllamaJsonClient


PROMPT_VERSION = "phase4-qwen3-v4.3-stage-cd.1"
INVENTORY_SYSTEM_PROMPT = (
    "You identify source-grounded calculation events. Return strict JSON. "
    "Do not inject outside formulas or subject-matter knowledge."
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

    _log(
        f"START {stage}; prompt_chars={len(user_prompt)}; "
        f"num_predict={num_predict}"
    )
    response = client.complete_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stage=stage,
        num_predict=num_predict,
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

    inventory = _run_inventory(
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

    retained_formulas: list[dict[str, Any]] = []
    entailment_reports: list[dict[str, Any]] = []
    rejected_formulas: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []

    for item in inventory.calculations:
        if item.source_mode is SourceMode.VISUAL_CUE:
            resolutions.append(
                {
                    "calculation_id": item.calculation_id,
                    "state": "visual_review_required",
                    "formula_ids": [],
                    "reason": (
                        "The source announces a visual equation; Stage C-D.1 "
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
        extraction_payload = _checkpointed_model_call(
            client=client,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_formula_extraction_prompt(
                item=item,
                segments=segments,
            ),
            stage=extraction_stage,
            checkpoint=extraction_checkpoint,
            expected={
                "schema_version": "1.0",
                "video_id": video_id,
                "source_package_sha256": source_sha,
                "prompt_version": PROMPT_VERSION,
                "stage": "formula_extraction",
                "calculation_id": item.calculation_id,
                "calculation_sha256": item_sha,
            },
            resume=resume,
            num_predict=detail_num_predict,
            invocations=invocations,
        )
        try:
            extraction = parse_formula_extraction_response(
                extraction_payload,
                item=item,
            )
        except Exception as exc:
            extraction_checkpoint.unlink(missing_ok=True)
            reason = (
                "Invalid formula extraction response: "
                f"{type(exc).__name__}: {exc}"
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
            entailment_payload = _checkpointed_model_call(
                client=client,
                system_prompt=ENTAILMENT_SYSTEM_PROMPT,
                user_prompt=build_entailment_prompt(
                    item=item,
                    candidate=candidate,
                    segments=segments,
                ),
                stage=entailment_stage,
                checkpoint=entailment_checkpoint,
                expected={
                    "schema_version": "1.0",
                    "video_id": video_id,
                    "source_package_sha256": source_sha,
                    "prompt_version": PROMPT_VERSION,
                    "stage": "formula_entailment",
                    "calculation_id": item.calculation_id,
                    "formula_id": candidate.formula_id,
                    "candidate_sha256": candidate_sha,
                },
                resume=resume,
                num_predict=detail_num_predict,
                invocations=invocations,
            )
            try:
                report = validate_entailment_response(
                    entailment_payload,
                    item=item,
                    candidate=candidate,
                    segments=segments,
                )
            except Exception as exc:
                entailment_checkpoint.unlink(missing_ok=True)
                reason = (
                    "Invalid entailment response: "
                    f"{type(exc).__name__}: {exc}"
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
                retained_formulas.append(candidate.to_dict())
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
        prompt_version=PROMPT_VERSION,
        payloads=payloads,
    )
    return (0 if coverage.passed else 2), result.package_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated research pipeline v4.3 Stage C-D.1 diagnostics."
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
