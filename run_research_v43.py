#!/usr/bin/env python3
"""Isolated v4.3 Stage C-D diagnostic runner.

This runner does not replace or import the production v4.1.1 runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from research_v43.artifacts import write_diagnostic_package
from research_v43.calculation_inventory import (
    CalculationInventory,
    SourceMode,
    build_inventory_prompt,
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


PROMPT_VERSION = "phase4-qwen3-v4.3-stage-cd"
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
    if isinstance(raw, Mapping):
        raw_segments = raw.get("segments")
    else:
        raw_segments = raw
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


def run_pipeline(
    *,
    video_id: str,
    client: Any,
    raw_root: str | Path = "Raw Transcripts",
    output_root: str | Path = "Research v43 Diagnostics",
) -> tuple[int, Path]:
    segments, source_sha, source_metadata = load_transcript_source(
        raw_root=raw_root,
        video_id=video_id,
    )

    invocations: list[dict[str, Any]] = []
    inventory_response = client.complete_json(
        system_prompt=INVENTORY_SYSTEM_PROMPT,
        user_prompt=build_inventory_prompt(
            video_id=video_id,
            segments=segments,
        ),
    )
    invocations.append(
        {
            "stage": "calculation_inventory",
            **inventory_response.invocation.to_dict(),
        }
    )
    inventory = parse_inventory_response(
        json.dumps(inventory_response.payload),
        expected_video_id=video_id,
        maximum_segment=len(segments) - 1,
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
                        "The source announces a visual equation; Stage C-D "
                        "does not yet perform frame recovery."
                    ),
                }
            )
            continue

        extraction_response = client.complete_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_formula_extraction_prompt(
                item=item,
                segments=segments,
            ),
        )
        invocations.append(
            {
                "stage": "formula_extraction",
                "calculation_id": item.calculation_id,
                **extraction_response.invocation.to_dict(),
            }
        )
        extraction = parse_formula_extraction_response(
            extraction_response.payload,
            item=item,
        )

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
            entailment_response = client.complete_json(
                system_prompt=ENTAILMENT_SYSTEM_PROMPT,
                user_prompt=build_entailment_prompt(
                    item=item,
                    candidate=candidate,
                    segments=segments,
                ),
            )
            invocations.append(
                {
                    "stage": "formula_entailment",
                    "calculation_id": item.calculation_id,
                    "formula_id": candidate.formula_id,
                    **entailment_response.invocation.to_dict(),
                }
            )
            report = validate_entailment_response(
                entailment_response.payload,
                item=item,
                candidate=candidate,
                segments=segments,
            )
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
        description="Run isolated research pipeline v4.3 Stage C-D diagnostics."
    )
    parser.add_argument("video_id")
    parser.add_argument("--raw-root", default="Raw Transcripts")
    parser.add_argument(
        "--output-root",
        default="Research v43 Diagnostics",
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
        default=int(
            os.environ.get("OLLAMA_RESEARCH_NUM_CTX", "8192")
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = OllamaJsonClient(
        host=args.host,
        model=args.model,
        think=True,
        num_ctx=args.num_ctx,
        timeout_seconds=args.timeout,
    )
    try:
        exit_code, package = run_pipeline(
            video_id=args.video_id,
            client=client,
            raw_root=args.raw_root,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Diagnostic package: {package}")
    if exit_code == 0:
        print("PASS: formula coverage is complete")
    else:
        print(
            "REVIEW REQUIRED: formula coverage contains unresolved items"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
