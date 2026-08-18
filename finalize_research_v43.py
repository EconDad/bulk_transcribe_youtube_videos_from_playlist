#!/usr/bin/env python3
"""Finalize one verified v4.3 diagnostic package into Processed Research."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from research_v43.finalization import (
    FINAL_PROMPT_VERSION,
    FinalizationError,
    build_citations_and_research,
    build_narrative_chunks,
    build_narrative_extraction_prompt,
    build_synthesis_prompt,
    merge_narrative_evidence,
    validate_diagnostic_for_finalization,
    verify_final_package,
    write_final_package,
)
from research_v43.narrative_recovery import recover_narrative_extraction
from research_v43.synthesis_recovery import recover_synthesis
from research_v43.model_client import ModelClientError, OllamaJsonClient
from run_research_v43 import load_transcript_source
from youtube_research_analysis import ResearchManifestStore, TranscriptSourcePackage
from youtube_research_io import canonical_youtube_url


NARRATIVE_SYSTEM_PROMPT = (
    "You extract and synthesize only source-grounded reader-facing research notes. "
    "Return strict JSON. Do not add outside facts, internal pipeline language, or "
    "machine identifiers."
)


def _source_url(video_id: str, metadata: Mapping[str, Any]) -> str:
    for key in ("canonical_url", "source_url", "webpage_url", "video_url", "url"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return canonical_youtube_url(video_id)


def finalize_video(
    *,
    video_id: str,
    raw_root: str | Path = "Raw Transcripts",
    diagnostic_root: str | Path = "Research v43 Diagnostics",
    processed_root: str | Path = "Processed Research",
    manifest_path: str | Path = "manifests/research.jsonl",
    client: Any,
    chunk_segments: int = 40,
    overlap_segments: int = 6,
    num_predict: int = 1536,
) -> Path:
    segments, source_sha, metadata = load_transcript_source(
        raw_root=raw_root,
        video_id=video_id,
    )
    diagnostic_dir = Path(diagnostic_root) / video_id
    diagnostic = validate_diagnostic_for_finalization(
        diagnostic_dir,
        source_package_sha256=source_sha,
    )
    formulas = diagnostic["formulas.json"].get("formulas") or []
    title = str(metadata.get("title") or video_id)

    source_package = TranscriptSourcePackage.load(raw_root, video_id)
    manifest = ResearchManifestStore(manifest_path)
    queued = manifest.queue(source_package)
    if queued.get("status") != "analyzing":
        manifest.transition(
            video_id=video_id,
            new_status="analyzing",
            title=title,
            url=_source_url(video_id, metadata),
            updates={
                "source_package_sha256": source_sha,
                "prompt_version": FINAL_PROMPT_VERSION,
                "analysis_backend": "ollama",
                "analysis_stage": "v4.3-stage-f.1",
            },
        )

    try:
        extracted: list[Mapping[str, Any]] = []
        invocations: list[dict[str, Any]] = []
        chunks = build_narrative_chunks(
            segments,
            chunk_segments=chunk_segments,
            overlap_segments=overlap_segments,
        )
        print(
            f"Narrative plan: {len(segments)} segments -> {len(chunks)} chunks",
            flush=True,
        )
        for position, (start, end) in enumerate(chunks, start=1):
            prompt = build_narrative_extraction_prompt(
                video_id=video_id,
                title=title,
                segments=segments,
                start_segment=start,
                end_segment=end,
            )
            stage = f"narrative_evidence chunk {position}/{len(chunks)} S{start}-S{end}"
            print(f"START {stage}", flush=True)
            response = client.complete_json(
                system_prompt=NARRATIVE_SYSTEM_PROMPT,
                user_prompt=prompt,
                stage=stage,
                num_predict=num_predict,
                think=False,
            )
            repairs: list[str] = []
            rejections: list[str] = []
            parsed = recover_narrative_extraction(
                response.payload,
                segments=segments,
                minimum_segment=start,
                maximum_segment=end,
                on_repair=repairs.append,
                on_reject=rejections.append,
            )
            for repair in repairs:
                print(f"REPAIR {stage}: {repair}", flush=True)
            for rejection in rejections:
                print(f"REJECT {stage}: {rejection}", flush=True)
            extracted.extend(parsed)
            invocations.append(response.invocation.to_dict())
            print(f"PASS {stage}: {len(parsed)} evidence item(s)", flush=True)

        evidence = merge_narrative_evidence(extracted)
        synthesis_prompt = build_synthesis_prompt(
            title=title,
            evidence=evidence,
            formulas=formulas,
        )
        last_error: Exception | None = None
        narrative = None
        for attempt in range(1, 4):
            stage = f"narrative_synthesis attempt {attempt}/3"
            print(f"START {stage}", flush=True)
            response = client.complete_json(
                system_prompt=NARRATIVE_SYSTEM_PROMPT,
                user_prompt=synthesis_prompt,
                stage=stage,
                num_predict=max(num_predict, 1900),
                think=False,
            )
            invocations.append(response.invocation.to_dict())
            repairs: list[str] = []
            try:
                narrative = recover_synthesis(
                    response.payload,
                    evidence=evidence,
                    segments=segments,
                    on_repair=repairs.append,
                )
            except FinalizationError as exc:
                last_error = exc
                print(f"RETRY {stage}: {exc}", flush=True)
                continue
            for repair in repairs:
                print(f"REPAIR {stage}: {repair}", flush=True)
            print(f"PASS {stage}", flush=True)
            break
        if narrative is None:
            raise FinalizationError(
                "Narrative synthesis failed deterministic validation three times: "
                f"{last_error}"
            )

        source_map, research, formulas_payload = build_citations_and_research(
            narrative=narrative,
            evidence=evidence,
            formulas=formulas,
            segments=segments,
        )
        result = write_final_package(
            output_root=processed_root,
            video_id=video_id,
            title=title,
            source_url=_source_url(video_id, metadata),
            source_package_sha256=source_sha,
            research=research,
            source_map=source_map,
            formulas_payload=formulas_payload,
            diagnostic_payloads=diagnostic,
            analysis_details={
                "provider": "ollama",
                "model": getattr(client, "model", ""),
                "think": False,
                "stage": "v4.3-stage-f.1",
                "narrative_evidence_count": len(evidence),
                "model_invocations": invocations,
            },
        )
        issues = verify_final_package(
            result.package_dir,
            source_package_sha256=source_sha,
        )
        if issues:
            raise FinalizationError(
                "Final package failed post-write verification: " + "; ".join(issues)
            )
        manifest.transition(
            video_id=video_id,
            new_status="research_ready",
            updates={
                "source_package_sha256": source_sha,
                "research_directory": str(result.package_dir),
                "research_package_sha256": result.package_sha256,
                "prompt_version": FINAL_PROMPT_VERSION,
                "analysis_backend": "ollama",
                "analysis_stage": "v4.3-stage-f.1",
            },
        )
        print(f"PASS: final research package written to {result.package_dir}")
        print(f"Package SHA-256: {result.package_sha256}")
        return result.package_dir
    except Exception as exc:
        current = manifest.get(video_id)
        if current and current.get("status") == "analyzing":
            manifest.transition(
                video_id=video_id,
                new_status="analysis_failed",
                error=f"{type(exc).__name__}: {exc}",
                updates={
                    "source_package_sha256": source_sha,
                    "prompt_version": FINAL_PROMPT_VERSION,
                    "analysis_stage": "v4.3-stage-f.1",
                },
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize one verified v4.3 diagnostic package into Processed Research."
    )
    parser.add_argument("video_id")
    parser.add_argument("--raw-root", default="Raw Transcripts")
    parser.add_argument("--diagnostic-root", default="Research v43 Diagnostics")
    parser.add_argument("--processed-root", default="Processed Research")
    parser.add_argument("--manifest", default="manifests/research.jsonl")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--num-predict", type=int, default=1536)
    parser.add_argument("--chunk-segments", type=int, default=40)
    parser.add_argument("--overlap-segments", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = OllamaJsonClient(
        host=args.host,
        model=args.model,
        think=False,
        num_ctx=args.num_ctx,
        timeout_seconds=args.timeout,
        num_predict=args.num_predict,
        keep_alive="30m",
    )
    try:
        finalize_video(
            video_id=args.video_id,
            raw_root=args.raw_root,
            diagnostic_root=args.diagnostic_root,
            processed_root=args.processed_root,
            manifest_path=args.manifest,
            client=client,
            chunk_segments=args.chunk_segments,
            overlap_segments=args.overlap_segments,
            num_predict=args.num_predict,
        )
    except (FinalizationError, ModelClientError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())