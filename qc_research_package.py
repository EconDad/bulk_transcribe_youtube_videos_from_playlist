from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from run_research_analysis import (
    MAX_CITATION_SEGMENTS,
    OllamaError,
    _text_similarity,
    _validate_formula_candidate,
    _validate_numeric_grounding,
)
from youtube_research_analysis import TranscriptSourcePackage


FORBIDDEN = (
    "evidence catalog",
    "json schema",
    "machine-readable relationship",
    "formula concept",
    "validation",
    "unavailable",
    "provided transcript",
    "provided data",
    "never output",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def citation_text(
    citation_ids: Sequence[str],
    citations: Mapping[str, Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
) -> str:
    parts: list[str] = []
    for citation_id in citation_ids:
        item = citations[citation_id]
        start = int(item["start_segment"])
        end = int(item["end_segment"])
        parts.extend(
            str(segments[index]["text"]).strip()
            for index in range(start, end + 1)
        )
    return " ".join(parts)


def prose_issues(
    text: str,
    *,
    support_text: str,
    context: str,
) -> list[str]:
    issues: list[str] = []
    stripped = text.strip()
    lowered = stripped.lower()

    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", stripped):
        issues.append("contains CJK/mixed-language text")
    if "_" in stripped:
        issues.append("contains machine-style variable names")
    for phrase in FORBIDDEN:
        if phrase in lowered:
            issues.append(f"contains internal wording: {phrase}")

    try:
        _validate_numeric_grounding(
            stripped,
            support_text,
            context=context,
        )
    except OllamaError as exc:
        issues.append(str(exc))

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concise human QC for one Phase 3 research package."
    )
    parser.add_argument("video_id")
    parser.add_argument("--raw-root", default="Raw Transcripts")
    parser.add_argument("--processed-root", default="Processed Research")
    args = parser.parse_args()

    source = TranscriptSourcePackage.load(
        args.raw_root,
        args.video_id,
    )
    processed = Path(args.processed_root) / args.video_id
    research = load_json(processed / "research.json")
    formulas = load_json(processed / "formulas.json")["formulas"]
    source_map = load_json(processed / "source_map.json")
    metadata = load_json(processed / "metadata.json")
    ready = load_json(processed / "_READY")

    citations = {
        item["citation_id"]: item
        for item in source_map["citations"]
    }
    errors: list[str] = []

    print("=" * 78)
    print("PHASE 3 QC VERDICT")
    print("=" * 78)
    print(f"Video:   {research.get('title') or args.video_id}")
    print(f"Prompt:  {metadata.get('prompt_version')}")
    print(f"Backend: {metadata.get('analysis_backend')}")
    print(f"SHA:     {ready.get('package_sha256')}")

    summary = str(research["executive_summary"]).strip()
    summary_support = citation_text(
        research["executive_summary_citation_ids"],
        citations,
        source.segments,
    )
    summary_issues = prose_issues(
        summary,
        support_text=summary_support,
        context="Executive summary",
    )

    print("\n1. EXECUTIVE SUMMARY")
    print("-" * 78)
    print(summary)
    print(
        "Verdict:",
        "PASS" if not summary_issues else "FAIL",
    )
    for issue in summary_issues:
        errors.append(f"Executive summary: {issue}")
        print(f"  - {issue}")

    print("\n2. TAKEAWAYS")
    print("-" * 78)
    prior_texts: list[str] = []
    for index, item in enumerate(
        research["key_takeaways"],
        start=1,
    ):
        value = str(item["text"]).strip()
        support = citation_text(
            item["citation_ids"],
            citations,
            source.segments,
        )
        issues = prose_issues(
            value,
            support_text=support,
            context=f"Takeaway {index}",
        )
        if any(
            _text_similarity(value, prior) >= 0.68
            for prior in prior_texts
        ):
            issues.append("duplicates an earlier takeaway")
        prior_texts.append(value)

        verdict = "PASS" if not issues else "FAIL"
        print(f"{index}. [{verdict}] {value}")
        print(f"   Sources: {', '.join(item['citation_ids'])}")
        for issue in issues:
            errors.append(f"Takeaway {index}: {issue}")
            print(f"   - {issue}")

    print("\n3. FORMULAS")
    print("-" * 78)
    if not formulas:
        print("No reusable symbolic formulas were extracted.")

    for index, formula in enumerate(formulas, start=1):
        formula_errors: list[str] = []
        steps = [
            str(step).strip()
            for step in formula.get("derivation_steps") or []
        ]
        description = next(
            (
                step.split(":", 1)[1].strip()
                for step in steps
                if step.startswith("Meaning:")
            ),
            str(formula.get("name") or "Formula"),
        )

        for citation_id in formula["citation_ids"]:
            citation = citations[citation_id]
            start = int(citation["start_segment"])
            end = int(citation["end_segment"])
            candidate = {
                **formula,
                "description": description,
                "derivation_steps": [
                    step
                    for step in steps
                    if not step.startswith("Meaning:")
                ]
                or [description],
                "start_segment": start,
                "end_segment": end,
            }
            try:
                _validate_formula_candidate(
                    candidate,
                    source=source,
                    context=(
                        f"Formula {index} citation {citation_id}"
                    ),
                    minimum_segment=0,
                    maximum_segment=len(source.segments) - 1,
                )
            except (OllamaError, KeyError, TypeError, ValueError) as exc:
                formula_errors.append(str(exc))

            excerpt = " ".join(
                str(source.segments[position]["text"]).strip()
                for position in range(start, end + 1)
            )
            print(
                f"   [{citation_id}] S{start:04d}-S{end:04d}: "
                f"{excerpt}"
            )

        if any(
            int(citations[cid]["end_segment"])
            - int(citations[cid]["start_segment"])
            + 1
            > MAX_CITATION_SEGMENTS
            for cid in formula["citation_ids"]
        ):
            formula_errors.append(
                "citation exceeds six transcript segments"
            )

        verdict = "PASS" if not formula_errors else "FAIL"
        print(
            f"{index}. [{verdict}] {formula['ascii']}"
        )
        print(f"   Meaning: {description}")
        for variable in formula.get("variables") or []:
            unit = str(variable.get("unit") or "unspecified")
            print(
                f"   - {variable.get('symbol')}: "
                f"{variable.get('meaning')} [{unit}]"
            )
        for issue in formula_errors:
            errors.append(
                f"Formula {index} ({formula.get('name')}): {issue}"
            )
            print(f"   - {issue}")
        print()

    print("4. CAVEATS")
    print("-" * 78)
    caveats = research.get("caveats") or []
    if caveats:
        print("FAIL: model-generated caveats should be empty in v3.6")
        for caveat in caveats:
            print(f"- {caveat}")
            errors.append(f"Unexpected caveat: {caveat}")
    else:
        print("PASS: no prompt/debug caveats were emitted")

    print("\n" + "=" * 78)
    if errors:
        print(f"OVERALL: FAIL ({len(errors)} issue(s))")
        for issue in errors:
            print(f"- {issue}")
        return 1

    print("OVERALL: PASS")
    print("The brief is readable, takeaways are distinct, and every formula is")
    print("entailed by its cited transcript window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
