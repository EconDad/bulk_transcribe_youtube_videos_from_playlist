"""Bounded finalization recovery for long formula source ranges.

Formula extraction may legitimately retain a source claim whose full grounded
range is wider than the reader-facing citation limit. Finalization should not
weaken that source claim merely to satisfy presentation constraints. Instead,
this module creates citation-sized windows for the source map while restoring
the original formula source claims in the emitted research artifacts.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .finalization import (
    MAX_CITATION_SEGMENTS,
    NarrativeEvidence,
    build_citations_and_research,
)


def _citation_claim_windows(
    claim: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Split one valid-looking claim into consecutive citation-sized windows.

    Invalid or reversed ranges are deliberately left unchanged so the frozen
    finalizer retains responsibility for rejecting them.
    """

    normalized = dict(claim)
    try:
        start = int(claim.get("start_segment", -1))
        end = int(claim.get("end_segment", start))
    except (TypeError, ValueError):
        return (normalized,)

    if end < start or end - start + 1 <= MAX_CITATION_SEGMENTS:
        return (normalized,)

    windows: list[dict[str, Any]] = []
    cursor = start
    while cursor <= end:
        window_end = min(end, cursor + MAX_CITATION_SEGMENTS - 1)
        window = dict(claim)
        window["start_segment"] = cursor
        window["end_segment"] = window_end
        windows.append(window)
        cursor = window_end + 1
    return tuple(windows)


def build_citations_and_research_with_formula_claim_splitting(
    *,
    narrative: Mapping[str, Any],
    evidence: Sequence[NarrativeEvidence],
    formulas: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build final artifacts with bounded citations and intact source claims."""

    working_formulas: list[dict[str, Any]] = []
    original_claims: list[Any] = []

    for formula in formulas:
        working = dict(formula)
        claims = formula.get("source_claims") or []
        original_claims.append(copy.deepcopy(claims))

        split_claims: list[Any] = []
        if isinstance(claims, Sequence) and not isinstance(claims, (str, bytes)):
            for claim in claims:
                if isinstance(claim, Mapping):
                    split_claims.extend(_citation_claim_windows(claim))
                else:
                    split_claims.append(claim)
        else:
            split_claims = copy.deepcopy(claims)

        working["source_claims"] = split_claims
        working_formulas.append(working)

    source_map, research, formulas_payload = build_citations_and_research(
        narrative=narrative,
        evidence=evidence,
        formulas=working_formulas,
        segments=segments,
    )

    normalized_formulas = formulas_payload.get("formulas") or []
    if len(normalized_formulas) != len(original_claims):
        raise RuntimeError("Finalizer changed formula count during citation recovery")

    restored_formulas: list[dict[str, Any]] = []
    for normalized, claims in zip(normalized_formulas, original_claims):
        restored = dict(normalized)
        restored["source_claims"] = copy.deepcopy(claims)
        restored_formulas.append(restored)

    restored_research = dict(research)
    restored_research["formulas"] = restored_formulas

    restored_payload = dict(formulas_payload)
    restored_payload["formulas"] = restored_formulas

    return source_map, restored_research, restored_payload
