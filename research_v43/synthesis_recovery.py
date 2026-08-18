"""Conservative deterministic repair for Stage F narrative synthesis.

The synthesis model occasionally returns otherwise grounded prose with too many
validated evidence IDs or sections in the wrong transcript order. This module
never rewrites prose, invents evidence, expands the model's citation set, or
weakens strict validation. It may only:

* deduplicate/prune already-selected evidence IDs to the schema maximum when a
  retained subset still grounds every numeric value in the prose; and
* stably reorder section objects by the transcript position of their own
  validated evidence IDs.

The repaired payload is always passed through the original strict validator.
"""

from __future__ import annotations

import copy
from itertools import combinations
import re
from typing import Any, Callable, Mapping, Sequence

from .finalization import (
    FinalizationError,
    NarrativeEvidence,
    _numeric_tokens,
    _segment_text,
    validate_synthesis,
)


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been",
    "being", "by", "for", "from", "has", "have", "he", "her", "his",
    "i", "in", "is", "it", "its", "of", "on", "or", "she", "that",
    "the", "their", "them", "they", "this", "to", "was", "we", "were",
    "what", "when", "which", "who", "why", "with", "you", "your",
}


def _words(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(token) >= 2 and token not in _STOP_WORDS
    }


def _unique_ids(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _support_text(
    ids: Sequence[str],
    *,
    by_id: Mapping[str, NarrativeEvidence],
    segments: Sequence[Mapping[str, Any]],
) -> str:
    return " ".join(
        _segment_text(
            segments,
            by_id[evidence_id].start_segment,
            by_id[evidence_id].end_segment,
        )
        for evidence_id in ids
    )


def _reader_support_text(
    ids: Sequence[str],
    *,
    by_id: Mapping[str, NarrativeEvidence],
) -> str:
    return " ".join(
        " ".join(
            (
                by_id[evidence_id].topic,
                by_id[evidence_id].text,
                by_id[evidence_id].explanation,
            )
        )
        for evidence_id in ids
    )


def _best_grounded_subset(
    values: Any,
    *,
    text: str,
    maximum: int,
    context: str,
    by_id: Mapping[str, NarrativeEvidence],
    segments: Sequence[Mapping[str, Any]],
) -> tuple[list[str], str | None]:
    """Return a bounded subset of the model's own IDs without losing numbers."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise FinalizationError(f"{context} evidence_ids must be an array")

    ids = _unique_ids(values)
    if not ids:
        raise FinalizationError(f"{context} has invalid evidence count")

    unknown = [value for value in ids if value not in by_id]
    if unknown:
        raise FinalizationError(
            f"{context} references unknown evidence: {unknown}"
        )

    if len(ids) <= maximum:
        if len(ids) != len(values):
            return ids, f"{context} deduplicated evidence IDs"
        return ids, None

    claimed_numbers = _numeric_tokens(text)
    text_words = _words(text)
    best: tuple[tuple[int, float, int, tuple[int, ...]], list[str]] | None = None

    # Search only subsets of the model's own citations. Prefer stronger lexical
    # support and, on ties, more retained citations and earlier original IDs.
    for size in range(1, maximum + 1):
        for indexes in combinations(range(len(ids)), size):
            subset = [ids[index] for index in indexes]
            supported_numbers = _numeric_tokens(
                _support_text(
                    subset,
                    by_id=by_id,
                    segments=segments,
                )
            )
            if not claimed_numbers.issubset(supported_numbers):
                continue

            support_words = _words(
                _reader_support_text(subset, by_id=by_id)
            )
            overlap = len(text_words & support_words)
            coverage = (
                overlap / len(text_words)
                if text_words
                else 1.0
            )
            score = (
                overlap,
                coverage,
                size,
                tuple(-index for index in indexes),
            )
            if best is None or score > best[0]:
                best = (score, subset)

    if best is None:
        raise FinalizationError(
            f"{context} has too many evidence IDs and no bounded subset "
            "preserves numeric grounding"
        )

    subset = best[1]
    return (
        subset,
        f"{context} pruned evidence IDs {ids} -> {subset}",
    )


def recover_synthesis(
    payload: Mapping[str, Any],
    *,
    evidence: Sequence[NarrativeEvidence],
    segments: Sequence[Mapping[str, Any]],
    on_repair: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Repair only bounded citation-count/order defects, then validate strictly."""

    try:
        return validate_synthesis(
            payload,
            evidence=evidence,
            segments=segments,
        )
    except FinalizationError:
        pass

    required = {
        "executive_summary",
        "executive_summary_evidence_ids",
        "key_takeaways",
        "sections",
    }
    if set(payload) != required:
        return validate_synthesis(
            payload,
            evidence=evidence,
            segments=segments,
        )

    by_id = {item.evidence_id: item for item in evidence}
    repaired = copy.deepcopy(dict(payload))
    repair_messages: list[str] = []

    summary = repaired.get("executive_summary")
    if isinstance(summary, str):
        ids, message = _best_grounded_subset(
            repaired.get("executive_summary_evidence_ids"),
            text=summary,
            maximum=6,
            context="executive_summary",
            by_id=by_id,
            segments=segments,
        )
        repaired["executive_summary_evidence_ids"] = ids
        if message:
            repair_messages.append(message)

    raw_takeaways = repaired.get("key_takeaways")
    if (
        not isinstance(raw_takeaways, (str, bytes))
        and isinstance(raw_takeaways, Sequence)
    ):
        for index, item in enumerate(raw_takeaways):
            if not isinstance(item, Mapping) or set(item) != {"text", "evidence_ids"}:
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            ids, message = _best_grounded_subset(
                item.get("evidence_ids"),
                text=text,
                maximum=2,
                context=f"key_takeaways[{index}]",
                by_id=by_id,
                segments=segments,
            )
            item["evidence_ids"] = ids
            if message:
                repair_messages.append(message)

    raw_sections = repaired.get("sections")
    if (
        not isinstance(raw_sections, (str, bytes))
        and isinstance(raw_sections, Sequence)
    ):
        sortable = True
        section_starts: list[int] = []
        for index, item in enumerate(raw_sections):
            if (
                not isinstance(item, Mapping)
                or set(item) != {"heading", "summary", "evidence_ids"}
            ):
                sortable = False
                continue
            summary_text = item.get("summary")
            if not isinstance(summary_text, str):
                sortable = False
                continue
            ids, message = _best_grounded_subset(
                item.get("evidence_ids"),
                text=summary_text,
                maximum=3,
                context=f"sections[{index}]",
                by_id=by_id,
                segments=segments,
            )
            item["evidence_ids"] = ids
            if message:
                repair_messages.append(message)
            if not ids:
                sortable = False
                continue
            section_starts.append(
                min(by_id[value].start_segment for value in ids)
            )

        if sortable and len(section_starts) == len(raw_sections):
            order = sorted(
                range(len(raw_sections)),
                key=lambda index: (section_starts[index], index),
            )
            if order != list(range(len(raw_sections))):
                repaired["sections"] = [
                    raw_sections[index]
                    for index in order
                ]
                repair_messages.append(
                    "sections reordered by transcript progression: "
                    + " -> ".join(str(index) for index in order)
                )

    validated = validate_synthesis(
        repaired,
        evidence=evidence,
        segments=segments,
    )
    if on_repair is not None:
        for message in repair_messages:
            on_repair(message)
    return validated
