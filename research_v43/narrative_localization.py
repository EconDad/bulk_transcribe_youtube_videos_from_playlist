"""Deterministic citation localization for Stage F narrative evidence.

The model may propose a broad bounded range. This module never expands that
range. When the proposal exceeds the six-segment reader citation limit, it
selects a smaller source-grounded subwindow using only lexical and numeric
support, then delegates to the strict Stage F parser for final validation.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from .finalization import (
    FinalizationError,
    MAX_CITATION_SEGMENTS,
    parse_narrative_extraction,
)


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _segment_text(
    segments: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> str:
    return " ".join(
        str(segments[index].get("text") or "").strip()
        for index in range(start, end + 1)
    ).strip()


def _words(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(word) >= 2 and word not in _STOP_WORDS
    ]


def _numeric_tokens(text: str) -> set[str]:
    return {
        token.replace(",", "")
        for token in re.findall(
            r"(?<![A-Za-z_])\d[\d,]*(?:\.\d+)?%?",
            str(text),
        )
    }


def _validate_proposed_bounds(
    *,
    start: int,
    end: int,
    segment_count: int,
    minimum_segment: int,
    maximum_segment: int,
    context: str,
) -> None:
    if (
        start < minimum_segment
        or end < start
        or end > maximum_segment
        or end >= segment_count
    ):
        raise FinalizationError(
            f"{context} has invalid segment range {start}-{end}"
        )


def _best_supported_window(
    *,
    claim_text: str,
    segments: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
    max_segments: int = MAX_CITATION_SEGMENTS,
) -> tuple[int, int]:
    query_words = _words(claim_text)
    query_set = set(query_words)
    if not query_set:
        raise FinalizationError(
            "Cannot localize narrative citation because the evidence item "
            "has no content-bearing words"
        )

    claimed_numbers = _numeric_tokens(claim_text)
    query_bigrams = {
        (query_words[index], query_words[index + 1])
        for index in range(len(query_words) - 1)
    }

    best: tuple[float, int, int, int] | None = None
    best_overlap = 0
    best_coverage = 0.0

    maximum_length = min(max_segments, end - start + 1)
    for length in range(1, maximum_length + 1):
        last_start = end - length + 1
        for candidate_start in range(start, last_start + 1):
            candidate_end = candidate_start + length - 1
            source_text = _segment_text(
                segments,
                candidate_start,
                candidate_end,
            )
            supported_numbers = _numeric_tokens(source_text)
            if not claimed_numbers.issubset(supported_numbers):
                continue

            source_words = _words(source_text)
            source_set = set(source_words)
            overlap = query_set & source_set
            if not overlap:
                continue

            overlap_count = len(overlap)
            coverage = overlap_count / len(query_set)
            source_bigrams = {
                (source_words[index], source_words[index + 1])
                for index in range(len(source_words) - 1)
            }
            bigram_matches = len(query_bigrams & source_bigrams)

            score = (
                coverage * 100.0
                + overlap_count * 8.0
                + bigram_matches * 6.0
                - length * 0.75
            )
            candidate = (
                score,
                -length,
                -candidate_start,
                candidate_end,
            )
            if best is None or candidate > best:
                best = candidate
                best_overlap = overlap_count
                best_coverage = coverage

    if best is None:
        raise FinalizationError(
            "Unable to localize broad narrative citation to a source-grounded "
            f"window inside segments {start}-{end}"
        )

    _, negative_length, negative_start, candidate_end = best
    candidate_start = -negative_start
    window_length = -negative_length

    if best_overlap < 2 and not (
        best_overlap == 1 and len(query_set) <= 2
    ):
        raise FinalizationError(
            "Best localized narrative citation has insufficient lexical "
            f"support ({best_overlap} matched content words)"
        )
    if best_coverage < 0.12:
        raise FinalizationError(
            "Best localized narrative citation covers too little of the "
            f"evidence item ({best_coverage:.1%})"
        )
    if window_length > max_segments:
        raise AssertionError("Narrative localization exceeded citation limit")

    return candidate_start, candidate_end


def localize_narrative_extraction(
    payload: Mapping[str, Any],
    *,
    segments: Sequence[Mapping[str, Any]],
    minimum_segment: int,
    maximum_segment: int,
    on_repair: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Localize broad evidence ranges, then run strict Stage F validation.

    Narrow model ranges are not changed. A broad range can only shrink within
    its own proposed bounds. If no narrow window supports both the prose and
    every claimed numeric token, finalization fails closed.
    """

    if set(payload) != {"evidence"}:
        return parse_narrative_extraction(
            payload,
            segments=segments,
            minimum_segment=minimum_segment,
            maximum_segment=maximum_segment,
        )

    raw_items = payload.get("evidence")
    if isinstance(raw_items, (str, bytes)) or not isinstance(raw_items, Sequence):
        return parse_narrative_extraction(
            payload,
            segments=segments,
            minimum_segment=minimum_segment,
            maximum_segment=maximum_segment,
        )

    repaired_items: list[Any] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            repaired_items.append(raw)
            continue

        required = {
            "topic",
            "text",
            "explanation",
            "start_segment",
            "end_segment",
        }
        if set(raw) != required:
            repaired_items.append(dict(raw))
            continue

        try:
            start = int(raw["start_segment"])
            end = int(raw["end_segment"])
        except (TypeError, ValueError):
            repaired_items.append(dict(raw))
            continue

        _validate_proposed_bounds(
            start=start,
            end=end,
            segment_count=len(segments),
            minimum_segment=minimum_segment,
            maximum_segment=maximum_segment,
            context=f"evidence[{index}]",
        )

        item = dict(raw)
        if end - start + 1 > MAX_CITATION_SEGMENTS:
            claim_text = " ".join(
                str(raw.get(field) or "")
                for field in ("topic", "text", "explanation")
            )
            localized_start, localized_end = _best_supported_window(
                claim_text=claim_text,
                segments=segments,
                start=start,
                end=end,
            )
            item["start_segment"] = localized_start
            item["end_segment"] = localized_end
            if on_repair is not None:
                on_repair(
                    f"evidence[{index}] citation {start}-{end} -> "
                    f"{localized_start}-{localized_end}"
                )
        repaired_items.append(item)

    return parse_narrative_extraction(
        {"evidence": repaired_items},
        segments=segments,
        minimum_segment=minimum_segment,
        maximum_segment=maximum_segment,
    )
