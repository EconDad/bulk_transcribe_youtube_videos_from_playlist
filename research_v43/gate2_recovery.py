"""Generic bounded recoveries exposed by Gate 2 formula-heavy acceptance.

This module remains domain-neutral. It does not inject formulas or textbook
relationships. It adds three conservative repairs:

* downgrade numeric outcome/example inventory items when the bounded source
  still cannot ground a reusable arithmetic relationship;
* allow deterministic inventory expansion to use the full already-bounded
  audit neighborhood rather than only three adjacent segments; and
* replace model-paraphrased entailment quotes with exact source text from the
  model's own cited ranges, then rerun the strict validator.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import re
from typing import Any, Mapping, Sequence

from .calculation_inventory import CalculationInventory, CalculationItem
from .entailment import _OPERATION_CUES, _has_operation_cue
from .inventory_evidence_audit import (
    _variable_appears,
    find_deterministic_expansion as _find_deterministic_expansion,
    item_needs_evidence_audit,
)
from .operation_fragment_recovery import (
    audit_with_incomplete_operation_fragment_downgrade,
)
from .semantic_recovery import (
    validate_entailment_response_with_grounding_hull_repair,
)


_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[$€£]\s*)?[+-]?\d[\d,]*(?:\.\d+)?%?"
)
_RESULT_CUE_RE = re.compile(
    r"\b(?:amounts?\s+to|comes?\s+to|would\s+have|would\s+be|"
    r"is|are|equals?|gets?|gives?|yields?|makes?|difference|total|yield)\b",
    re.IGNORECASE,
)
_EXPLICIT_RESULT_CUE_RE = re.compile(
    r"\b(?:result(?:s)?(?:\s+(?:is|are|equals?))?|amounts?\s+to|"
    r"comes?\s+to|equals?|gets?|gives?|yields?|makes?|"
    r"difference(?:\s+of)?|total(?:\s+of)?)\b",
    re.IGNORECASE,
)
_EXAMPLE_CUE_RE = re.compile(
    r"\b(?:for example|example|suppose|let(?:'s| us) say|assume|scenario)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _segment_text(
    segments: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> str:
    return " ".join(
        str(segments[index].get("text") or "").strip()
        for index in range(start, end + 1)
    ).strip()


def _is_numeric_variable(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return False
    remainder = _NUMBER_RE.sub(" ", text)
    return not re.search(r"[A-Za-z]", remainder)


def _normalize_numeric_literal(value: str) -> str:
    return re.sub(r"[\s,$€£]", "", str(value)).casefold()


def _numeric_literals(value: str) -> set[str]:
    return {
        _normalize_numeric_literal(match)
        for match in _NUMBER_RE.findall(str(value))
    }


def _canonical_operations(item: CalculationItem) -> tuple[str, ...]:
    return tuple(
        operation
        for operation in item.operations_mentioned
        if operation in _OPERATION_CUES
    )


def _compact_numeric_example_without_joint_operation(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
    lookback_segments: int = 8,
    lookahead_segments: int = 2,
    max_joint_window: int = 3,
) -> bool:
    """Detect a numeric instance whose specific operands are not jointly stated.

    This is intentionally stricter than merely seeing an operation somewhere
    nearby. It requires all inventory variables to be numeric literals and asks
    whether those exact literals plus every canonical operation can be grounded
    in one short contiguous source window. A window containing competing numeric
    operands is not accepted merely because the target literals happen to occur
    nearby; at most one extra numeric value is allowed, and only when the same
    window explicitly presents it as a result. If the inventory's numeric
    operands do not appear anywhere in the bounded context, the event is left
    to the more general outcome-only classifier rather than being called a
    worked instance.
    """

    if not item.variables_mentioned or not all(
        _is_numeric_variable(value)
        for value in item.variables_mentioned
    ):
        return False

    operations = _canonical_operations(item)
    if not operations:
        return False

    context_start = max(0, item.start_segment - lookback_segments)
    context_end = min(
        len(segments) - 1,
        item.end_segment + lookahead_segments,
    )
    context_text = _segment_text(segments, context_start, context_end)
    if _EXAMPLE_CUE_RE.search(context_text) is None:
        return False

    target_numbers: set[str] = set()
    for variable in item.variables_mentioned:
        target_numbers.update(_numeric_literals(variable))

    context_numbers = _numeric_literals(context_text)
    if not target_numbers or not target_numbers.issubset(context_numbers):
        return False

    for start in range(context_start, context_end + 1):
        for end in range(
            start,
            min(context_end, start + max_joint_window - 1) + 1,
        ):
            text = _segment_text(segments, start, end)
            if not all(
                _variable_appears(variable, text)
                for variable in item.variables_mentioned
            ):
                continue
            if not all(
                _has_operation_cue(operation, text)
                for operation in operations
            ):
                continue

            window_numbers = _numeric_literals(text)
            extra_numbers = window_numbers - target_numbers
            if not extra_numbers:
                return False

            # One extra number may be the explicitly stated result of the
            # operation. Generic copulas such as "price is 800" are not enough
            # to prove that the extra value is the result of this arithmetic.
            if (
                len(extra_numbers) == 1
                and _EXPLICIT_RESULT_CUE_RE.search(text) is not None
            ):
                return False

    item_text = _segment_text(
        segments,
        item.start_segment,
        item.end_segment,
    )
    return bool(_NUMBER_RE.search(item_text) and _RESULT_CUE_RE.search(item_text))


def _outcome_without_complete_reusable_support(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
    lookback_segments: int = 8,
    lookahead_segments: int = 2,
) -> bool:
    """Detect a quantitative outcome whose inventory claims stay ungrounded.

    The original span must state a numeric result/outcome. We then test the same
    inventory claims against the full bounded neighborhood. If they still do
    not ground, the source does not support the inventory's claimed reusable
    symbolic relationship and the event is downgraded rather than left
    unresolved.
    """

    item_text = _segment_text(
        segments,
        item.start_segment,
        item.end_segment,
    )
    if _NUMBER_RE.search(item_text) is None:
        return False
    if _RESULT_CUE_RE.search(item_text) is None:
        return False

    context_start = max(0, item.start_segment - lookback_segments)
    context_end = min(
        len(segments) - 1,
        item.end_segment + lookahead_segments,
    )
    expanded = replace(
        item,
        start_segment=context_start,
        end_segment=context_end,
    )
    still_needs_audit, _ = item_needs_evidence_audit(
        item=expanded,
        segments=segments,
    )
    return still_needs_audit


def audit_with_gate2_semantic_downgrades(
    *,
    inventory: CalculationInventory,
    segments: Sequence[Mapping[str, Any]],
) -> tuple[CalculationInventory, tuple[dict[str, Any], ...]]:
    """Layer Gate 2 semantic downgrades over accepted Stage F.1 recovery."""

    audited, existing_records = (
        audit_with_incomplete_operation_fragment_downgrade(
            inventory=inventory,
            segments=segments,
        )
    )

    items: list[CalculationItem] = []
    records: list[dict[str, Any]] = list(existing_records)

    for item in audited.calculations:
        if not item.formula_expected or item.visual_equation_cue:
            items.append(item)
            continue

        action: str | None = None
        reason: str | None = None

        if _compact_numeric_example_without_joint_operation(
            item=item,
            segments=segments,
        ):
            action = "downgrade_non_symbolic_numeric_instance"
            reason = (
                "Deterministic Gate 2 pre-audit found a worked numeric instance "
                "whose specific numeric operands and arithmetic operation are "
                "not jointly stated in a compact bounded source window."
            )
        elif _outcome_without_complete_reusable_support(
            item=item,
            segments=segments,
        ):
            action = "downgrade_non_symbolic_outcome_only"
            reason = (
                "Deterministic Gate 2 pre-audit found a quantitative outcome, "
                "comparison, or reported result whose full bounded source still "
                "does not ground the inventory's claimed reusable arithmetic "
                "relationship."
            )

        if action is None or reason is None:
            items.append(item)
            continue

        updated = replace(
            item,
            formula_expected=False,
            reason=item.reason + " " + reason,
        )
        items.append(updated)
        records.append(
            {
                "calculation_id": item.calculation_id,
                "action": action,
                "decision_source": "deterministic_gate2",
                "before": {
                    "formula_expected": item.formula_expected,
                    "start_segment": item.start_segment,
                    "end_segment": item.end_segment,
                    "variables_mentioned": list(item.variables_mentioned),
                    "operations_mentioned": list(item.operations_mentioned),
                },
                "after": {
                    "formula_expected": False,
                    "start_segment": updated.start_segment,
                    "end_segment": updated.end_segment,
                    "variables_mentioned": list(updated.variables_mentioned),
                    "operations_mentioned": list(updated.operations_mentioned),
                },
                "reason": reason,
            }
        )

    return (
        CalculationInventory(
            schema_version=audited.schema_version,
            video_id=audited.video_id,
            calculations=tuple(items),
        ),
        tuple(records),
    )


def find_deterministic_expansion_gate2(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
    neighborhood_start: int,
    neighborhood_end: int,
    max_auto_distance: int = 3,
):
    """Use the already-bounded audit neighborhood for deterministic expansion."""

    bounded_distance = max(
        max_auto_distance,
        item.start_segment - neighborhood_start,
        neighborhood_end - item.end_segment,
    )
    return _find_deterministic_expansion(
        item=item,
        segments=segments,
        neighborhood_start=neighborhood_start,
        neighborhood_end=neighborhood_end,
        max_auto_distance=bounded_distance,
    )


def _normalized_words(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(str(value))
        if len(token) >= 2
    }


def _numeric_tokens(value: str) -> set[str]:
    return {
        re.sub(r"[\s,$€£]", "", token)
        for token in _NUMBER_RE.findall(str(value))
    }


def _quote_has_source_overlap(quote: str, source: str) -> bool:
    quote_words = _normalized_words(quote)
    source_words = _normalized_words(source)
    word_overlap = len(quote_words & source_words)
    if quote_words and word_overlap >= min(3, len(quote_words)):
        return True

    quote_numbers = _numeric_tokens(quote)
    source_numbers = _numeric_tokens(source)
    return bool(quote_numbers & source_numbers)


def _repair_quotes_to_cited_source(
    payload: Mapping[str, Any],
    *,
    segments: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    repaired = copy.deepcopy(dict(payload))
    raw_nodes = repaired.get("nodes")
    if isinstance(raw_nodes, (str, bytes)) or not isinstance(raw_nodes, Sequence):
        return repaired, False

    changed = False
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping):
            continue
        for field in ("evidence", "identifier_groundings"):
            records = raw_node.get(field)
            if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                start = record.get("start_segment")
                end = record.get("end_segment")
                quote = record.get("quote")
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, int)
                    or not isinstance(end, int)
                    or not isinstance(quote, str)
                    or start < 0
                    or end < start
                    or end >= len(segments)
                ):
                    continue
                source = _segment_text(segments, start, end)
                normalized_quote = re.sub(r"\s+", " ", quote.casefold()).strip()
                normalized_source = re.sub(r"\s+", " ", source.casefold()).strip()
                if normalized_quote and normalized_quote in normalized_source:
                    continue
                if not _quote_has_source_overlap(quote, source):
                    continue
                record["quote"] = source
                changed = True

    return repaired, changed


def validate_entailment_response_with_gate2_quote_repair(
    payload: Mapping[str, Any],
    *,
    item: CalculationItem,
    candidate: Any,
    segments: Sequence[Mapping[str, Any]],
):
    """Localize paraphrased quotes to exact text inside unchanged cited ranges."""

    report = validate_entailment_response_with_grounding_hull_repair(
        payload,
        item=item,
        candidate=candidate,
        segments=segments,
    )
    if report.passed:
        return report

    quote_issues = tuple(
        issue
        for issue in report.issues
        if (
            "quote is not present in cited segments" in issue
            or "grounding quote for" in issue
            and "is not present in cited segments" in issue
        )
    )
    if not quote_issues:
        return report
    if len(quote_issues) != len(report.issues):
        return report

    repaired, changed = _repair_quotes_to_cited_source(
        payload,
        segments=segments,
    )
    if not changed:
        return report

    return validate_entailment_response_with_grounding_hull_repair(
        repaired,
        item=item,
        candidate=candidate,
        segments=segments,
    )
