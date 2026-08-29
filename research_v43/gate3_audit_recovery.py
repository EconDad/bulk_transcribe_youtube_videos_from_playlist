"""Strict Gate 3 inventory-audit fallback with durable error handling."""

from __future__ import annotations

import json
import re
from typing import Mapping, Sequence

from .calculation_inventory import CalculationItem
from .gate3_recovery import (
    _canonical_operation,
    _gate3_has_operation_cue,
    _gate3_variable_appears,
    _normalized_audit_payload,
    _valid_evidence_ids,
)
from .inventory_evidence_audit import (
    AuditAction,
    InventoryAuditDecision,
    InventoryEvidenceAuditError,
    _range_text,
    parse_inventory_evidence_audit_response,
)


_ADJACENT_OPERAND_EXTENSION = 4
_LOCAL_ASSOCIATION_WINDOW = 4
_RESULT_CUE_RE = re.compile(
    r"\b(?:amounts?\s+to|comes?\s+to|goes?\s+up\s+to|"
    r"rose\s+to|risen\s+to|would\s+have|"
    r"would\s+be|equals?|gets?|gives?|yields?|makes?|made|"
    r"difference|total|return|result)\b",
    re.IGNORECASE,
)
_ARITHMETIC_BOUNDARY_RE = re.compile(
    r"\b(?:add(?:ed|ing)?|plus|subtract(?:ed|ing)?|minus|"
    r"multipl(?:y|ied|ying)|times|divid(?:e|ed|ing))\b",
    re.IGNORECASE,
)


def _numeric_only_claim(value: str) -> bool:
    """Return whether a claim is a numeric literal with optional unit words."""

    from .inventory_evidence_audit import _numeric_signatures

    signatures = _numeric_signatures(value)
    if len(signatures) != 1:
        return False
    remainder = re.sub(
        r"(?<![a-z0-9.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
        r"\s*(?:%|percent\b)?",
        " ",
        value.casefold(),
    )
    tokens = set(re.findall(r"[a-z]+", remainder))
    return tokens.issubset(
        {
            "dollar",
            "dollars",
            "percent",
            "percentage",
            "thousand",
            "million",
            "billion",
            "trillion",
        }
    )


def _numeric_claim_is_reported_result(
    variable: str,
    *,
    segments: Sequence[Mapping[str, object]],
    start: int,
    end: int,
) -> bool:
    """Recognize a claimed numeric operand presented as an outcome instead."""

    if not _numeric_only_claim(variable):
        return False
    for index in range(start, end + 1):
        text = _range_text(segments, index, index)
        for cue in _RESULT_CUE_RE.finditer(text):
            scope = text[cue.start():]
            if index < end:
                scope += " " + _range_text(segments, index + 1, index + 1)
            boundary = _ARITHMETIC_BOUNDARY_RE.search(
                scope,
                cue.end() - cue.start(),
            )
            if boundary is not None:
                scope = scope[:boundary.start()]
            if _gate3_variable_appears(variable, scope):
                return True
    return False


def _claims_use_reported_result_as_operand(
    *,
    variables: Sequence[str],
    operations: Sequence[str],
    segments: Sequence[Mapping[str, object]],
    start: int,
    end: int,
) -> bool:
    """Reject a binary numeric claim that substitutes an outcome for an input."""

    if len(variables) != 2 or len(operations) != 1:
        return False
    if operations[0] not in {"addition", "subtraction", "multiplication", "division"}:
        return False
    if not all(_numeric_only_claim(variable) for variable in variables):
        return False
    return all(
        _numeric_claim_is_reported_result(
            variable,
            segments=segments,
            start=start,
            end=end,
        )
        for variable in variables
    )


def _operation_has_compact_claim_association(
    *,
    variables: Sequence[str],
    operations: Sequence[str],
    segments: Sequence[Mapping[str, object]],
    start: int,
    end: int,
) -> bool:
    """Require an adjacent operation guard to bind the unchanged claims."""

    for window_start in range(start, end + 1):
        window_end = min(end, window_start + _LOCAL_ASSOCIATION_WINDOW - 1)
        text = _range_text(segments, window_start, window_end)
        if not all(
            _gate3_variable_appears(variable, text)
            for variable in variables
        ):
            continue
        if all(
            _gate3_has_operation_cue(operation, text)
            for operation in operations
        ):
            return True
    return False


def _grounded_original_variables(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, object]],
    start: int,
    end: int,
) -> tuple[str, ...]:
    """Return source-extractive original claims, excluding synthetic aliases."""

    text = _range_text(segments, start, end)
    normalized_text = re.sub(r"\s+", " ", text.casefold()).strip()
    grounded: list[str] = []
    for variable in item.variables_mentioned:
        if not _gate3_variable_appears(variable, text):
            continue
        if re.fullmatch(r"[a-z][a-z0-9_]*", variable) and "_" in variable:
            spoken = variable.replace("_", " ")
            if spoken not in normalized_text:
                continue
        if variable not in grounded:
            grounded.append(variable)
    return tuple(grounded)


def _find_minimal_grounded_span(
    *,
    item: CalculationItem,
    evidence_ids: Sequence[int],
    variables: Sequence[str],
    operations: Sequence[str],
    segments: Sequence[Mapping[str, object]],
    neighborhood_start: int,
    neighborhood_end: int,
) -> tuple[int, int] | None:
    """Find the smallest supplied hull that grounds unchanged revised claims."""

    required_start = min(item.start_segment, *evidence_ids)
    required_end = max(item.end_segment, *evidence_ids)

    candidates: list[tuple[int, int, int, int]] = []
    for start in range(neighborhood_start, required_start + 1):
        for end in range(required_end, neighborhood_end + 1):
            text = _range_text(segments, start, end)
            if not all(
                _gate3_variable_appears(variable, text)
                for variable in variables
            ):
                continue
            if not all(
                _gate3_has_operation_cue(operation, text)
                for operation in operations
            ):
                continue
            width = end - start + 1
            added = (required_start - start) + (end - required_end)
            candidates.append((added, width, start, end))

    if not candidates:
        return None

    _, _, start, end = min(candidates)
    return start, end


def _adjacent_bounds(
    *,
    segment_count: int,
    neighborhood_start: int,
    neighborhood_end: int,
) -> tuple[int, int]:
    """Return the fixed small guard/operand extension around an audit window."""

    return (
        max(0, neighborhood_start - _ADJACENT_OPERAND_EXTENSION),
        min(segment_count - 1, neighborhood_end + _ADJACENT_OPERAND_EXTENSION),
    )


def parse_inventory_evidence_audit_response_with_gate3_repair(
    response_text: str,
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, object]],
    neighborhood_start: int,
    neighborhood_end: int,
) -> InventoryAuditDecision:
    """Resolve only mechanically grounded Gate 3 audit failures.

    A failed reconcile is accepted only when every revised variable and revised
    operation is grounded in a contiguous source span under the generic Gate 3
    lexical rules. The model-selected span is tried first, followed by the
    original bounded audit neighborhood. If the selected span itself already
    contains every claimed operation cue, Python may make one final, tightly
    bounded adjacent search of at most four segments on either side solely to
    locate the unchanged named operands. This handles result-after-list source
    structure without inventing an operation or changing the calculation target.

    Otherwise the item is downgraded only when the claimed operation is absent
    from the entire original audit neighborhood *and* the same fixed adjacent
    guard contains no such operation cue. An adjacent cue is never borrowed to
    validate the calculation, but its presence prevents an unsupported
    non-symbolic downgrade when the audit window itself likely clipped the
    operation evidence.
    """

    saved_error: InventoryEvidenceAuditError | None = None
    try:
        return parse_inventory_evidence_audit_response(
            response_text,
            item=item,
            segments=segments,
            neighborhood_start=neighborhood_start,
            neighborhood_end=neighborhood_end,
        )
    except InventoryEvidenceAuditError as exc:
        saved_error = exc

    normalized = _normalized_audit_payload(response_text)
    if normalized is None:
        assert saved_error is not None
        raise saved_error

    try:
        return parse_inventory_evidence_audit_response(
            json.dumps(normalized),
            item=item,
            segments=segments,
            neighborhood_start=neighborhood_start,
            neighborhood_end=neighborhood_end,
        )
    except InventoryEvidenceAuditError:
        pass

    action = (
        str(normalized.get("action") or "")
        .strip()
        .casefold()
        .replace("-", "_")
    )
    if action != "reconcile":
        assert saved_error is not None
        raise saved_error

    evidence_ids = _valid_evidence_ids(
        normalized,
        neighborhood_start=neighborhood_start,
        neighborhood_end=neighborhood_end,
    )
    if evidence_ids is None:
        assert saved_error is not None
        raise saved_error

    raw_variables = normalized.get("revised_variables_mentioned")
    raw_operations = normalized.get("revised_operations_mentioned")
    if (
        isinstance(raw_variables, (str, bytes))
        or not isinstance(raw_variables, Sequence)
        or isinstance(raw_operations, (str, bytes))
        or not isinstance(raw_operations, Sequence)
        or not raw_variables
        or not raw_operations
        or not all(
            isinstance(value, str) and value.strip()
            for value in raw_variables
        )
        or not all(
            isinstance(value, str) and value.strip()
            for value in raw_operations
        )
    ):
        assert saved_error is not None
        raise saved_error

    variables = tuple(str(value).strip() for value in raw_variables)
    operations = tuple(
        _canonical_operation(str(value))
        for value in raw_operations
    )

    selected_start = min(item.start_segment, *evidence_ids)
    selected_end = max(item.end_segment, *evidence_ids)
    selected_text = _range_text(segments, selected_start, selected_end)

    variables_grounded = all(
        _gate3_variable_appears(variable, selected_text)
        for variable in variables
    )
    operations_grounded = all(
        _gate3_has_operation_cue(operation, selected_text)
        for operation in operations
    )

    reason = str(normalized.get("reason") or "").strip()
    if variables_grounded and operations_grounded:
        if _claims_use_reported_result_as_operand(
            variables=variables,
            operations=operations,
            segments=segments,
            start=selected_start,
            end=selected_end,
        ):
            return InventoryAuditDecision(
                calculation_id=item.calculation_id,
                action=AuditAction.DOWNGRADE_NON_SYMBOLIC,
                evidence_segment_ids=evidence_ids,
                reason=(
                    "Gate 4 bounded recovery found that a claimed numeric "
                    "operand is presented by the source as the reported "
                    "result; the event is retained as non-symbolic rather "
                    "than treating an outcome as an input."
                ),
            )
        return InventoryAuditDecision(
            calculation_id=item.calculation_id,
            action=AuditAction.RECONCILE,
            evidence_segment_ids=evidence_ids,
            reason=(
                reason
                or "Gate 3 bounded lexical recovery grounded the revised claims."
            ),
            revised_variables_mentioned=variables,
            revised_operations_mentioned=operations,
        )

    # First search only the original bounded audit neighborhood.
    widened = _find_minimal_grounded_span(
        item=item,
        evidence_ids=evidence_ids,
        variables=variables,
        operations=operations,
        segments=segments,
        neighborhood_start=neighborhood_start,
        neighborhood_end=neighborhood_end,
    )
    widened_adjacent = False

    # If the model-selected result span already states the operation, allow a
    # tiny deterministic adjacent extension solely to recover named operands
    # that immediately precede/follow that result. No extra operation may be
    # borrowed from the extension.
    if widened is None and operations_grounded:
        extended_start, extended_end = _adjacent_bounds(
            segment_count=len(segments),
            neighborhood_start=neighborhood_start,
            neighborhood_end=neighborhood_end,
        )
        if (
            extended_start != neighborhood_start
            or extended_end != neighborhood_end
        ):
            widened = _find_minimal_grounded_span(
                item=item,
                evidence_ids=evidence_ids,
                variables=variables,
                operations=operations,
                segments=segments,
                neighborhood_start=extended_start,
                neighborhood_end=extended_end,
            )
            widened_adjacent = widened is not None

    if widened is not None:
        widened_start, widened_end = widened
        widened_ids = tuple(sorted({*evidence_ids, widened_start, widened_end}))
        if _claims_use_reported_result_as_operand(
            variables=variables,
            operations=operations,
            segments=segments,
            start=widened_start,
            end=widened_end,
        ):
            return InventoryAuditDecision(
                calculation_id=item.calculation_id,
                action=AuditAction.DOWNGRADE_NON_SYMBOLIC,
                evidence_segment_ids=widened_ids,
                reason=(
                    "Gate 4 bounded recovery found that a claimed numeric "
                    "operand is presented by the source as the reported "
                    "result; the event is retained as non-symbolic rather "
                    "than treating an outcome as an input."
                ),
            )
        suffix = (
            " Gate 3 adjacent operand recovery extended beyond the standard "
            "audit neighborhood only after the selected span independently "
            "grounded every claimed operation cue."
            if widened_adjacent
            else ""
        )
        return InventoryAuditDecision(
            calculation_id=item.calculation_id,
            action=AuditAction.RECONCILE,
            evidence_segment_ids=widened_ids,
            reason=(
                ((reason + " ") if reason else "")
                + "Gate 3 bounded evidence-hull recovery included nearby "
                "source segments needed to ground the unchanged revised claims."
                + suffix
            ),
            revised_variables_mentioned=variables,
            revised_operations_mentioned=operations,
        )

    # The audit model sometimes replaces source-extractive original values
    # with unspoken semantic aliases.  Recover only a mechanically grounded
    # subset of the original claims, and only when at least two distinct
    # quantities plus every unchanged operation are supported in the selected
    # hull.  Formula extraction remains responsible for naming the result.
    original_variables = _grounded_original_variables(
        item=item,
        segments=segments,
        start=selected_start,
        end=selected_end,
    )
    if (
        len(original_variables) >= 2
        and operations_grounded
        and not _claims_use_reported_result_as_operand(
            variables=original_variables,
            operations=operations,
            segments=segments,
            start=selected_start,
            end=selected_end,
        )
    ):
        return InventoryAuditDecision(
            calculation_id=item.calculation_id,
            action=AuditAction.RECONCILE,
            evidence_segment_ids=evidence_ids,
            reason=(
                ((reason + " ") if reason else "")
                + "Gate 4 source-extractive recovery retained only original "
                "inventory quantities and operations that are literally "
                "grounded in the selected source."
            ),
            revised_variables_mentioned=original_variables,
            revised_operations_mentioned=operations,
        )

    neighborhood_text = _range_text(
        segments,
        neighborhood_start,
        neighborhood_end,
    )
    operation_supported_anywhere = any(
        _gate3_has_operation_cue(operation, neighborhood_text)
        for operation in operations
    )
    if not operation_supported_anywhere:
        # Do not use an operation cue outside the original audit neighborhood
        # to validate the calculation. But before downgrading to non-symbolic,
        # inspect the same tiny adjacent guard used for operand recovery. If an
        # operation cue is immediately outside the audit window, fail closed:
        # the window may simply have clipped relevant operation evidence.
        extended_start, extended_end = _adjacent_bounds(
            segment_count=len(segments),
            neighborhood_start=neighborhood_start,
            neighborhood_end=neighborhood_end,
        )
        if (
            extended_start != neighborhood_start
            or extended_end != neighborhood_end
        ):
            adjacent_operation_supported = (
                _operation_has_compact_claim_association(
                    variables=variables,
                    operations=operations,
                    segments=segments,
                    start=extended_start,
                    end=extended_end,
                )
            )
            if adjacent_operation_supported:
                assert saved_error is not None
                raise saved_error

        return InventoryAuditDecision(
            calculation_id=item.calculation_id,
            action=AuditAction.DOWNGRADE_NON_SYMBOLIC,
            evidence_segment_ids=evidence_ids,
            reason=(
                "Gate 3 bounded recovery found no source cue for the claimed "
                "arithmetic operation anywhere in the audit neighborhood or "
                "its fixed adjacent guard; the event is retained as "
                "non-symbolic rather than inventing an operation."
            ),
        )

    assert saved_error is not None
    raise saved_error
