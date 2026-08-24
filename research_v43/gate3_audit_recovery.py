"""Strict Gate 3 inventory-audit fallback with durable error handling."""

from __future__ import annotations

import json
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
    operation is grounded inside the model-selected contiguous span under the
    generic Gate 3 lexical rules. Otherwise the item is downgraded only when
    the claimed operation is absent from the *entire* bounded neighborhood.
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
        return InventoryAuditDecision(
            calculation_id=item.calculation_id,
            action=AuditAction.DOWNGRADE_NON_SYMBOLIC,
            evidence_segment_ids=evidence_ids,
            reason=(
                "Gate 3 bounded recovery found no source cue for the claimed "
                "arithmetic operation anywhere in the audit neighborhood; "
                "the event is retained as non-symbolic rather than inventing "
                "an operation."
            ),
        )

    assert saved_error is not None
    raise saved_error
