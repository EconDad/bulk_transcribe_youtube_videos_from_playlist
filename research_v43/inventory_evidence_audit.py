"""Deterministic-first bounded evidence audit for v4.3 inventory items."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import re
from typing import Any, Mapping, Sequence

from .calculation_inventory import CalculationItem, SourceMode
from .entailment import _OPERATION_CUES, _has_operation_cue


class InventoryEvidenceAuditError(ValueError):
    """Raised when a bounded inventory-audit response is invalid."""


class AuditAction(StrEnum):
    EXPAND = "expand"
    RECONCILE = "reconcile"
    DOWNGRADE_NON_SYMBOLIC = "downgrade_non_symbolic"


@dataclass(frozen=True, slots=True)
class InventoryAuditDecision:
    calculation_id: str
    action: AuditAction
    evidence_segment_ids: tuple[int, ...]
    reason: str
    revised_variables_mentioned: tuple[str, ...] = ()
    revised_operations_mentioned: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation_id": self.calculation_id,
            "action": self.action.value,
            "evidence_segment_ids": list(self.evidence_segment_ids),
            "reason": self.reason,
            "revised_variables_mentioned": list(
                self.revised_variables_mentioned
            ),
            "revised_operations_mentioned": list(
                self.revised_operations_mentioned
            ),
        }


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9.%$+-]+", "", value.casefold())


_NUMERIC_LITERAL_RE = re.compile(
    r"(?<![a-z0-9.])"
    r"(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<percent>%|percent\b)?",
    re.IGNORECASE,
)


def _word_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def _numeric_signatures(value: str) -> tuple[tuple[str, bool], ...]:
    signatures: list[tuple[str, bool]] = []
    for match in _NUMERIC_LITERAL_RE.finditer(value):
        raw_number = match.group("number").replace(",", "")
        try:
            numeric = format(float(raw_number), ".15g")
        except ValueError:
            continue
        signatures.append((numeric, bool(match.group("percent"))))
    return tuple(signatures)


def _semantic_word_tokens(value: str) -> tuple[str, ...]:
    without_numbers = _NUMERIC_LITERAL_RE.sub(" ", value)
    return tuple(
        token
        for token in re.findall(r"[a-z]+", without_numbers.casefold())
        if len(token) > 1
    )


def _singularize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _range_text(
    segments: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> str:
    parts: list[str] = []
    for index in range(start, end + 1):
        if index < 0 or index >= len(segments):
            raise InventoryEvidenceAuditError(
                f"segment range {start}-{end} exceeds transcript"
            )
        text = segments[index].get("text")
        if not isinstance(text, str):
            raise InventoryEvidenceAuditError(
                f"segments[{index}].text must be a string"
            )
        parts.append(text.strip())
    return " ".join(parts)


def _variable_appears(variable: str, text: str) -> bool:
    normalized_variable = _normalize(variable)
    normalized_text = _normalize(text)
    if normalized_variable and normalized_variable in normalized_text:
        return True

    variable_numbers = _numeric_signatures(variable)
    if variable_numbers:
        text_numbers = _numeric_signatures(text)
        for signature in variable_numbers:
            if signature not in text_numbers:
                return False

        variable_words = [
            _singularize_token(token)
            for token in _semantic_word_tokens(variable)
        ]
        if not variable_words:
            return True

        text_words = {
            _singularize_token(token)
            for token in _semantic_word_tokens(text)
        }
        return all(token in text_words for token in variable_words)

    compact_variable = _compact(variable)
    compact_text = _compact(text)
    if compact_variable and compact_variable in compact_text:
        return True

    variable_tokens = [
        _singularize_token(token)
        for token in _word_tokens(variable)
        if len(token) > 1
    ]
    text_tokens = {
        _singularize_token(token)
        for token in _word_tokens(text)
        if len(token) > 1
    }
    return bool(variable_tokens) and all(
        token in text_tokens
        for token in variable_tokens
    )


def _distance_to_item(index: int, item: CalculationItem) -> int:
    if item.start_segment <= index <= item.end_segment:
        return 0
    if index < item.start_segment:
        return item.start_segment - index
    return index - item.end_segment


def item_needs_evidence_audit(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
) -> tuple[bool, tuple[str, ...]]:
    """Select items whose current span does not ground inventory claims."""

    if not item.formula_expected or item.visual_equation_cue:
        return False, ()

    text = _range_text(segments, item.start_segment, item.end_segment)
    reasons: list[str] = []

    missing_variables = [
        variable
        for variable in item.variables_mentioned
        if not _variable_appears(variable, text)
    ]
    if missing_variables:
        reasons.append(
            "current span lacks mentioned variables: "
            + ", ".join(missing_variables)
        )

    unsupported_operations = [
        operation
        for operation in item.operations_mentioned
        if not _has_operation_cue(operation, text)
    ]
    if unsupported_operations:
        reasons.append(
            "current span lacks operation cues: "
            + ", ".join(unsupported_operations)
        )

    if not item.variables_mentioned and not item.operations_mentioned:
        reasons.append("formula-expected item has no inventory evidence claims")

    return bool(reasons), tuple(reasons)


def find_deterministic_expansion(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
    neighborhood_start: int,
    neighborhood_end: int,
    max_auto_distance: int = 3,
) -> InventoryAuditDecision | None:
    """Find the minimal expansion that grounds every current inventory claim."""

    needs_audit, _ = item_needs_evidence_audit(
        item=item,
        segments=segments,
    )
    if not needs_audit:
        return None

    selected: set[int] = set()
    current_text = _range_text(
        segments,
        item.start_segment,
        item.end_segment,
    )

    for variable in item.variables_mentioned:
        if _variable_appears(variable, current_text):
            continue
        matches = [
            index
            for index in range(neighborhood_start, neighborhood_end + 1)
            if _distance_to_item(index, item) <= max_auto_distance
            and _variable_appears(
                variable,
                _range_text(segments, index, index),
            )
        ]
        if not matches:
            return None
        selected.add(
            min(
                matches,
                key=lambda index: (
                    _distance_to_item(index, item),
                    index,
                ),
            )
        )

    for operation in item.operations_mentioned:
        if _has_operation_cue(operation, current_text):
            continue
        matches = [
            index
            for index in range(neighborhood_start, neighborhood_end + 1)
            if _distance_to_item(index, item) <= max_auto_distance
            and _has_operation_cue(
                operation,
                _range_text(segments, index, index),
            )
        ]
        if not matches:
            return None
        selected.add(
            min(
                matches,
                key=lambda index: (
                    _distance_to_item(index, item),
                    index,
                ),
            )
        )

    if not selected:
        return None

    start = min(item.start_segment, *selected)
    end = max(item.end_segment, *selected)

    candidate = CalculationItem(
        calculation_id=item.calculation_id,
        name=item.name,
        source_mode=SourceMode(item.source_mode),
        start_segment=start,
        end_segment=end,
        variables_mentioned=item.variables_mentioned,
        operations_mentioned=item.operations_mentioned,
        visual_equation_cue=item.visual_equation_cue,
        formula_expected=item.formula_expected,
        reason=item.reason,
    )
    still_needs_audit, _ = item_needs_evidence_audit(
        item=candidate,
        segments=segments,
    )
    if still_needs_audit:
        return None

    evidence_ids = tuple(
        sorted(
            {
                *selected,
                *range(item.start_segment, item.end_segment + 1),
            }
        )
    )
    return InventoryAuditDecision(
        calculation_id=item.calculation_id,
        action=AuditAction.EXPAND,
        evidence_segment_ids=evidence_ids,
        reason=(
            "Deterministic bounded search found the missing inventory "
            "variables and operation cues."
        ),
        revised_variables_mentioned=item.variables_mentioned,
        revised_operations_mentioned=item.operations_mentioned,
    )


def build_inventory_evidence_audit_prompt(
    *,
    item: CalculationItem,
    neighborhood_segments: Sequence[Mapping[str, Any]],
    selection_reasons: Sequence[str],
) -> str:
    """Ask the model to reconcile faulty claims or downgrade the event."""

    allowed_operations = sorted(_OPERATION_CUES)
    schema = {
        "calculation_id": item.calculation_id,
        "action": "reconcile | downgrade_non_symbolic",
        "evidence_segment_ids": [item.start_segment],
        "revised_variables_mentioned": ["source-grounded variable"],
        "revised_operations_mentioned": ["multiplication"],
        "reason": "Source-grounded reason.",
    }

    return (
        "Audit one previously discovered calculation event using only the "
        "bounded transcript neighborhood below. Python has already tried to "
        "ground the current inventory claims deterministically and could not. "
        "The inventory variables or operations may themselves be inaccurate.\n\n"
        "Return segment IDs and corrected inventory claims; do not return a "
        "formula, transcript quotes, evidence categories, or start/end ranges. "
        "Python will copy source text, compute the span, and revalidate every "
        "claim.\n\n"
        "Choose RECONCILE when the same calculation event is formula-bearing "
        "but the existing variables, operations, or span are wrong or "
        "incomplete. Prefer reusable short noun phrases actually spoken in "
        "the selected source; use exact numeric literals when the source does "
        "not name the quantity. Revised "
        "operations must use only the canonical operation names listed below. "
        "The corrected claims must preserve the semantic target of the same "
        "calculation event; do not substitute a different neighboring example "
        "or arithmetic sub-step.\n\n"
        "Choose DOWNGRADE_NON_SYMBOLIC when the bounded source provides only a "
        "numeric outcome, example, comparison, or observation without enough "
        "detail for a reusable symbolic arithmetic relationship. For a "
        "downgrade, return empty revised-variable and revised-operation arrays. "
        "Do not invent textbook relationships or use outside knowledge.\n\n"
        "Do not borrow an operation from a neighboring numeric example unless "
        "the transcript explicitly links that procedure to this same event. "
        "All evidence_segment_ids must refer only to supplied segments. Return "
        "JSON only and match the schema exactly.\n\n"
        f"CANONICAL OPERATIONS:\n{json.dumps(allowed_operations)}\n\n"
        f"CURRENT ITEM:\n{json.dumps(item.to_dict(), indent=2)}\n\n"
        "WHY DETERMINISTIC AUDIT COULD NOT FULLY GROUND IT:\n"
        f"{json.dumps(list(selection_reasons), indent=2)}\n\n"
        f"SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
        "BOUNDED TRANSCRIPT NEIGHBORHOOD:\n"
        f"{json.dumps(list(neighborhood_segments), indent=2)}"
    )


def build_inventory_evidence_repair_prompt(
    *,
    original_prompt: str,
    previous_response: Mapping[str, Any],
    validation_error: str,
) -> str:
    """Request one correction using the same bounded source."""

    return (
        f"{original_prompt}\n\n"
        "Your previous response failed deterministic validation. Correct only "
        "the action, evidence segment IDs, or revised inventory claims. If the "
        "same calculation event cannot be reconciled using explicit bounded "
        "source evidence, choose DOWNGRADE_NON_SYMBOLIC and return empty "
        "revised claim arrays. Do not add quotes, formulas, evidence types, "
        "start_segment, or end_segment.\n\n"
        f"VALIDATION ERROR:\n{validation_error}\n\n"
        "PREVIOUS RESPONSE:\n"
        f"{json.dumps(dict(previous_response), indent=2)}"
    )


def _edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, start=1):
        current = [i]
        for j, right in enumerate(second, start=1):
            substitution = previous[j - 1] + (left != right)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def _parse_audit_action(value: Any) -> AuditAction:
    if not isinstance(value, str):
        raise InventoryEvidenceAuditError("invalid audit action")

    normalized = re.sub(r"[\s-]+", "_", value.strip().casefold())
    try:
        return AuditAction(normalized)
    except ValueError:
        pass

    candidates = []
    for action in AuditAction:
        distance = _edit_distance(normalized, action.value)
        candidates.append((distance, action))

    candidates.sort(key=lambda item: (item[0], item[1].value))
    if (
        candidates
        and candidates[0][0] <= 2
        and (
            len(candidates) == 1
            or candidates[0][0] < candidates[1][0]
        )
    ):
        return candidates[0][1]

    raise InventoryEvidenceAuditError("invalid audit action")


def _parse_string_array(
    value: Any,
    *,
    field: str,
    max_items: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InventoryEvidenceAuditError(f"{field} must be an array")
    if len(value) > max_items:
        raise InventoryEvidenceAuditError(
            f"{field} may contain at most {max_items} items"
        )
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise InventoryEvidenceAuditError(
                f"{field}[{index}] must be nonempty text"
            )
        normalized = item.strip()
        if normalized in seen:
            raise InventoryEvidenceAuditError(
                f"{field} contains duplicate value: {normalized}"
            )
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _selected_source_text(
    *,
    item: CalculationItem,
    evidence_segment_ids: Sequence[int],
    segments: Sequence[Mapping[str, Any]],
) -> str:
    ids = {
        *range(item.start_segment, item.end_segment + 1),
        *evidence_segment_ids,
    }
    return " ".join(
        _range_text(segments, index, index)
        for index in sorted(ids)
    )


def _validate_revised_claims(
    *,
    variables: Sequence[str],
    operations: Sequence[str],
    source_text: str,
) -> None:
    if not variables:
        raise InventoryEvidenceAuditError(
            "reconcile requires at least one revised variable"
        )
    if not operations:
        raise InventoryEvidenceAuditError(
            "reconcile requires at least one revised operation"
        )

    missing_variables = [
        variable
        for variable in variables
        if not _variable_appears(variable, source_text)
    ]
    if missing_variables:
        raise InventoryEvidenceAuditError(
            "revised variables are not grounded in selected source: "
            + ", ".join(missing_variables)
        )

    unknown_operations = [
        operation
        for operation in operations
        if operation not in _OPERATION_CUES
    ]
    if unknown_operations:
        raise InventoryEvidenceAuditError(
            "revised operations are not canonical: "
            + ", ".join(unknown_operations)
        )

    missing_operations = [
        operation
        for operation in operations
        if not _has_operation_cue(operation, source_text)
    ]
    if missing_operations:
        raise InventoryEvidenceAuditError(
            "revised operations are not grounded in selected source: "
            + ", ".join(missing_operations)
        )


def parse_inventory_evidence_audit_response(
    response_text: str,
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
    neighborhood_start: int,
    neighborhood_end: int,
) -> InventoryAuditDecision:
    """Validate bounded model output and source-ground revised claims."""

    try:
        raw = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise InventoryEvidenceAuditError(
            f"audit response is not valid JSON: {exc.msg}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise InventoryEvidenceAuditError("audit response must be an object")

    legacy_fields = {
        "calculation_id",
        "action",
        "evidence_segment_ids",
        "reason",
    }
    reconcile_fields = {
        *legacy_fields,
        "revised_variables_mentioned",
        "revised_operations_mentioned",
    }
    fields = set(raw)
    if fields != legacy_fields and fields != reconcile_fields:
        raise InventoryEvidenceAuditError(
            "audit response has invalid fields; expected either legacy "
            f"{sorted(legacy_fields)} or reconciliation "
            f"{sorted(reconcile_fields)}"
        )
    if raw["calculation_id"] != item.calculation_id:
        raise InventoryEvidenceAuditError("calculation_id does not match item")

    action = _parse_audit_action(raw["action"])

    evidence_ids_raw = raw["evidence_segment_ids"]
    if (
        isinstance(evidence_ids_raw, (str, bytes))
        or not isinstance(evidence_ids_raw, Sequence)
        or not evidence_ids_raw
    ):
        raise InventoryEvidenceAuditError(
            "evidence_segment_ids must be a nonempty array"
        )

    evidence_ids: list[int] = []
    seen: set[int] = set()
    for index, value in enumerate(evidence_ids_raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InventoryEvidenceAuditError(
                f"evidence_segment_ids[{index}] must be an integer"
            )
        if value < neighborhood_start or value > neighborhood_end:
            raise InventoryEvidenceAuditError(
                f"evidence segment {value} falls outside bounded neighborhood"
            )
        if value not in seen:
            seen.add(value)
            evidence_ids.append(value)

    reason = raw["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise InventoryEvidenceAuditError("reason must be nonempty text")

    has_revised_fields = fields == reconcile_fields
    if has_revised_fields:
        revised_variables = _parse_string_array(
            raw["revised_variables_mentioned"],
            field="revised_variables_mentioned",
            max_items=8,
        )
        revised_operations = _parse_string_array(
            raw["revised_operations_mentioned"],
            field="revised_operations_mentioned",
            max_items=4,
        )
    else:
        revised_variables = item.variables_mentioned
        revised_operations = item.operations_mentioned

    if action is AuditAction.DOWNGRADE_NON_SYMBOLIC:
        if has_revised_fields and (revised_variables or revised_operations):
            raise InventoryEvidenceAuditError(
                "downgrade_non_symbolic requires empty revised claim arrays"
            )
        return InventoryAuditDecision(
            calculation_id=item.calculation_id,
            action=action,
            evidence_segment_ids=tuple(sorted(evidence_ids)),
            reason=reason.strip(),
        )

    if action is AuditAction.RECONCILE and not has_revised_fields:
        raise InventoryEvidenceAuditError(
            "reconcile requires revised claim arrays"
        )

    # Legacy model EXPAND remains valid for regression compatibility. New
    # model responses use RECONCILE; an EXPAND carrying revised fields is
    # normalized to RECONCILE because semantic claims may have changed.
    if action is AuditAction.EXPAND and has_revised_fields:
        action = AuditAction.RECONCILE

    source_text = _selected_source_text(
        item=item,
        evidence_segment_ids=evidence_ids,
        segments=segments,
    )
    _validate_revised_claims(
        variables=revised_variables,
        operations=revised_operations,
        source_text=source_text,
    )

    decision = InventoryAuditDecision(
        calculation_id=item.calculation_id,
        action=action,
        evidence_segment_ids=tuple(sorted(evidence_ids)),
        reason=reason.strip(),
        revised_variables_mentioned=tuple(revised_variables),
        revised_operations_mentioned=tuple(revised_operations),
    )
    updated = apply_inventory_audit_decision(
        item=item,
        decision=decision,
    )

    range_changed = (
        updated.start_segment != item.start_segment
        or updated.end_segment != item.end_segment
    )
    claims_changed = (
        updated.variables_mentioned != item.variables_mentioned
        or updated.operations_mentioned != item.operations_mentioned
    )
    if action is AuditAction.EXPAND and not range_changed:
        raise InventoryEvidenceAuditError(
            "expand evidence does not widen the inventory range"
        )
    if action is AuditAction.RECONCILE and not (range_changed or claims_changed):
        raise InventoryEvidenceAuditError(
            "reconcile must change the span or inventory claims"
        )

    still_needs_audit, reasons = item_needs_evidence_audit(
        item=updated,
        segments=segments,
    )
    if still_needs_audit:
        raise InventoryEvidenceAuditError(
            "reconciled evidence does not ground revised inventory claims: "
            + "; ".join(reasons)
        )

    return decision


def apply_inventory_audit_decision(
    *,
    item: CalculationItem,
    decision: InventoryAuditDecision,
) -> CalculationItem:
    """Compute span, claims, and formula expectation deterministically."""

    if decision.calculation_id != item.calculation_id:
        raise InventoryEvidenceAuditError("decision belongs to another item")

    if decision.action is AuditAction.DOWNGRADE_NON_SYMBOLIC:
        start = item.start_segment
        end = item.end_segment
        formula_expected = False
        variables = item.variables_mentioned
        operations = item.operations_mentioned
    else:
        start = min(item.start_segment, *decision.evidence_segment_ids)
        end = max(item.end_segment, *decision.evidence_segment_ids)
        formula_expected = True
        variables = (
            decision.revised_variables_mentioned
            or item.variables_mentioned
        )
        operations = (
            decision.revised_operations_mentioned
            or item.operations_mentioned
        )

    return CalculationItem(
        calculation_id=item.calculation_id,
        name=item.name,
        source_mode=SourceMode(item.source_mode),
        start_segment=start,
        end_segment=end,
        variables_mentioned=tuple(variables),
        operations_mentioned=tuple(operations),
        visual_equation_cue=item.visual_equation_cue,
        formula_expected=formula_expected,
        reason=f"{item.reason} Evidence audit: {decision.reason}",
    )


def decision_evidence_records(
    *,
    decision: InventoryAuditDecision,
    segments: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Copy canonical source text for selected evidence IDs."""

    return tuple(
        {
            "segment_id": segment_id,
            "source_text": _range_text(segments, segment_id, segment_id),
        }
        for segment_id in decision.evidence_segment_ids
    )
