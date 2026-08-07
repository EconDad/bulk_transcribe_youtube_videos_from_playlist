"""Deterministic-first bounded evidence audit for v4.3 inventory items."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import re
from typing import Any, Mapping, Sequence

from .calculation_inventory import CalculationItem, SourceMode
from .entailment import _has_operation_cue


class InventoryEvidenceAuditError(ValueError):
    """Raised when a bounded inventory-audit response is invalid."""


class AuditAction(StrEnum):
    EXPAND = "expand"
    DOWNGRADE_NON_SYMBOLIC = "downgrade_non_symbolic"


@dataclass(frozen=True, slots=True)
class InventoryAuditDecision:
    calculation_id: str
    action: AuditAction
    evidence_segment_ids: tuple[int, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation_id": self.calculation_id,
            "action": self.action.value,
            "evidence_segment_ids": list(self.evidence_segment_ids),
            "reason": self.reason,
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
    )


def build_inventory_evidence_audit_prompt(
    *,
    item: CalculationItem,
    neighborhood_segments: Sequence[Mapping[str, Any]],
    selection_reasons: Sequence[str],
) -> str:
    """Ask the model only for source segment IDs and a terminal action."""

    schema = {
        "calculation_id": item.calculation_id,
        "action": "expand | downgrade_non_symbolic",
        "evidence_segment_ids": [item.start_segment],
        "reason": "Source-grounded reason.",
    }

    return (
        "Audit one previously discovered calculation event using only the "
        "bounded transcript neighborhood below. Python has already tried to "
        "locate every inventory variable and operation cue deterministically "
        "and could not fully ground the current formula claim.\n\n"
        "Return segment IDs only. Do not reproduce transcript quotes. Do not "
        "classify evidence into categories. Do not calculate start/end ranges; "
        "Python will do that deterministically.\n\n"
        "Choose EXPAND only if the selected transcript segments, together with "
        "the original item span, explicitly ground the reusable arithmetic "
        "relationship or procedure already claimed by the inventory item. "
        "Choose DOWNGRADE_NON_SYMBOLIC when the bounded source gives only a "
        "numeric result, example, comparison, or observation without enough "
        "source detail for that reusable symbolic relationship. Do not invent "
        "a textbook formula or use outside subject-matter knowledge.\n\n"
        "The evidence_segment_ids must refer only to supplied segment IDs. "
        "Return JSON only and match the schema exactly.\n\n"
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
        "the action or evidence segment IDs. If an EXPAND action cannot make "
        "the existing inventory variables and operations source-grounded, "
        "choose DOWNGRADE_NON_SYMBOLIC instead. Do not add quotes, evidence "
        "types, start_segment, or end_segment.\n\n"
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


def parse_inventory_evidence_audit_response(
    response_text: str,
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
    neighborhood_start: int,
    neighborhood_end: int,
) -> InventoryAuditDecision:
    """Validate segment-ID-only model output."""

    try:
        raw = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise InventoryEvidenceAuditError(
            f"audit response is not valid JSON: {exc.msg}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise InventoryEvidenceAuditError("audit response must be an object")

    required = {
        "calculation_id",
        "action",
        "evidence_segment_ids",
        "reason",
    }
    if set(raw) != required:
        raise InventoryEvidenceAuditError(
            f"audit response must contain exactly {sorted(required)}"
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

    decision = InventoryAuditDecision(
        calculation_id=item.calculation_id,
        action=action,
        evidence_segment_ids=tuple(sorted(evidence_ids)),
        reason=reason.strip(),
    )

    if action is AuditAction.EXPAND:
        updated = apply_inventory_audit_decision(
            item=item,
            decision=decision,
        )
        if (
            updated.start_segment == item.start_segment
            and updated.end_segment == item.end_segment
        ):
            raise InventoryEvidenceAuditError(
                "expand evidence does not widen the inventory range"
            )

        still_needs_audit, reasons = item_needs_evidence_audit(
            item=updated,
            segments=segments,
        )
        if still_needs_audit:
            raise InventoryEvidenceAuditError(
                "expanded evidence does not ground current inventory claims: "
                + "; ".join(reasons)
            )

    return decision


def apply_inventory_audit_decision(
    *,
    item: CalculationItem,
    decision: InventoryAuditDecision,
) -> CalculationItem:
    """Compute span and formula expectation deterministically."""

    if decision.calculation_id != item.calculation_id:
        raise InventoryEvidenceAuditError("decision belongs to another item")

    if decision.action is AuditAction.DOWNGRADE_NON_SYMBOLIC:
        start = item.start_segment
        end = item.end_segment
        formula_expected = False
    else:
        start = min(item.start_segment, *decision.evidence_segment_ids)
        end = max(item.end_segment, *decision.evidence_segment_ids)
        formula_expected = item.formula_expected

    return CalculationItem(
        calculation_id=item.calculation_id,
        name=item.name,
        source_mode=SourceMode(item.source_mode),
        start_segment=start,
        end_segment=end,
        variables_mentioned=item.variables_mentioned,
        operations_mentioned=item.operations_mentioned,
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
