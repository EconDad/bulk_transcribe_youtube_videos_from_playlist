"""Bounded evidence audit for merged v4.3 calculation inventory items."""

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
    KEEP = "keep"
    EXPAND = "expand"
    DOWNGRADE_NON_SYMBOLIC = "downgrade_non_symbolic"


class EvidenceKind(StrEnum):
    RELATIONSHIP = "relationship"
    OPERAND = "operand"
    RESULT = "result"


@dataclass(frozen=True, slots=True)
class AuditEvidence:
    kind: EvidenceKind
    start_segment: int
    end_segment: int
    quote: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "start_segment": self.start_segment,
            "end_segment": self.end_segment,
            "quote": self.quote,
        }


@dataclass(frozen=True, slots=True)
class InventoryAuditDecision:
    calculation_id: str
    action: AuditAction
    start_segment: int
    end_segment: int
    reason: str
    evidence: tuple[AuditEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation_id": self.calculation_id,
            "action": self.action.value,
            "start_segment": self.start_segment,
            "end_segment": self.end_segment,
            "reason": self.reason,
            "evidence": [item.to_dict() for item in self.evidence],
        }


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9.%$+-]+", "", value.casefold())


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

    compact_variable = _compact(variable)
    compact_text = _compact(text)
    if compact_variable and compact_variable in compact_text:
        return True

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", normalized_variable)
        if len(token) > 1
    ]
    return bool(tokens) and all(token in normalized_text for token in tokens)


def item_needs_evidence_audit(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
) -> tuple[bool, tuple[str, ...]]:
    """Select only inventory items whose current span lacks claimed evidence."""

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


def build_inventory_evidence_audit_prompt(
    *,
    item: CalculationItem,
    neighborhood_segments: Sequence[Mapping[str, Any]],
    selection_reasons: Sequence[str],
) -> str:
    """Build a domain-neutral, bounded evidence-audit prompt."""

    if not neighborhood_segments:
        raise InventoryEvidenceAuditError("neighborhood cannot be empty")

    schema = {
        "calculation_id": item.calculation_id,
        "action": "keep | expand | downgrade_non_symbolic",
        "start_segment": item.start_segment,
        "end_segment": item.end_segment,
        "reason": "Source-grounded reason.",
        "evidence": [
            {
                "kind": "relationship | operand | result",
                "start_segment": item.start_segment,
                "end_segment": item.end_segment,
                "quote": "Exact transcript quote.",
            }
        ],
    }

    return (
        "Audit one previously discovered calculation event using only the "
        "bounded transcript neighborhood below. Do not invent a formula and "
        "do not use outside subject-matter knowledge.\n\n"
        "Choose KEEP only when the current inventory span already contains "
        "enough source evidence for the reusable arithmetic relationship or "
        "procedure claimed by the item.\n"
        "Choose EXPAND only when nearby transcript segments supply missing "
        "operands, operation language, result language, or relationship "
        "context. The expanded range must be the smallest contiguous range "
        "that includes the original inventory range and all cited evidence.\n"
        "Choose DOWNGRADE_NON_SYMBOLIC only when the bounded source gives a "
        "numeric outcome, comparison, or observation but does not ground a "
        "reusable symbolic arithmetic relationship or procedure. Do not "
        "downgrade merely because the equation is difficult.\n"
        "Every evidence quote must be copied exactly from the supplied "
        "transcript. Return JSON only and match the schema exactly.\n\n"
        f"CURRENT ITEM:\n{json.dumps(item.to_dict(), indent=2)}\n\n"
        "WHY THIS ITEM WAS SELECTED FOR AUDIT:\n"
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
    """Request one schema-only correction without adding source evidence."""

    return (
        f"{original_prompt}\n\n"
        "Your previous JSON response failed deterministic validation. "
        "Correct only the response structure, ranges, action, or quoted "
        "evidence using the same bounded transcript. Do not invent new "
        "evidence or outside facts.\n\n"
        f"VALIDATION ERROR:\n{validation_error}\n\n"
        "PREVIOUS RESPONSE:\n"
        f"{json.dumps(dict(previous_response), indent=2)}"
    )


def parse_inventory_evidence_audit_response(
    response_text: str,
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
    neighborhood_start: int,
    neighborhood_end: int,
) -> InventoryAuditDecision:
    """Parse and deterministically validate one audit response."""

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
        "start_segment",
        "end_segment",
        "reason",
        "evidence",
    }
    if set(raw) != required:
        raise InventoryEvidenceAuditError(
            f"audit response must contain exactly {sorted(required)}"
        )

    if raw["calculation_id"] != item.calculation_id:
        raise InventoryEvidenceAuditError("calculation_id does not match item")

    try:
        action = AuditAction(raw["action"])
    except ValueError as exc:
        raise InventoryEvidenceAuditError("invalid audit action") from exc

    start = raw["start_segment"]
    end = raw["end_segment"]
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise InventoryEvidenceAuditError("audit ranges must be integers")
    if start < neighborhood_start or end > neighborhood_end or end < start:
        raise InventoryEvidenceAuditError(
            "audit range falls outside bounded neighborhood"
        )
    if start > item.start_segment or end < item.end_segment:
        raise InventoryEvidenceAuditError(
            "audit range must include original inventory range"
        )

    if action is AuditAction.KEEP and (
        start != item.start_segment or end != item.end_segment
    ):
        raise InventoryEvidenceAuditError(
            "keep action must preserve the original inventory range"
        )
    if action is AuditAction.EXPAND and (
        start == item.start_segment and end == item.end_segment
    ):
        raise InventoryEvidenceAuditError(
            "expand action must widen the inventory range"
        )
    if action is AuditAction.DOWNGRADE_NON_SYMBOLIC and (
        start != item.start_segment or end != item.end_segment
    ):
        raise InventoryEvidenceAuditError(
            "downgrade_non_symbolic must preserve the original range"
        )

    reason = raw["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise InventoryEvidenceAuditError("reason must be nonempty text")

    raw_evidence = raw["evidence"]
    if (
        isinstance(raw_evidence, (str, bytes))
        or not isinstance(raw_evidence, Sequence)
        or not raw_evidence
    ):
        raise InventoryEvidenceAuditError(
            "evidence must be a nonempty array"
        )

    evidence: list[AuditEvidence] = []
    for index, record in enumerate(raw_evidence):
        if not isinstance(record, Mapping):
            raise InventoryEvidenceAuditError(
                f"evidence[{index}] must be an object"
            )
        expected = {"kind", "start_segment", "end_segment", "quote"}
        if set(record) != expected:
            raise InventoryEvidenceAuditError(
                f"evidence[{index}] must contain exactly {sorted(expected)}"
            )
        try:
            kind = EvidenceKind(record["kind"])
        except ValueError as exc:
            raise InventoryEvidenceAuditError(
                f"evidence[{index}] has invalid kind"
            ) from exc

        ev_start = record["start_segment"]
        ev_end = record["end_segment"]
        quote = record["quote"]
        if (
            isinstance(ev_start, bool)
            or isinstance(ev_end, bool)
            or not isinstance(ev_start, int)
            or not isinstance(ev_end, int)
            or ev_end < ev_start
        ):
            raise InventoryEvidenceAuditError(
                f"evidence[{index}] has invalid range"
            )
        if ev_start < start or ev_end > end:
            raise InventoryEvidenceAuditError(
                f"evidence[{index}] falls outside audited item range"
            )
        if not isinstance(quote, str) or not quote.strip():
            raise InventoryEvidenceAuditError(
                f"evidence[{index}].quote must be nonempty"
            )
        source = _range_text(segments, ev_start, ev_end)
        if _normalize(quote) not in _normalize(source):
            raise InventoryEvidenceAuditError(
                f"evidence[{index}] quote is not present in cited segments"
            )

        evidence.append(
            AuditEvidence(
                kind=kind,
                start_segment=ev_start,
                end_segment=ev_end,
                quote=quote.strip(),
            )
        )

    if action in {AuditAction.KEEP, AuditAction.EXPAND}:
        if not any(
            item.kind is EvidenceKind.RELATIONSHIP
            for item in evidence
        ):
            raise InventoryEvidenceAuditError(
                "keep/expand requires relationship evidence"
            )

    return InventoryAuditDecision(
        calculation_id=item.calculation_id,
        action=action,
        start_segment=start,
        end_segment=end,
        reason=reason.strip(),
        evidence=tuple(evidence),
    )


def apply_inventory_audit_decision(
    *,
    item: CalculationItem,
    decision: InventoryAuditDecision,
) -> CalculationItem:
    """Apply a validated audit decision without changing semantic claims."""

    if decision.calculation_id != item.calculation_id:
        raise InventoryEvidenceAuditError("decision belongs to another item")

    formula_expected = item.formula_expected
    if decision.action is AuditAction.DOWNGRADE_NON_SYMBOLIC:
        formula_expected = False

    return CalculationItem(
        calculation_id=item.calculation_id,
        name=item.name,
        source_mode=SourceMode(item.source_mode),
        start_segment=decision.start_segment,
        end_segment=decision.end_segment,
        variables_mentioned=item.variables_mentioned,
        operations_mentioned=item.operations_mentioned,
        visual_equation_cue=item.visual_equation_cue,
        formula_expected=formula_expected,
        reason=f"{item.reason} Evidence audit: {decision.reason}",
    )
