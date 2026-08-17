"""Conservative recovery for incomplete formula-like transcript fragments.

A reusable assignment needs both a source-grounded operation and a grounded
result concept. Some ASR/inventory items stop at an antecedent such as
"if you divide X by Y," and the model then invents a result label. This module
marks only that narrow pattern non-symbolic before formula extraction.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Mapping, Sequence

from .calculation_inventory import CalculationInventory, CalculationItem
from .entailment import _OPERATION_CUES, _has_operation_cue
from .inventory_evidence_audit import item_needs_evidence_audit
from .semantic_recovery import (
    audit_visual_equation_cues_with_semantic_downgrades,
)


_GENERIC_NAME_TOKENS = {
    "calculation",
    "compute",
    "computation",
    "equation",
    "formula",
    "operation",
    "relationship",
}

_RESULT_CUE_RE = re.compile(
    r"\b(?:get|gets|got|give|gives|yield|yields|produce|produces|"
    r"equal|equals|become|becomes|result|results)\b",
    re.IGNORECASE,
)


def _words(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value).casefold())
        if len(token) >= 3
    }


def _range_text(
    segments: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> str:
    return " ".join(
        str(segments[index].get("text") or "").strip()
        for index in range(start, end + 1)
    ).strip()


def _distinctive_result_tokens(item: CalculationItem) -> set[str]:
    variable_tokens: set[str] = set()
    for variable in item.variables_mentioned:
        variable_tokens.update(_words(variable))

    return (
        _words(item.name)
        - variable_tokens
        - _GENERIC_NAME_TOKENS
    )


def _is_incomplete_unnamed_operation_fragment(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
) -> bool:
    if not item.formula_expected or item.visual_equation_cue:
        return False

    operations = [
        operation
        for operation in item.operations_mentioned
        if operation in _OPERATION_CUES
    ]
    if not operations or len(operations) != len(item.operations_mentioned):
        return False

    needs_audit, _ = item_needs_evidence_audit(
        item=item,
        segments=segments,
    )
    if needs_audit:
        return False

    source_text = _range_text(
        segments,
        item.start_segment,
        item.end_segment,
    )
    if not source_text.rstrip().endswith((",", ";", ":")):
        return False

    if _RESULT_CUE_RE.search(source_text) is not None:
        return False

    if not all(
        _has_operation_cue(operation, source_text)
        for operation in operations
    ):
        return False

    distinctive = _distinctive_result_tokens(item)
    if not distinctive:
        return False

    source_words = _words(source_text)
    if distinctive & source_words:
        return False

    return True


def audit_with_incomplete_operation_fragment_downgrade(
    *,
    inventory: CalculationInventory,
    segments: Sequence[Mapping[str, Any]],
) -> tuple[CalculationInventory, tuple[dict[str, Any], ...]]:
    """Apply existing semantic audit, then downgrade unnamed fragments."""

    audited, prior_records = (
        audit_visual_equation_cues_with_semantic_downgrades(
            inventory=inventory,
            segments=segments,
        )
    )

    items: list[CalculationItem] = []
    records: list[dict[str, Any]] = list(prior_records)

    for item in audited.calculations:
        if not _is_incomplete_unnamed_operation_fragment(
            item=item,
            segments=segments,
        ):
            items.append(item)
            continue

        reason = (
            "Deterministic Stage F pre-audit found a source-grounded arithmetic "
            "antecedent that ends as an incomplete clause without a named or "
            "otherwise grounded result concept. Treating it as a reusable "
            "assignment would require inventing the left-hand result."
        )
        updated = replace(
            item,
            formula_expected=False,
            reason=item.reason + " " + reason,
        )
        items.append(updated)
        records.append(
            {
                "calculation_id": item.calculation_id,
                "action": "downgrade_non_symbolic_incomplete_operation_fragment",
                "decision_source": "deterministic_stage_f",
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
