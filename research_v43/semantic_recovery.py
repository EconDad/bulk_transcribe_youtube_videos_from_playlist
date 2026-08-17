"""Generic fail-closed semantic recovery for v4.3 Stage F acceptance.

These repairs are intentionally domain-neutral and do not rewrite formulas or
source claims. They address three recurring model-shape failures:

* calculation inventory items that are only comparisons or worked numeric
  examples are downgraded to non-symbolic when the bounded transcript does not
  state a reusable arithmetic operation;
* formula candidates missing only mechanical identifier metadata receive a
  variable definition for the identifier without changing the expression;
* entailment evidence may widen inside the already-validated calculation span
  to include source ranges the same node already used for identifier grounding.
"""

from __future__ import annotations

from dataclasses import replace
import copy
import re
from typing import Any, Mapping, Sequence

from .calculation_inventory import (
    CalculationInventory,
    CalculationItem,
    audit_visual_equation_cues as _audit_visual_equation_cues,
)
from .entailment import (
    FormulaEntailmentReport,
    _OPERATION_CUES,
    _has_operation_cue,
    validate_entailment_response as _validate_entailment_response,
)
from .expression_ast import ExpressionValidationError, FormulaCandidate, parse_formula
from .formula_extraction import (
    FormulaExtractionResult,
    parse_formula_extraction_response as _parse_formula_extraction_response,
)
from .inventory_evidence_audit import item_needs_evidence_audit


_COMPARISON_OPERATIONS = {
    "comparison",
    "compare",
    "inequality",
    "greater_than",
    "less_than",
    "higher_than",
    "lower_than",
}

_EXAMPLE_CUE_RE = re.compile(
    r"\b(?:for example|example|suppose|let(?:'s| us) say|assume)\b",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[$€£]\s*)?[+-]?\d[\d,]*(?:\.\d+)?%?"
)


def _segment_text(
    segments: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> str:
    return " ".join(
        str(segments[index].get("text") or "").strip()
        for index in range(start, end + 1)
    ).strip()


def _normalized_operation(value: str) -> str:
    return re.sub(r"[\s-]+", "_", str(value).strip().casefold())


def _comparison_only(item: CalculationItem) -> bool:
    operations = tuple(
        _normalized_operation(value)
        for value in item.operations_mentioned
        if str(value).strip()
    )
    return bool(operations) and all(
        operation in _COMPARISON_OPERATIONS
        for operation in operations
    )


def _worked_numeric_example_without_stated_operation(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
    lookback_segments: int = 8,
    lookahead_segments: int = 2,
) -> bool:
    """Recognize a numeric worked example that lacks reusable operation text.

    This deliberately requires both an explicit example cue and a failure to
    ground the inventory's variables and operation cues. It therefore does not
    downgrade examples where the source actually states the reusable procedure.
    """

    if not item.operations_mentioned:
        return False

    canonical_operations = [
        operation
        for operation in item.operations_mentioned
        if operation in _OPERATION_CUES
    ]
    if not canonical_operations:
        return False

    needs_audit, reasons = item_needs_evidence_audit(
        item=item,
        segments=segments,
    )
    if not needs_audit:
        return False

    has_missing_variables = any(
        reason.startswith("current span lacks mentioned variables:")
        for reason in reasons
    )
    has_missing_operation = any(
        reason.startswith("current span lacks operation cues:")
        for reason in reasons
    )
    if not (has_missing_variables and has_missing_operation):
        return False

    context_start = max(0, item.start_segment - lookback_segments)
    context_end = min(
        len(segments) - 1,
        item.end_segment + lookahead_segments,
    )
    context_text = _segment_text(segments, context_start, context_end)
    item_text = _segment_text(
        segments,
        item.start_segment,
        item.end_segment,
    )

    if _EXAMPLE_CUE_RE.search(context_text) is None:
        return False

    if len(_NUMBER_RE.findall(item_text)) < 2:
        return False

    distinct_numbers = {
        re.sub(r"[\s,$€£]", "", match)
        for match in _NUMBER_RE.findall(context_text)
    }
    if len(distinct_numbers) < 3:
        return False

    if any(
        _has_operation_cue(operation, context_text)
        for operation in canonical_operations
    ):
        return False

    return True


def audit_visual_equation_cues_with_semantic_downgrades(
    *,
    inventory: CalculationInventory,
    segments: Sequence[Mapping[str, Any]],
) -> tuple[CalculationInventory, tuple[dict[str, Any], ...]]:
    """Run the frozen visual audit, then apply bounded semantic downgrades."""

    audited, visual_records = _audit_visual_equation_cues(
        inventory=inventory,
        segments=segments,
    )

    items: list[CalculationItem] = []
    records: list[dict[str, Any]] = list(visual_records)

    for item in audited.calculations:
        if not item.formula_expected or item.visual_equation_cue:
            items.append(item)
            continue

        reason: str | None = None
        action: str | None = None

        if _comparison_only(item):
            reason = (
                "Deterministic Stage F pre-audit classified the event as a "
                "comparison/inequality rather than a supported symbolic "
                "arithmetic expression."
            )
            action = "downgrade_non_symbolic_comparison"
        elif _worked_numeric_example_without_stated_operation(
            item=item,
            segments=segments,
        ):
            reason = (
                "Deterministic Stage F pre-audit found an explicit worked "
                "numeric example whose bounded source does not state the "
                "inventory's reusable arithmetic operation or variable names."
            )
            action = "downgrade_non_symbolic_numeric_example"

        if reason is None or action is None:
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


def _complete_candidate_variable_metadata(
    raw_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(raw_candidate))
    ascii_formula = candidate.get("ascii")
    if not isinstance(ascii_formula, str) or not ascii_formula.strip():
        return candidate

    try:
        parsed = parse_formula(ascii_formula)
    except ExpressionValidationError:
        return candidate

    raw_variables = candidate.get("variables")
    if (
        isinstance(raw_variables, (str, bytes))
        or not isinstance(raw_variables, Sequence)
    ):
        return candidate

    variables = [
        copy.deepcopy(dict(value))
        for value in raw_variables
        if isinstance(value, Mapping)
    ]
    if len(variables) != len(raw_variables):
        return candidate

    existing = {
        str(value.get("symbol") or "").strip()
        for value in variables
    }
    for symbol in parsed.identifiers:
        if symbol in existing:
            continue
        variables.append(
            {
                "symbol": symbol,
                "meaning": symbol.replace("_", " "),
                "unit": "",
            }
        )
        existing.add(symbol)

    candidate["variables"] = variables
    return candidate


def parse_formula_extraction_response_with_variable_completion(
    payload: Mapping[str, Any],
    *,
    item: CalculationItem,
) -> FormulaExtractionResult:
    """Complete missing AST-identifier metadata, then run strict validation.

    The formula expression, formula ID, source claims, derivation, and any model
    supplied variable definitions are unchanged. Only missing definitions for
    identifiers already present in the parsed expression are added.
    """

    repaired = copy.deepcopy(dict(payload))
    raw_candidates = repaired.get("candidates")
    if (
        not isinstance(raw_candidates, (str, bytes))
        and isinstance(raw_candidates, Sequence)
    ):
        repaired["candidates"] = [
            _complete_candidate_variable_metadata(candidate)
            if isinstance(candidate, Mapping)
            else candidate
            for candidate in raw_candidates
        ]

    return _parse_formula_extraction_response(repaired, item=item)


def _node_operation_issue(
    report: FormulaEntailmentReport,
    *,
    node_id: str,
    operation: str,
) -> bool:
    expected = f"{node_id} evidence lacks a cue for {operation}"
    return expected in report.issues


def validate_entailment_response_with_grounding_hull_repair(
    payload: Mapping[str, Any],
    *,
    item: CalculationItem,
    candidate: FormulaCandidate,
    segments: Sequence[Mapping[str, Any]],
) -> FormulaEntailmentReport:
    """Allow operation evidence to span the node's own grounding ranges.

    The first strict validation runs unchanged. If its only local defect for a
    node is a missing operation cue, the evidence range may widen to the
    contiguous hull of that node's existing evidence and identifier-grounding
    ranges, but never outside the calculation item. The source itself must then
    contain the required operation cue. No quote or grounding is invented.
    """

    report = _validate_entailment_response(
        payload,
        item=item,
        candidate=candidate,
        segments=segments,
    )
    if report.passed:
        return report

    repaired = copy.deepcopy(dict(payload))
    raw_nodes = repaired.get("nodes")
    if (
        isinstance(raw_nodes, (str, bytes))
        or not isinstance(raw_nodes, Sequence)
    ):
        return report

    changed = False
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping):
            continue
        node_id = str(raw_node.get("node_id") or "")
        operation = str(raw_node.get("operation") or "")
        if not _node_operation_issue(
            report,
            node_id=node_id,
            operation=operation,
        ):
            continue

        evidence = raw_node.get("evidence")
        groundings = raw_node.get("identifier_groundings")
        if (
            isinstance(evidence, (str, bytes))
            or not isinstance(evidence, Sequence)
            or not evidence
            or isinstance(groundings, (str, bytes))
            or not isinstance(groundings, Sequence)
        ):
            continue

        ranges: list[tuple[int, int]] = []
        for record in [*evidence, *groundings]:
            if not isinstance(record, Mapping):
                ranges = []
                break
            start = record.get("start_segment")
            end = record.get("end_segment")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
            ):
                ranges = []
                break
            ranges.append((start, end))
        if not ranges:
            continue

        hull_start = min(start for start, _ in ranges)
        hull_end = max(end for _, end in ranges)
        if (
            hull_start < item.start_segment
            or hull_end > item.end_segment
            or hull_end < hull_start
        ):
            continue

        source_text = _segment_text(segments, hull_start, hull_end)
        if not _has_operation_cue(operation, source_text):
            continue

        first = evidence[0]
        if not isinstance(first, Mapping):
            continue
        first["start_segment"] = hull_start
        first["end_segment"] = hull_end
        changed = True

    if not changed:
        return report

    return _validate_entailment_response(
        repaired,
        item=item,
        candidate=candidate,
        segments=segments,
    )
