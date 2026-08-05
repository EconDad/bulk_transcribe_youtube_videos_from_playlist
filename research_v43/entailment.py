"""Expression-node entailment for research pipeline v4.3."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
import json
import re
from typing import Any, Mapping, Sequence

from .calculation_inventory import CalculationItem
from .expression_ast import FormulaCandidate


class EntailmentValidationError(ValueError):
    """Raised when an entailment response is structurally invalid."""


class NodeStatus(StrEnum):
    ENTAILED = "entailed"
    DERIVED = "derived"


_OPERATION_CUES: dict[str, tuple[str, ...]] = {
    "addition": ("add", "added", "plus", "sum", "total", "combined"),
    "subtraction": (
        "subtract",
        "subtracted",
        "minus",
        "difference",
        "less",
        "deduct",
    ),
    "multiplication": (
        "multiply",
        "multiplied",
        "times",
        "product",
    ),
    "division": (
        "divide",
        "divided",
        "division",
        "ratio",
        "over",
        "per",
    ),
    "exponentiation": (
        "power",
        "exponent",
        "squared",
        "cubed",
        "raised",
    ),
    "unary_plus": ("positive", "plus"),
    "unary_minus": ("negative", "minus"),
    "function:sum": ("sum", "total", "add"),
    "function:sqrt": ("square root", "root"),
    "function:log": ("log", "logarithm"),
    "function:exp": ("exponential", "exp"),
    "function:abs": ("absolute", "magnitude"),
    "function:min": ("minimum", "smaller", "lowest"),
    "function:max": ("maximum", "larger", "highest"),
}


@dataclass(frozen=True, slots=True)
class EvidenceRange:
    start_segment: int
    end_segment: int
    quote: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvidenceRange":
        required = {"start_segment", "end_segment", "quote"}
        if set(raw) != required:
            raise EntailmentValidationError(
                "Evidence range must contain exactly "
                f"{sorted(required)}"
            )
        start = _require_segment(raw["start_segment"], "start_segment")
        end = _require_segment(raw["end_segment"], "end_segment")
        if end < start:
            raise EntailmentValidationError(
                "Evidence end_segment cannot precede start_segment"
            )
        return cls(
            start_segment=start,
            end_segment=end,
            quote=_require_string(raw["quote"], "quote"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_segment": self.start_segment,
            "end_segment": self.end_segment,
            "quote": self.quote,
        }


@dataclass(frozen=True, slots=True)
class NodeEntailment:
    node_id: str
    expression: str
    operation: str
    status: NodeStatus
    evidence: tuple[EvidenceRange, ...]
    depends_on_node_ids: tuple[str, ...]
    derivation_step: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NodeEntailment":
        required = {
            "node_id",
            "expression",
            "operation",
            "status",
            "evidence",
            "depends_on_node_ids",
            "derivation_step",
        }
        if set(raw) != required:
            raise EntailmentValidationError(
                "Node entailment must contain exactly "
                f"{sorted(required)}"
            )

        try:
            status = NodeStatus(_require_string(raw["status"], "status"))
        except ValueError as exc:
            raise EntailmentValidationError(
                "Node status must be entailed or derived"
            ) from exc

        raw_evidence = raw["evidence"]
        if isinstance(raw_evidence, (str, bytes)) or not isinstance(
            raw_evidence, Sequence
        ):
            raise EntailmentValidationError("evidence must be an array")
        evidence: list[EvidenceRange] = []
        for index, item in enumerate(raw_evidence):
            if not isinstance(item, Mapping):
                raise EntailmentValidationError(
                    f"evidence[{index}] must be an object"
                )
            evidence.append(EvidenceRange.from_mapping(item))

        dependencies = _require_string_array(
            raw["depends_on_node_ids"], "depends_on_node_ids"
        )
        derivation_step = raw["derivation_step"]
        if not isinstance(derivation_step, str):
            raise EntailmentValidationError(
                "derivation_step must be a string"
            )
        derivation_step = derivation_step.strip()

        if status is NodeStatus.ENTAILED:
            if not evidence:
                raise EntailmentValidationError(
                    "entailed node requires evidence"
                )
        else:
            if not dependencies:
                raise EntailmentValidationError(
                    "derived node requires dependencies"
                )
            if not derivation_step:
                raise EntailmentValidationError(
                    "derived node requires a derivation_step"
                )

        return cls(
            node_id=_require_string(raw["node_id"], "node_id"),
            expression=_require_string(raw["expression"], "expression"),
            operation=_require_string(raw["operation"], "operation"),
            status=status,
            evidence=tuple(evidence),
            depends_on_node_ids=tuple(dependencies),
            derivation_step=derivation_step,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "expression": self.expression,
            "operation": self.operation,
            "status": self.status.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "depends_on_node_ids": list(self.depends_on_node_ids),
            "derivation_step": self.derivation_step,
        }


@dataclass(frozen=True, slots=True)
class FormulaEntailmentReport:
    calculation_id: str
    formula_id: str
    passed: bool
    issues: tuple[str, ...]
    nodes: tuple[NodeEntailment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation_id": self.calculation_id,
            "formula_id": self.formula_id,
            "passed": self.passed,
            "issues": list(self.issues),
            "nodes": [item.to_dict() for item in self.nodes],
        }


def build_entailment_prompt(
    *,
    item: CalculationItem,
    candidate: FormulaCandidate,
    segments: Sequence[Mapping[str, Any]],
) -> str:
    """Build a prompt that asks for evidence for every expression node."""

    selected: list[dict[str, Any]] = []
    for index in range(item.start_segment, item.end_segment + 1):
        if index >= len(segments):
            raise EntailmentValidationError(
                f"{item.calculation_id} exceeds available segments"
            )
        text = segments[index].get("text")
        if not isinstance(text, str):
            raise EntailmentValidationError(
                f"segments[{index}].text must be a string"
            )
        selected.append({"segment_id": index, "text": text.strip()})

    node_schema = {
        "node_id": "NODE_0001",
        "expression": "Canonical node expression",
        "operation": "Operation copied from parsed formula",
        "status": "entailed | derived",
        "evidence": [
            {
                "start_segment": item.start_segment,
                "end_segment": item.end_segment,
                "quote": "Exact quote from those segments",
            }
        ],
        "depends_on_node_ids": [],
        "derivation_step": "Empty for entailed; required for derived.",
    }
    response_schema = {
        "calculation_id": item.calculation_id,
        "formula_id": candidate.formula_id,
        "nodes": [node_schema],
    }

    return (
        "Validate every operation node in the proposed formula against the "
        "source. Use status entailed only when the transcript directly states "
        "the operands and arithmetic relationship. Use status derived only "
        "when the node follows algebraically from other validated nodes, and "
        "list those dependencies. Quotes must be exact substrings of the cited "
        "segment range. Do not use outside knowledge. Return JSON only.\n\n"
        f"CALCULATION EVENT:\n{json.dumps(item.to_dict(), indent=2)}\n\n"
        f"FORMULA CANDIDATE:\n{json.dumps(candidate.to_dict(), indent=2)}\n\n"
        f"PARSED OPERATION NODES:\n"
        f"{json.dumps([node.to_dict() for node in candidate.parsed.operations], indent=2)}\n\n"
        f"RESPONSE SCHEMA:\n{json.dumps(response_schema, indent=2)}\n\n"
        f"SOURCE SEGMENTS:\n{json.dumps(selected, indent=2)}"
    )


def validate_entailment_response(
    payload: Mapping[str, Any],
    *,
    item: CalculationItem,
    candidate: FormulaCandidate,
    segments: Sequence[Mapping[str, Any]],
) -> FormulaEntailmentReport:
    """Validate structural and lexical support for every formula AST node."""

    required = {"calculation_id", "formula_id", "nodes"}
    if set(payload) != required:
        raise EntailmentValidationError(
            "Entailment response must contain exactly "
            f"{sorted(required)}"
        )
    if payload["calculation_id"] != item.calculation_id:
        raise EntailmentValidationError(
            "Entailment calculation_id does not match inventory"
        )
    if payload["formula_id"] != candidate.formula_id:
        raise EntailmentValidationError(
            "Entailment formula_id does not match candidate"
        )

    raw_nodes = payload["nodes"]
    if isinstance(raw_nodes, (str, bytes)) or not isinstance(
        raw_nodes, Sequence
    ):
        raise EntailmentValidationError("nodes must be an array")

    nodes: list[NodeEntailment] = []
    seen: set[str] = set()
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, Mapping):
            raise EntailmentValidationError(
                f"nodes[{index}] must be an object"
            )
        node = NodeEntailment.from_mapping(raw_node)
        if node.node_id in seen:
            raise EntailmentValidationError(
                f"Duplicate node entailment: {node.node_id}"
            )
        seen.add(node.node_id)
        nodes.append(node)

    parsed_by_id = {
        node.node_id: node
        for node in candidate.parsed.operations
    }
    provided_by_id = {node.node_id: node for node in nodes}

    issues: list[str] = []
    missing = set(parsed_by_id) - set(provided_by_id)
    extra = set(provided_by_id) - set(parsed_by_id)
    if missing:
        issues.append(f"Missing node entailments: {sorted(missing)}")
    if extra:
        issues.append(f"Unknown node entailments: {sorted(extra)}")

    variable_phrases = _variable_phrases(candidate)

    for node_id in sorted(set(parsed_by_id) & set(provided_by_id)):
        parsed_node = parsed_by_id[node_id]
        record = provided_by_id[node_id]
        if record.expression != parsed_node.expression:
            issues.append(f"{node_id} expression does not match AST")
        if record.operation != parsed_node.operation:
            issues.append(f"{node_id} operation does not match AST")

        combined_evidence: list[str] = []
        for evidence in record.evidence:
            if (
                evidence.start_segment < item.start_segment
                or evidence.end_segment > item.end_segment
            ):
                issues.append(
                    f"{node_id} evidence falls outside inventory range"
                )
                continue
            source_text = _range_text(
                segments,
                evidence.start_segment,
                evidence.end_segment,
            )
            if _normalize(evidence.quote) not in _normalize(source_text):
                issues.append(
                    f"{node_id} quote is not present in cited segments"
                )
            combined_evidence.append(source_text)

        if record.status is NodeStatus.ENTAILED:
            evidence_text = " ".join(combined_evidence)
            if not _has_operation_cue(record.operation, evidence_text):
                issues.append(
                    f"{node_id} evidence lacks a cue for {record.operation}"
                )
            dependency_identifiers: set[str] = set()
            for dependency in record.depends_on_node_ids:
                dependency_node = parsed_by_id.get(dependency)
                if dependency_node is not None:
                    dependency_identifiers.update(
                        _identifiers_in_expression(
                            dependency_node.expression
                        )
                    )
            for identifier in _identifiers_in_expression(
                parsed_node.expression
            ):
                if identifier in dependency_identifiers:
                    continue
                phrases = variable_phrases.get(identifier, (identifier,))
                if not any(
                    _normalize(phrase) in _normalize(evidence_text)
                    for phrase in phrases
                    if phrase
                ):
                    issues.append(
                        f"{node_id} evidence does not identify {identifier}"
                    )

        for dependency in record.depends_on_node_ids:
            if dependency not in parsed_by_id:
                issues.append(
                    f"{node_id} depends on unknown node {dependency}"
                )
            if dependency == node_id:
                issues.append(f"{node_id} cannot depend on itself")

    issues.extend(_dependency_cycle_issues(nodes))

    return FormulaEntailmentReport(
        calculation_id=item.calculation_id,
        formula_id=candidate.formula_id,
        passed=not issues,
        issues=tuple(issues),
        nodes=tuple(nodes),
    )


def _range_text(
    segments: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> str:
    parts: list[str] = []
    for index in range(start, end + 1):
        if index >= len(segments):
            raise EntailmentValidationError(
                "Evidence range exceeds transcript"
            )
        text = segments[index].get("text")
        if not isinstance(text, str):
            raise EntailmentValidationError(
                f"segments[{index}].text must be a string"
            )
        parts.append(text.strip())
    return " ".join(parts)


def _variable_phrases(
    candidate: FormulaCandidate,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for variable in candidate.variables:
        symbol = str(variable["symbol"])
        meaning = str(variable["meaning"])
        result[symbol] = (
            symbol,
            symbol.replace("_", " "),
            meaning,
        )
    return result


def _identifiers_in_expression(expression: str) -> set[str]:
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise EntailmentValidationError(
            f"Invalid AST node expression: {expression}"
        ) from exc
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id not in {
            "sum",
            "sqrt",
            "log",
            "exp",
            "abs",
            "min",
            "max",
        }
    }


def _has_operation_cue(operation: str, text: str) -> bool:
    cues = _OPERATION_CUES.get(operation)
    if not cues:
        return False
    normalized = _normalize(text)
    return any(_normalize(cue) in normalized for cue in cues)


def _dependency_cycle_issues(
    nodes: Sequence[NodeEntailment],
) -> list[str]:
    graph = {
        node.node_id: set(node.depends_on_node_ids)
        for node in nodes
    }
    visiting: set[str] = set()
    visited: set[str] = set()
    issues: list[str] = []

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            issues.append(f"Dependency cycle includes {node_id}")
            return
        visiting.add(node_id)
        for dependency in graph.get(node_id, set()):
            if dependency in graph:
                visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in graph:
        visit(node_id)
    return issues


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EntailmentValidationError(
            f"{field} must be a nonempty string"
        )
    return value.strip()


def _require_segment(value: Any, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise EntailmentValidationError(
            f"{field} must be a nonnegative integer"
        )
    return value


def _require_string_array(value: Any, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EntailmentValidationError(f"{field} must be an array")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _require_string(item, f"{field} item")
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result
