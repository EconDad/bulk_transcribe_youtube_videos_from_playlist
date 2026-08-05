"""Safe, domain-neutral formula parsing for research pipeline v4.3."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any, Mapping, Sequence


_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_ALLOWED_FUNCTION_ARITIES: dict[str, tuple[int, int | None]] = {
    "sum": (1, None),
    "sqrt": (1, 1),
    "log": (1, 2),
    "exp": (1, 1),
    "abs": (1, 1),
    "min": (2, None),
    "max": (2, None),
}

_BINARY_OPERATION_NAMES: dict[type[ast.operator], str] = {
    ast.Add: "addition",
    ast.Sub: "subtraction",
    ast.Mult: "multiplication",
    ast.Div: "division",
    ast.Pow: "exponentiation",
}

_UNARY_OPERATION_NAMES: dict[type[ast.unaryop], str] = {
    ast.UAdd: "unary_plus",
    ast.USub: "unary_minus",
}


class ExpressionValidationError(ValueError):
    """Raised when a formula or formula candidate is invalid."""


class DerivationType(StrEnum):
    STATED = "stated"
    DERIVED = "derived"
    APPROXIMATION = "approximation"
    STATED_VISUAL = "stated_visual"


@dataclass(frozen=True, slots=True)
class ExpressionNode:
    node_id: str
    expression: str
    operation: str
    operands: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "expression": self.expression,
            "operation": self.operation,
            "operands": list(self.operands),
        }


@dataclass(frozen=True, slots=True)
class ParsedFormula:
    left_symbol: str
    canonical_ascii: str
    identifiers: frozenset[str]
    operations: tuple[ExpressionNode, ...]

    @property
    def right_identifiers(self) -> frozenset[str]:
        return frozenset(
            identifier
            for identifier in self.identifiers
            if identifier != self.left_symbol
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_symbol": self.left_symbol,
            "canonical_ascii": self.canonical_ascii,
            "identifiers": sorted(self.identifiers),
            "operations": [item.to_dict() for item in self.operations],
        }


@dataclass(frozen=True, slots=True)
class FormulaCandidate:
    calculation_id: str
    formula_id: str
    name: str
    ascii: str
    latex: str
    derivation_type: DerivationType
    variables: tuple[Mapping[str, str], ...]
    derivation_steps: tuple[str, ...]
    source_claims: tuple[Mapping[str, Any], ...]
    parsed: ParsedFormula
    visual_source: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FormulaCandidate":
        required = {
            "calculation_id",
            "formula_id",
            "name",
            "ascii",
            "latex",
            "derivation_type",
            "variables",
            "derivation_steps",
            "source_claims",
        }
        optional = {"visual_source"}
        missing = required - set(raw)
        unexpected = set(raw) - required - optional
        if missing or unexpected:
            raise ExpressionValidationError(
                "Formula candidate has invalid keys; "
                f"missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        calculation_id = _require_nonempty_string(
            raw["calculation_id"], "calculation_id"
        )
        formula_id = _require_identifier(raw["formula_id"], "formula_id")
        name = _require_nonempty_string(raw["name"], "name")
        ascii_formula = _require_nonempty_string(raw["ascii"], "ascii")
        latex = _require_nonempty_string(raw["latex"], "latex")

        try:
            derivation_type = DerivationType(
                _require_nonempty_string(
                    raw["derivation_type"], "derivation_type"
                )
            )
        except ValueError as exc:
            raise ExpressionValidationError(
                "derivation_type must be one of: "
                + ", ".join(item.value for item in DerivationType)
            ) from exc

        variables = _validate_variables(raw["variables"])
        steps = _validate_string_sequence(
            raw["derivation_steps"], "derivation_steps", minimum=1
        )
        source_claims = _validate_source_claims(raw["source_claims"])

        visual_source = raw.get("visual_source")
        if derivation_type is DerivationType.STATED_VISUAL:
            if not isinstance(visual_source, Mapping):
                raise ExpressionValidationError(
                    "stated_visual formulas require visual_source"
                )
            visual_source = dict(visual_source)
        elif visual_source is not None:
            raise ExpressionValidationError(
                "visual_source is only valid for stated_visual formulas"
            )

        parsed = parse_formula(ascii_formula)
        variable_symbols = {item["symbol"] for item in variables}
        missing_definitions = parsed.identifiers - variable_symbols
        extra_definitions = variable_symbols - parsed.identifiers
        if missing_definitions or extra_definitions:
            raise ExpressionValidationError(
                "Variable definitions do not match the expression; "
                f"missing={sorted(missing_definitions)}, "
                f"extra={sorted(extra_definitions)}"
            )

        return cls(
            calculation_id=calculation_id,
            formula_id=formula_id,
            name=name,
            ascii=ascii_formula,
            latex=latex,
            derivation_type=derivation_type,
            variables=tuple(variables),
            derivation_steps=tuple(steps),
            source_claims=tuple(source_claims),
            parsed=parsed,
            visual_source=visual_source,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "calculation_id": self.calculation_id,
            "formula_id": self.formula_id,
            "name": self.name,
            "ascii": self.ascii,
            "latex": self.latex,
            "derivation_type": self.derivation_type.value,
            "variables": [dict(item) for item in self.variables],
            "derivation_steps": list(self.derivation_steps),
            "source_claims": [dict(item) for item in self.source_claims],
            "parsed": self.parsed.to_dict(),
        }
        if self.visual_source is not None:
            result["visual_source"] = dict(self.visual_source)
        return result


def parse_formula(ascii_formula: str) -> ParsedFormula:
    """Parse one assignment using a safe mathematical expression subset."""

    if not isinstance(ascii_formula, str):
        raise ExpressionValidationError("Formula ASCII must be a string")

    normalized = ascii_formula.strip()
    if not normalized:
        raise ExpressionValidationError("Formula ASCII cannot be empty")
    if normalized.count("=") != 1:
        raise ExpressionValidationError(
            "Formula ASCII must contain exactly one '='"
        )

    left, right = (part.strip() for part in normalized.split("=", 1))
    _require_identifier(left, "left side")
    if not right:
        raise ExpressionValidationError("Formula right side cannot be empty")

    try:
        tree = ast.parse(right.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise ExpressionValidationError(
            f"Invalid mathematical expression: {exc.msg}"
        ) from exc

    identifiers: set[str] = {left}
    operations: list[ExpressionNode] = []
    operation_counter = 0

    def add_operation(
        *,
        expression: str,
        operation: str,
        operands: tuple[str, ...],
    ) -> None:
        nonlocal operation_counter
        operation_counter += 1
        operations.append(
            ExpressionNode(
                node_id=f"NODE_{operation_counter:04d}",
                expression=expression,
                operation=operation,
                operands=operands,
            )
        )

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            visit(node.body)
            return

        if isinstance(node, ast.Name):
            _require_identifier(node.id, "variable")
            identifiers.add(node.id)
            return

        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ExpressionValidationError(
                    "Only integer and floating-point constants are allowed"
                )
            return

        if isinstance(node, ast.BinOp):
            operation_name = _BINARY_OPERATION_NAMES.get(type(node.op))
            if operation_name is None:
                raise ExpressionValidationError(
                    f"Unsupported binary operation: {type(node.op).__name__}"
                )
            visit(node.left)
            visit(node.right)
            add_operation(
                expression=_canonical_expression(node),
                operation=operation_name,
                operands=(
                    _canonical_expression(node.left),
                    _canonical_expression(node.right),
                ),
            )
            return

        if isinstance(node, ast.UnaryOp):
            operation_name = _UNARY_OPERATION_NAMES.get(type(node.op))
            if operation_name is None:
                raise ExpressionValidationError(
                    f"Unsupported unary operation: {type(node.op).__name__}"
                )
            visit(node.operand)
            add_operation(
                expression=_canonical_expression(node),
                operation=operation_name,
                operands=(_canonical_expression(node.operand),),
            )
            return

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionValidationError(
                    "Only direct calls to whitelisted functions are allowed"
                )
            function_name = node.func.id
            if function_name not in _ALLOWED_FUNCTION_ARITIES:
                raise ExpressionValidationError(
                    f"Function is not allowed: {function_name}"
                )
            if node.keywords:
                raise ExpressionValidationError(
                    "Keyword arguments are not allowed"
                )
            minimum, maximum = _ALLOWED_FUNCTION_ARITIES[function_name]
            argument_count = len(node.args)
            if argument_count < minimum or (
                maximum is not None and argument_count > maximum
            ):
                if maximum is None:
                    expected = f"at least {minimum}"
                elif minimum == maximum:
                    expected = str(minimum)
                else:
                    expected = f"{minimum} to {maximum}"
                raise ExpressionValidationError(
                    f"{function_name} expects {expected} argument(s), "
                    f"received {argument_count}"
                )
            for argument in node.args:
                visit(argument)
            add_operation(
                expression=_canonical_expression(node),
                operation=f"function:{function_name}",
                operands=tuple(
                    _canonical_expression(argument)
                    for argument in node.args
                ),
            )
            return

        raise ExpressionValidationError(
            f"Unsupported syntax: {type(node).__name__}"
        )

    visit(tree)

    canonical_right = _canonical_expression(tree.body)
    return ParsedFormula(
        left_symbol=left,
        canonical_ascii=f"{left} = {canonical_right}",
        identifiers=frozenset(identifiers),
        operations=tuple(operations),
    )


def _canonical_expression(node: ast.AST) -> str:
    return ast.unparse(node).replace("**", "^")


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ExpressionValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ExpressionValidationError(
            f"{field} must be one snake_case identifier"
        )
    return normalized


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExpressionValidationError(
            f"{field} must be a nonempty string"
        )
    return value.strip()


def _validate_string_sequence(
    value: Any,
    field: str,
    *,
    minimum: int,
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExpressionValidationError(f"{field} must be an array")
    result = [
        _require_nonempty_string(item, f"{field} item")
        for item in value
    ]
    if len(result) < minimum:
        raise ExpressionValidationError(
            f"{field} must contain at least {minimum} item(s)"
        )
    return result


def _validate_variables(value: Any) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExpressionValidationError("variables must be an array")

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ExpressionValidationError(
                f"variables[{index}] must be an object"
            )
        required = {"symbol", "meaning", "unit"}
        if set(item) != required:
            raise ExpressionValidationError(
                f"variables[{index}] must contain exactly {sorted(required)}"
            )
        symbol = _require_identifier(
            item["symbol"], f"variables[{index}].symbol"
        )
        if symbol in seen:
            raise ExpressionValidationError(
                f"Duplicate variable definition: {symbol}"
            )
        seen.add(symbol)
        unit = item["unit"]
        if not isinstance(unit, str):
            raise ExpressionValidationError(
                f"variables[{index}].unit must be a string"
            )
        result.append(
            {
                "symbol": symbol,
                "meaning": _require_nonempty_string(
                    item["meaning"], f"variables[{index}].meaning"
                ),
                "unit": unit.strip(),
            }
        )
    if not result:
        raise ExpressionValidationError("variables must not be empty")
    return result


def _validate_source_claims(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExpressionValidationError("source_claims must be an array")

    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ExpressionValidationError(
                f"source_claims[{index}] must be an object"
            )
        required = {"start_segment", "end_segment", "relationship"}
        if set(item) != required:
            raise ExpressionValidationError(
                f"source_claims[{index}] must contain exactly "
                f"{sorted(required)}"
            )
        start = item["start_segment"]
        end = item["end_segment"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ExpressionValidationError(
                f"source_claims[{index}] has invalid segment range"
            )
        result.append(
            {
                "start_segment": start,
                "end_segment": end,
                "relationship": _require_nonempty_string(
                    item["relationship"],
                    f"source_claims[{index}].relationship",
                ),
            }
        )
    if not result:
        raise ExpressionValidationError("source_claims must not be empty")
    return result
