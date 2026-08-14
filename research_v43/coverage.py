"""Calculation-to-formula coverage reconciliation for v4.3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

from .calculation_inventory import CalculationInventory


class CoverageValidationError(ValueError):
    """Raised when coverage inputs are structurally invalid."""


class CoverageState(StrEnum):
    FORMULA_RETAINED = "formula_retained"
    NON_SYMBOLIC_CALCULATION = "non_symbolic_calculation"
    INSUFFICIENT_SOURCE_DETAIL = "insufficient_source_detail"
    VISUAL_REVIEW_REQUIRED = "visual_review_required"
    FORMULA_REJECTED = "formula_rejected"


@dataclass(frozen=True, slots=True)
class CoverageResolution:
    calculation_id: str
    state: CoverageState
    formula_ids: tuple[str, ...]
    reason: str

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "CoverageResolution":
        required = {"calculation_id", "state", "formula_ids", "reason"}
        if set(raw) != required:
            raise CoverageValidationError(
                "Coverage resolution must contain exactly "
                f"{sorted(required)}"
            )

        calculation_id = _require_string(
            raw["calculation_id"], "calculation_id"
        )
        try:
            state = CoverageState(_require_string(raw["state"], "state"))
        except ValueError as exc:
            raise CoverageValidationError(
                "Unknown coverage state"
            ) from exc

        formula_ids = _require_formula_ids(raw["formula_ids"])
        reason = _require_string(raw["reason"], "reason")

        if state is CoverageState.FORMULA_RETAINED:
            if not formula_ids:
                raise CoverageValidationError(
                    "formula_retained requires formula_ids"
                )
        elif formula_ids:
            raise CoverageValidationError(
                f"{state.value} cannot contain formula_ids"
            )

        return cls(
            calculation_id=calculation_id,
            state=state,
            formula_ids=tuple(formula_ids),
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation_id": self.calculation_id,
            "state": self.state.value,
            "formula_ids": list(self.formula_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CoverageReport:
    passed: bool
    identified_calculations: int
    formulas_retained: int
    non_symbolic_calculations: int
    insufficient_source_detail: int
    visual_review_required: int
    formula_rejected: int
    unresolved: int
    issues: tuple[str, ...]
    resolutions: tuple[CoverageResolution, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "identified_calculations": self.identified_calculations,
            "formulas_retained": self.formulas_retained,
            "non_symbolic_calculations": self.non_symbolic_calculations,
            "insufficient_source_detail": self.insufficient_source_detail,
            "visual_review_required": self.visual_review_required,
            "formula_rejected": self.formula_rejected,
            "unresolved": self.unresolved,
            "issues": list(self.issues),
            "resolutions": [
                item.to_dict()
                for item in self.resolutions
            ],
        }



FormulaKey = tuple[str, str]


def _formula_key(
    calculation_id: str,
    formula_id: str,
) -> FormulaKey:
    return (calculation_id, formula_id)


def _index_formulas(
    formulas: Sequence[Mapping[str, Any]],
    inventory_by_id: Mapping[str, Any],
) -> dict[FormulaKey, Mapping[str, Any]]:
    formula_by_key: dict[FormulaKey, Mapping[str, Any]] = {}

    for index, formula in enumerate(formulas):
        if not isinstance(formula, Mapping):
            raise CoverageValidationError(
                f"formulas[{index}] must be an object"
            )

        formula_id = _require_string(
            formula.get("formula_id"),
            f"formulas[{index}].formula_id",
        )
        calculation_id = _require_string(
            formula.get("calculation_id"),
            f"formulas[{index}].calculation_id",
        )

        key = _formula_key(calculation_id, formula_id)
        if key in formula_by_key:
            raise CoverageValidationError(
                "Duplicate formula_id within calculation "
                f"{calculation_id}: {formula_id}"
            )

        if calculation_id not in inventory_by_id:
            raise CoverageValidationError(
                f"Formula {formula_id} references unknown "
                f"calculation {calculation_id}"
            )

        formula_by_key[key] = formula

    return formula_by_key


def reconcile_coverage(
    *,
    inventory: CalculationInventory,
    resolutions: Sequence[CoverageResolution | Mapping[str, Any]],
    formulas: Sequence[Mapping[str, Any]],
) -> CoverageReport:
    """Reconcile every inventory item with a formula or terminal state."""

    normalized_resolutions = tuple(
        item
        if isinstance(item, CoverageResolution)
        else CoverageResolution.from_mapping(item)
        for item in resolutions
    )

    inventory_by_id = {
        item.calculation_id: item
        for item in inventory.calculations
    }

    resolution_by_id: dict[str, CoverageResolution] = {}
    for resolution in normalized_resolutions:
        if resolution.calculation_id not in inventory_by_id:
            raise CoverageValidationError(
                "Coverage resolution references unknown calculation: "
                f"{resolution.calculation_id}"
            )
        if resolution.calculation_id in resolution_by_id:
            raise CoverageValidationError(
                "Duplicate coverage resolution: "
                f"{resolution.calculation_id}"
            )
        resolution_by_id[resolution.calculation_id] = resolution

    formula_by_key = _index_formulas(
        formulas,
        inventory_by_id,
    )

    issues: list[str] = []
    ordered_resolutions: list[CoverageResolution] = []

    for item in inventory.calculations:
        resolution = resolution_by_id.get(item.calculation_id)
        if resolution is None:
            issues.append(
                f"{item.calculation_id} has no coverage resolution"
            )
            continue

        ordered_resolutions.append(resolution)

        if resolution.state is CoverageState.FORMULA_RETAINED:
            for formula_id in resolution.formula_ids:
                formula = formula_by_key.get(
                    _formula_key(
                        item.calculation_id,
                        formula_id,
                    )
                )
                if formula is None:
                    other_owners = sorted(
                        calculation_id
                        for calculation_id, scoped_formula_id in formula_by_key
                        if scoped_formula_id == formula_id
                        and calculation_id != item.calculation_id
                    )
                    if len(other_owners) == 1:
                        issues.append(
                            f"Formula {formula_id} belongs to "
                            f"{other_owners[0]}, not {item.calculation_id}"
                        )
                    elif other_owners:
                        issues.append(
                            f"{item.calculation_id} references missing "
                            f"formula {formula_id}; that formula_id exists "
                            f"for calculations {', '.join(other_owners)}"
                        )
                    else:
                        issues.append(
                            f"{item.calculation_id} references missing "
                            f"formula {formula_id}"
                        )
                    continue
                if formula.get("calculation_id") != item.calculation_id:
                    issues.append(
                        f"Formula {formula_id} belongs to "
                        f"{formula.get('calculation_id')}, not "
                        f"{item.calculation_id}"
                    )

        if (
            item.formula_expected
            and resolution.state
            in {
                CoverageState.NON_SYMBOLIC_CALCULATION,
                CoverageState.INSUFFICIENT_SOURCE_DETAIL,
                CoverageState.VISUAL_REVIEW_REQUIRED,
                CoverageState.FORMULA_REJECTED,
            }
        ):
            issues.append(
                f"{item.calculation_id} expected a formula but ended as "
                f"{resolution.state.value}"
            )

        if (
            not item.formula_expected
            and resolution.state is CoverageState.FORMULA_RETAINED
        ):
            issues.append(
                f"{item.calculation_id} retained a formula even though "
                "formula_expected=false"
            )

    referenced_formula_keys = {
        _formula_key(
            resolution.calculation_id,
            formula_id,
        )
        for resolution in ordered_resolutions
        if resolution.state is CoverageState.FORMULA_RETAINED
        for formula_id in resolution.formula_ids
    }

    for calculation_id, formula_id in sorted(
        set(formula_by_key) - referenced_formula_keys
    ):
        issues.append(
            f"Formula {formula_id} for {calculation_id} is not referenced "
            "by a formula_retained resolution"
        )

    counts = {
        state: sum(
            resolution.state is state
            for resolution in ordered_resolutions
        )
        for state in CoverageState
    }

    unresolved = (
        len(inventory.calculations)
        - len(ordered_resolutions)
        + counts[CoverageState.INSUFFICIENT_SOURCE_DETAIL]
        + counts[CoverageState.VISUAL_REVIEW_REQUIRED]
        + counts[CoverageState.FORMULA_REJECTED]
    )

    return CoverageReport(
        passed=not issues,
        identified_calculations=len(inventory.calculations),
        formulas_retained=len(referenced_formula_keys),
        non_symbolic_calculations=counts[
            CoverageState.NON_SYMBOLIC_CALCULATION
        ],
        insufficient_source_detail=counts[
            CoverageState.INSUFFICIENT_SOURCE_DETAIL
        ],
        visual_review_required=counts[
            CoverageState.VISUAL_REVIEW_REQUIRED
        ],
        formula_rejected=counts[
            CoverageState.FORMULA_REJECTED
        ],
        unresolved=unresolved,
        issues=tuple(issues),
        resolutions=tuple(ordered_resolutions),
    )


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoverageValidationError(
            f"{field} must be a nonempty string"
        )
    return value.strip()


def _require_formula_ids(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CoverageValidationError(
            "formula_ids must be an array"
        )

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        formula_id = _require_string(item, "formula_ids item")
        if formula_id in seen:
            raise CoverageValidationError(
                f"Duplicate formula_id in resolution: {formula_id}"
            )
        result.append(formula_id)
        seen.add(formula_id)
    return result
