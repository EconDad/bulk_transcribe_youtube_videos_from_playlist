"""Per-calculation formula extraction for research pipeline v4.3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from typing import Any, Mapping, Sequence

from .calculation_inventory import CalculationItem
from .expression_ast import FormulaCandidate, ExpressionValidationError


class FormulaExtractionError(ValueError):
    """Raised when formula extraction output is invalid."""


class ExtractionDisposition(StrEnum):
    CANDIDATES_PROPOSED = "candidates_proposed"
    NON_SYMBOLIC_CALCULATION = "non_symbolic_calculation"
    INSUFFICIENT_SOURCE_DETAIL = "insufficient_source_detail"
    VISUAL_REVIEW_REQUIRED = "visual_review_required"


def allowed_extraction_dispositions(
    item: CalculationItem,
) -> tuple[ExtractionDisposition, ...]:
    """Return only dispositions permitted by the inventory item."""

    allowed = [
        ExtractionDisposition.CANDIDATES_PROPOSED,
        ExtractionDisposition.NON_SYMBOLIC_CALCULATION,
        ExtractionDisposition.INSUFFICIENT_SOURCE_DETAIL,
    ]
    if item.visual_equation_cue:
        allowed.append(ExtractionDisposition.VISUAL_REVIEW_REQUIRED)
    return tuple(allowed)


@dataclass(frozen=True, slots=True)
class FormulaExtractionResult:
    calculation_id: str
    disposition: ExtractionDisposition
    reason: str
    candidates: tuple[FormulaCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation_id": self.calculation_id,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "candidates": [item.to_dict() for item in self.candidates],
        }


def build_formula_extraction_prompt(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
) -> str:
    """Build a source-bounded, domain-neutral formula prompt."""

    selected: list[dict[str, Any]] = []
    for index in range(item.start_segment, item.end_segment + 1):
        if index >= len(segments):
            raise FormulaExtractionError(
                f"{item.calculation_id} exceeds available segments"
            )
        segment = segments[index]
        if not isinstance(segment, Mapping):
            raise FormulaExtractionError(
                f"segments[{index}] must be an object"
            )
        text = segment.get("text")
        if not isinstance(text, str):
            raise FormulaExtractionError(
                f"segments[{index}].text must be a string"
            )
        selected.append({"segment_id": index, "text": text.strip()})

    allowed_dispositions = allowed_extraction_dispositions(item)
    allowed_values = " | ".join(
        disposition.value for disposition in allowed_dispositions
    )

    response_schema = {
        "calculation_id": item.calculation_id,
        "disposition": allowed_values,
        "reason": "Source-grounded explanation.",
        "candidates": [
            {
                "calculation_id": item.calculation_id,
                "formula_id": "snake_case_identifier",
                "name": "Human-readable formula name",
                "ascii": "result = expression",
                "latex": "LaTeX expression",
                "derivation_type": "stated | derived | approximation",
                "variables": [
                    {
                        "symbol": "snake_case_symbol",
                        "meaning": "Source-grounded meaning",
                        "unit": "unit or empty string",
                    }
                ],
                "derivation_steps": ["Complete sentence."],
                "source_claims": [
                    {
                        "start_segment": item.start_segment,
                        "end_segment": item.end_segment,
                        "relationship": "Source-stated relationship",
                    }
                ],
            }
        ],
    }

    item_payload = item.to_dict()
    return (
        "Extract reusable symbolic formulas only for the identified "
        "calculation event. Use only the supplied transcript segments. "
        "Do not inject a textbook formula that is absent from the source. "
        "A derived formula is allowed only when each derivation step follows "
        "from source-stated relationships. The only allowed dispositions for "
        "this item are listed in the response schema. Choose "
        "visual_review_required only when it is explicitly listed. For every "
        "candidate, parse the ASCII formula mentally and provide exactly one "
        "variable definition for every snake_case identifier appearing anywhere "
        "in that formula, including the identifier to the left of '='. Do not "
        "define identifiers that do not appear in the ASCII formula. Return JSON "
        "only and match the schema exactly.\n\n"
        f"CALCULATION EVENT:\n{json.dumps(item_payload, indent=2)}\n\n"
        f"RESPONSE SCHEMA:\n{json.dumps(response_schema, indent=2)}\n\n"
        f"SOURCE SEGMENTS:\n{json.dumps(selected, indent=2)}"
    )


def build_formula_extraction_repair_prompt(
    *,
    item: CalculationItem,
    segments: Sequence[Mapping[str, Any]],
    invalid_payload: Mapping[str, Any],
    validation_error: str,
) -> str:
    """Build one bounded structural-repair request."""

    return (
        build_formula_extraction_prompt(item=item, segments=segments)
        + "\n\nREPAIR REQUEST:\n"
        + "The previous JSON failed deterministic validation. Correct only the "
        + "reported structural or schema defects. Preserve all source-bounded "
        + "claims that remain valid. Do not add outside formulas, quantities, "
        + "or relationships. Every ASCII identifier, including the left-hand "
        + "result, must have exactly one variable definition, with no extras. "
        + "Return a complete replacement JSON object only.\n\n"
        + f"VALIDATION ERROR:\n{validation_error}\n\n"
        + "INVALID RESPONSE:\n"
        + json.dumps(dict(invalid_payload), indent=2)
    )


def parse_formula_extraction_response(
    payload: Mapping[str, Any],
    *,
    item: CalculationItem,
) -> FormulaExtractionResult:
    """Strictly validate one per-calculation extraction response."""

    required = {"calculation_id", "disposition", "reason", "candidates"}
    if set(payload) != required:
        raise FormulaExtractionError(
            "Formula extraction response must contain exactly "
            f"{sorted(required)}"
        )

    calculation_id = _require_string(
        payload["calculation_id"], "calculation_id"
    )
    if calculation_id != item.calculation_id:
        raise FormulaExtractionError(
            "Formula extraction calculation_id does not match inventory"
        )

    try:
        disposition = ExtractionDisposition(
            _require_string(payload["disposition"], "disposition")
        )
    except ValueError as exc:
        raise FormulaExtractionError(
            "Unknown formula extraction disposition"
        ) from exc

    allowed = allowed_extraction_dispositions(item)
    if (
        disposition is ExtractionDisposition.VISUAL_REVIEW_REQUIRED
        and not item.visual_equation_cue
    ):
        raise FormulaExtractionError(
            "visual_review_required requires a visual equation cue"
        )
    if disposition not in allowed:
        raise FormulaExtractionError(
            f"{disposition.value} is not allowed for this inventory item; "
            f"allowed={[entry.value for entry in allowed]}"
        )

    reason = _require_string(payload["reason"], "reason")
    raw_candidates = payload["candidates"]
    if isinstance(raw_candidates, (str, bytes)) or not isinstance(
        raw_candidates, Sequence
    ):
        raise FormulaExtractionError("candidates must be an array")

    candidates: list[FormulaCandidate] = []
    seen_formula_ids: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, Mapping):
            raise FormulaExtractionError(
                f"candidates[{index}] must be an object"
            )
        try:
            candidate = FormulaCandidate.from_mapping(raw_candidate)
        except ExpressionValidationError as exc:
            raise FormulaExtractionError(
                f"candidates[{index}] is invalid: {exc}"
            ) from exc

        if candidate.calculation_id != item.calculation_id:
            raise FormulaExtractionError(
                f"candidates[{index}] references the wrong calculation"
            )
        if candidate.formula_id in seen_formula_ids:
            raise FormulaExtractionError(
                f"Duplicate formula_id: {candidate.formula_id}"
            )
        seen_formula_ids.add(candidate.formula_id)

        for claim in candidate.source_claims:
            start = int(claim["start_segment"])
            end = int(claim["end_segment"])
            if start < item.start_segment or end > item.end_segment:
                raise FormulaExtractionError(
                    f"{candidate.formula_id} cites outside its inventory item"
                )

        if (
            candidate.derivation_type.value == "stated_visual"
            and not item.visual_equation_cue
        ):
            raise FormulaExtractionError(
                "stated_visual candidate requires a visual equation cue"
            )

        candidates.append(candidate)

    if disposition is ExtractionDisposition.CANDIDATES_PROPOSED:
        if not candidates:
            raise FormulaExtractionError(
                "candidates_proposed requires at least one candidate"
            )
    elif candidates:
        raise FormulaExtractionError(
            f"{disposition.value} must not include candidates"
        )

    if (
        disposition is ExtractionDisposition.VISUAL_REVIEW_REQUIRED
        and not item.visual_equation_cue
    ):
        raise FormulaExtractionError(
            "visual_review_required requires a visual equation cue"
        )

    return FormulaExtractionResult(
        calculation_id=calculation_id,
        disposition=disposition,
        reason=reason,
        candidates=tuple(candidates),
    )


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormulaExtractionError(
            f"{field} must be a nonempty string"
        )
    return value.strip()
