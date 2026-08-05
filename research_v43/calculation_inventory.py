"""Calculation-inventory schema and response validation for v4.3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
import re
from typing import Any, Mapping, Sequence


_CALCULATION_ID_RE = re.compile(r"^CALC_[0-9]{4}$")


class InventoryValidationError(ValueError):
    """Raised when an inventory or model response is invalid."""


class SourceMode(StrEnum):
    SPOKEN = "spoken"
    VISUAL_CUE = "visual_cue"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class CalculationItem:
    calculation_id: str
    name: str
    source_mode: SourceMode
    start_segment: int
    end_segment: int
    variables_mentioned: tuple[str, ...]
    operations_mentioned: tuple[str, ...]
    visual_equation_cue: bool
    formula_expected: bool
    reason: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CalculationItem":
        required = {
            "calculation_id",
            "name",
            "source_mode",
            "start_segment",
            "end_segment",
            "variables_mentioned",
            "operations_mentioned",
            "visual_equation_cue",
            "formula_expected",
            "reason",
        }
        if set(raw) != required:
            raise InventoryValidationError(
                "Calculation item must contain exactly "
                f"{sorted(required)}"
            )

        calculation_id = _require_string(
            raw["calculation_id"], "calculation_id"
        )
        if not _CALCULATION_ID_RE.fullmatch(calculation_id):
            raise InventoryValidationError(
                "calculation_id must match CALC_0001"
            )

        try:
            source_mode = SourceMode(
                _require_string(raw["source_mode"], "source_mode")
            )
        except ValueError as exc:
            raise InventoryValidationError(
                "source_mode must be spoken, visual_cue, or mixed"
            ) from exc

        start = _require_segment(raw["start_segment"], "start_segment")
        end = _require_segment(raw["end_segment"], "end_segment")
        if end < start:
            raise InventoryValidationError(
                "end_segment cannot precede start_segment"
            )

        visual_equation_cue = _require_bool(
            raw["visual_equation_cue"], "visual_equation_cue"
        )
        formula_expected = _require_bool(
            raw["formula_expected"], "formula_expected"
        )

        if (
            source_mode is SourceMode.VISUAL_CUE
            and not visual_equation_cue
        ):
            raise InventoryValidationError(
                "visual_cue source_mode requires visual_equation_cue=true"
            )

        return cls(
            calculation_id=calculation_id,
            name=_require_string(raw["name"], "name"),
            source_mode=source_mode,
            start_segment=start,
            end_segment=end,
            variables_mentioned=tuple(
                _require_string_array(
                    raw["variables_mentioned"], "variables_mentioned"
                )
            ),
            operations_mentioned=tuple(
                _require_string_array(
                    raw["operations_mentioned"], "operations_mentioned"
                )
            ),
            visual_equation_cue=visual_equation_cue,
            formula_expected=formula_expected,
            reason=_require_string(raw["reason"], "reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation_id": self.calculation_id,
            "name": self.name,
            "source_mode": self.source_mode.value,
            "start_segment": self.start_segment,
            "end_segment": self.end_segment,
            "variables_mentioned": list(self.variables_mentioned),
            "operations_mentioned": list(self.operations_mentioned),
            "visual_equation_cue": self.visual_equation_cue,
            "formula_expected": self.formula_expected,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CalculationInventory:
    schema_version: str
    video_id: str
    calculations: tuple[CalculationItem, ...]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "CalculationInventory":
        required = {"schema_version", "video_id", "calculations"}
        if set(raw) != required:
            raise InventoryValidationError(
                "Inventory must contain exactly "
                f"{sorted(required)}"
            )

        schema_version = _require_string(
            raw["schema_version"], "schema_version"
        )
        if schema_version != "1.0":
            raise InventoryValidationError(
                "Unsupported inventory schema_version"
            )

        video_id = _require_string(raw["video_id"], "video_id")
        raw_items = raw["calculations"]
        if isinstance(raw_items, (str, bytes)) or not isinstance(
            raw_items, Sequence
        ):
            raise InventoryValidationError(
                "calculations must be an array"
            )

        items: list[CalculationItem] = []
        for index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, Mapping):
                raise InventoryValidationError(
                    f"calculations[{index}] must be an object"
                )
            items.append(CalculationItem.from_mapping(raw_item))

        seen: set[str] = set()
        for item in items:
            if item.calculation_id in seen:
                raise InventoryValidationError(
                    f"Duplicate calculation_id: {item.calculation_id}"
                )
            seen.add(item.calculation_id)

        ordered = sorted(
            items,
            key=lambda item: (
                item.start_segment,
                item.end_segment,
                item.calculation_id,
            ),
        )
        if ordered != items:
            raise InventoryValidationError(
                "calculations must be ordered by transcript progression"
            )

        return cls(
            schema_version=schema_version,
            video_id=video_id,
            calculations=tuple(items),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "calculations": [
                item.to_dict()
                for item in self.calculations
            ],
        }


def parse_inventory_response(
    response_text: str,
    *,
    expected_video_id: str,
    maximum_segment: int,
) -> CalculationInventory:
    """Parse and validate the model's strict JSON response."""

    if not isinstance(response_text, str):
        raise InventoryValidationError(
            "Inventory response must be text"
        )
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise InventoryValidationError(
            f"Inventory response is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise InventoryValidationError(
            "Inventory response must be a JSON object"
        )

    inventory = CalculationInventory.from_mapping(payload)
    if inventory.video_id != expected_video_id:
        raise InventoryValidationError(
            "Inventory video_id does not match the source video"
        )
    for item in inventory.calculations:
        if item.end_segment > maximum_segment:
            raise InventoryValidationError(
                f"{item.calculation_id} exceeds the source segment range"
            )
    return inventory


def build_inventory_prompt(
    *,
    video_id: str,
    segments: Sequence[Mapping[str, Any]],
) -> str:
    """Construct a domain-neutral calculation-discovery prompt."""

    if not isinstance(video_id, str) or not video_id.strip():
        raise InventoryValidationError("video_id cannot be empty")

    normalized_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise InventoryValidationError(
                f"segments[{index}] must be an object"
            )
        segment_id = segment.get("segment_id", index)
        text = segment.get("text")
        if (
            isinstance(segment_id, bool)
            or not isinstance(segment_id, int)
            or segment_id < 0
        ):
            raise InventoryValidationError(
                f"segments[{index}].segment_id is invalid"
            )
        if not isinstance(text, str) or not text.strip():
            raise InventoryValidationError(
                f"segments[{index}].text must be nonempty"
            )
        normalized_segments.append(
            {
                "segment_id": segment_id,
                "text": text.strip(),
            }
        )

    schema = {
        "schema_version": "1.0",
        "video_id": video_id,
        "calculations": [
            {
                "calculation_id": "CALC_0001",
                "name": "Human-readable calculation name",
                "source_mode": "spoken | visual_cue | mixed",
                "start_segment": 0,
                "end_segment": 0,
                "variables_mentioned": [],
                "operations_mentioned": [],
                "visual_equation_cue": False,
                "formula_expected": True,
                "reason": "Source-grounded reason.",
            }
        ],
    }

    return (
        "Identify every calculation event in the transcript segment set. "
        "A calculation event requires an arithmetic relationship, a reusable "
        "procedure, an equation or formula cue, or a numerical transformation. "
        "Do not invent textbook formulas. Do not classify ordinary conceptual "
        "contrasts such as 'the difference between two strategies' as "
        "arithmetic. Return JSON only, matching this schema exactly.\n\n"
        f"SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
        "TRANSCRIPT SEGMENTS:\n"
        f"{json.dumps(normalized_segments, indent=2)}"
    )


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InventoryValidationError(
            f"{field} must be a nonempty string"
        )
    return value.strip()


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise InventoryValidationError(
            f"{field} must be a boolean"
        )
    return value


def _require_segment(value: Any, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise InventoryValidationError(
            f"{field} must be a nonnegative integer"
        )
    return value


def _require_string_array(
    value: Any,
    field: str,
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InventoryValidationError(f"{field} must be an array")

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _require_string(item, f"{field} item")
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result
