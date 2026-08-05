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


@dataclass(frozen=True, slots=True)
class InventoryChunk:
    """One bounded transcript window for inventory extraction."""

    chunk_index: int
    start_segment: int
    end_segment: int
    segments: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "start_segment": self.start_segment,
            "end_segment": self.end_segment,
            "segment_count": len(self.segments),
        }


def build_inventory_chunks(
    segments: Sequence[Mapping[str, Any]],
    *,
    chunk_segments: int = 40,
    overlap_segments: int = 6,
) -> tuple[InventoryChunk, ...]:
    """Split a transcript into bounded, overlapping segment windows."""

    if chunk_segments < 2:
        raise InventoryValidationError(
            "chunk_segments must be at least 2"
        )
    if overlap_segments < 0:
        raise InventoryValidationError(
            "overlap_segments cannot be negative"
        )
    if overlap_segments >= chunk_segments:
        raise InventoryValidationError(
            "overlap_segments must be smaller than chunk_segments"
        )
    if not segments:
        return tuple()

    chunks: list[InventoryChunk] = []
    start = 0
    chunk_index = 0
    while start < len(segments):
        stop = min(len(segments), start + chunk_segments)
        selected = tuple(segments[start:stop])
        chunks.append(
            InventoryChunk(
                chunk_index=chunk_index,
                start_segment=start,
                end_segment=stop - 1,
                segments=selected,
            )
        )
        if stop == len(segments):
            break
        start = stop - overlap_segments
        chunk_index += 1
    return tuple(chunks)


def merge_inventories(
    *,
    video_id: str,
    inventories: Sequence[CalculationInventory],
) -> CalculationInventory:
    """Merge overlapping chunk inventories and assign stable global IDs."""

    candidates: list[CalculationItem] = []
    for inventory in inventories:
        if inventory.video_id != video_id:
            raise InventoryValidationError(
                "Cannot merge inventories for different videos"
            )
        candidates.extend(inventory.calculations)

    candidates.sort(
        key=lambda item: (
            item.start_segment,
            item.end_segment,
            item.name.casefold(),
        )
    )

    merged: list[CalculationItem] = []
    for candidate in candidates:
        match_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _same_calculation(existing, candidate)
            ),
            None,
        )
        if match_index is None:
            merged.append(candidate)
        else:
            merged[match_index] = _merge_calculation_items(
                merged[match_index],
                candidate,
            )

    merged.sort(
        key=lambda item: (
            item.start_segment,
            item.end_segment,
            item.name.casefold(),
        )
    )
    renumbered = tuple(
        CalculationItem(
            calculation_id=f"CALC_{index:04d}",
            name=item.name,
            source_mode=item.source_mode,
            start_segment=item.start_segment,
            end_segment=item.end_segment,
            variables_mentioned=item.variables_mentioned,
            operations_mentioned=item.operations_mentioned,
            visual_equation_cue=item.visual_equation_cue,
            formula_expected=item.formula_expected,
            reason=item.reason,
        )
        for index, item in enumerate(merged, start=1)
    )
    return CalculationInventory(
        schema_version="1.0",
        video_id=video_id,
        calculations=renumbered,
    )


def _same_calculation(
    first: CalculationItem,
    second: CalculationItem,
) -> bool:
    if first.end_segment < second.start_segment:
        return False
    if second.end_segment < first.start_segment:
        return False

    if first.visual_equation_cue and second.visual_equation_cue:
        return True

    first_name = _token_set(first.name)
    second_name = _token_set(second.name)
    name_similarity = _jaccard(first_name, second_name)
    if name_similarity >= 0.67:
        return True

    first_variables = _normalized_set(first.variables_mentioned)
    second_variables = _normalized_set(second.variables_mentioned)
    first_operations = _normalized_set(first.operations_mentioned)
    second_operations = _normalized_set(second.operations_mentioned)
    variable_similarity = _jaccard(first_variables, second_variables)
    operation_overlap = bool(first_operations & second_operations)
    return (
        name_similarity >= 0.30
        and variable_similarity >= 0.50
        and operation_overlap
    )


def _merge_calculation_items(
    first: CalculationItem,
    second: CalculationItem,
) -> CalculationItem:
    if first.source_mode is second.source_mode:
        source_mode = first.source_mode
    else:
        source_mode = SourceMode.MIXED

    reasons = []
    for reason in (first.reason, second.reason):
        if reason not in reasons:
            reasons.append(reason)

    return CalculationItem(
        calculation_id=first.calculation_id,
        name=(
            first.name
            if len(first.name) >= len(second.name)
            else second.name
        ),
        source_mode=source_mode,
        start_segment=min(first.start_segment, second.start_segment),
        end_segment=max(first.end_segment, second.end_segment),
        variables_mentioned=tuple(
            _ordered_union(
                first.variables_mentioned,
                second.variables_mentioned,
            )
        ),
        operations_mentioned=tuple(
            _ordered_union(
                first.operations_mentioned,
                second.operations_mentioned,
            )
        ),
        visual_equation_cue=(
            first.visual_equation_cue
            or second.visual_equation_cue
        ),
        formula_expected=(
            first.formula_expected
            or second.formula_expected
        ),
        reason=" ".join(reasons),
    )


def _token_set(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1
    }


def _normalized_set(values: Sequence[str]) -> set[str]:
    return {
        " ".join(sorted(_token_set(value)))
        for value in values
        if value.strip()
    }


def _jaccard(first: set[str], second: set[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _ordered_union(
    first: Sequence[str],
    second: Sequence[str],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in (*first, *second):
        key = value.casefold().strip()
        if key and key not in seen:
            result.append(value)
            seen.add(key)
    return result
