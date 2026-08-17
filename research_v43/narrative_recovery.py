"""Item-level fail-closed recovery for Stage F narrative evidence."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .finalization import FinalizationError
from .narrative_localization import localize_narrative_extraction


def _clean_nested_context(message: str) -> str:
    prefix = "evidence[0] "
    return message[len(prefix):] if message.startswith(prefix) else message


def recover_narrative_extraction(
    payload: Mapping[str, Any],
    *,
    segments: Sequence[Mapping[str, Any]],
    minimum_segment: int,
    maximum_segment: int,
    on_repair: Callable[[str], None] | None = None,
    on_reject: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate model evidence independently and retain only grounded items.

    No prose is rewritten and no citation is expanded. Each item still passes
    through deterministic localization and the strict Stage F validator. An
    isolated invalid item can therefore be rejected without discarding other
    grounded items from the same chunk.
    """

    if set(payload) != {"evidence"}:
        return localize_narrative_extraction(
            payload,
            segments=segments,
            minimum_segment=minimum_segment,
            maximum_segment=maximum_segment,
            on_repair=on_repair,
        )

    raw_items = payload.get("evidence")
    if isinstance(raw_items, (str, bytes)) or not isinstance(
        raw_items, Sequence
    ):
        return localize_narrative_extraction(
            payload,
            segments=segments,
            minimum_segment=minimum_segment,
            maximum_segment=maximum_segment,
            on_repair=on_repair,
        )

    if len(raw_items) > 5:
        raise FinalizationError(
            "Narrative extraction returned more than five items"
        )

    retained: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_items):
        repairs: list[str] = []
        try:
            parsed = localize_narrative_extraction(
                {"evidence": [raw]},
                segments=segments,
                minimum_segment=minimum_segment,
                maximum_segment=maximum_segment,
                on_repair=repairs.append,
            )
        except FinalizationError as exc:
            if on_reject is not None:
                on_reject(
                    f"evidence[{index}]: "
                    f"{_clean_nested_context(str(exc))}"
                )
            continue

        if on_repair is not None:
            for repair in repairs:
                on_repair(
                    f"evidence[{index}]: "
                    f"{_clean_nested_context(repair)}"
                )
        retained.extend(parsed)

    return tuple(retained)
