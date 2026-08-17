"""Bounded recovery helpers for v4.3 calculation inventory generation."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .calculation_inventory import CalculationInventory, parse_inventory_response
from .model_client import ModelClientError, OllamaJsonClient


def normalize_inventory_order(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with calculation items ordered by transcript progression.

    This is a serialization repair only. Item contents and model-provided IDs are
    preserved exactly. Invalid non-array payloads are left unchanged so the
    strict parser can reject them normally.
    """

    normalized = dict(payload)
    calculations = payload.get("calculations")
    if not isinstance(calculations, list):
        return normalized
    if not all(isinstance(item, Mapping) for item in calculations):
        return normalized

    def key(item: Mapping[str, Any]) -> tuple[int, int, str]:
        start = item.get("start_segment")
        end = item.get("end_segment")
        calculation_id = item.get("calculation_id")
        return (
            start if isinstance(start, int) and not isinstance(start, bool) else 10**12,
            end if isinstance(end, int) and not isinstance(end, bool) else 10**12,
            calculation_id if isinstance(calculation_id, str) else "",
        )

    normalized["calculations"] = [dict(item) for item in sorted(calculations, key=key)]
    return normalized


def parse_inventory_response_with_order_repair(
    response_text: str,
    *,
    expected_video_id: str,
    maximum_segment: int,
) -> CalculationInventory:
    """Strictly parse inventory, repairing only list ordering when necessary."""

    try:
        return parse_inventory_response(
            response_text,
            expected_video_id=expected_video_id,
            maximum_segment=maximum_segment,
        )
    except ValueError as exc:
        if "ordered by transcript progression" not in str(exc):
            raise

    payload = json.loads(response_text)
    if not isinstance(payload, Mapping):
        return parse_inventory_response(
            response_text,
            expected_video_id=expected_video_id,
            maximum_segment=maximum_segment,
        )

    repaired = normalize_inventory_order(payload)
    return parse_inventory_response(
        json.dumps(repaired),
        expected_video_id=expected_video_id,
        maximum_segment=maximum_segment,
    )


class AdaptiveInventoryOllamaClient(OllamaJsonClient):
    """Retry one inventory generation when thinking consumes the output budget."""

    inventory_retry_floor = 3072
    inventory_retry_cap = 4096

    def complete_json(self, *, system_prompt: str, user_prompt: str, stage: str = "model", num_predict: int | None = None, think: bool | None = None):
        try:
            return super().complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage=stage,
                num_predict=num_predict,
                think=think,
            )
        except ModelClientError as exc:
            message = str(exc)
            is_inventory = stage.startswith("calculation_inventory chunk ")
            is_length_exhaustion = (
                "content is empty" in message
                and "done_reason=length" in message
            )
            if not (is_inventory and is_length_exhaustion):
                raise

            original = self.num_predict if num_predict is None else int(num_predict)
            retry_budget = min(
                self.inventory_retry_cap,
                max(self.inventory_retry_floor, original * 2),
            )
            if retry_budget <= original:
                raise

            print(
                f"RETRY {stage}: generation budget exhausted; "
                f"num_predict {original} -> {retry_budget}",
                flush=True,
            )
            return super().complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage=stage,
                num_predict=retry_budget,
                think=think,
            )
