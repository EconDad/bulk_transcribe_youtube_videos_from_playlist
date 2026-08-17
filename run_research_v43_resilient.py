#!/usr/bin/env python3
"""Resilient v4.3 diagnostic entrypoint for Stage F acceptance.

This keeps the frozen core diagnostic runner intact while adding two bounded
recoveries proven useful by real-video integration:
- retry inventory generation once when thinking exhausts the response budget;
- repair calculation-array ordering without changing item contents.
"""

from __future__ import annotations

import run_research_v43 as base

from research_v43.inventory_recovery import (
    AdaptiveInventoryOllamaClient,
    parse_inventory_response_with_order_repair,
)


def main(argv=None) -> int:
    base.OllamaJsonClient = AdaptiveInventoryOllamaClient
    base.parse_inventory_response = parse_inventory_response_with_order_repair
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
