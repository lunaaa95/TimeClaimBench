"""Token and cost accounting for LLM experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

from tsr.utils.io import read_yaml


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer fallback.

    Provider usage metadata is preferred. This estimate is only used for dummy
    calls or old cache entries without token counts.
    """

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def load_model_prices(path: str | Path) -> Dict[str, dict]:
    data = read_yaml(path)
    return data.get("models", {})


def lookup_price(model: str, prices: Optional[Dict[str, dict]]) -> dict:
    if not prices:
        return {}
    return dict(prices.get(model, {}))


def estimate_cost_usd(input_tokens: int, output_tokens: int, price: dict) -> float:
    input_rate = float(price.get("input_per_million", 0.0))
    output_rate = float(price.get("output_per_million", 0.0))
    return (input_tokens / 1_000_000.0) * input_rate + (output_tokens / 1_000_000.0) * output_rate


def make_usage_record(
    prompt: str,
    response: str,
    model: str,
    prices: Optional[Dict[str, dict]] = None,
    provider_usage: Optional[Dict[str, Any]] = None,
) -> dict:
    provider_usage = provider_usage or {}
    input_tokens = provider_usage.get("input_tokens") or provider_usage.get("prompt_tokens")
    output_tokens = provider_usage.get("output_tokens") or provider_usage.get("completion_tokens")
    total_tokens = provider_usage.get("total_tokens")
    estimated = False
    if input_tokens is None:
        input_tokens = estimate_tokens(prompt)
        estimated = True
    if output_tokens is None:
        output_tokens = estimate_tokens(response)
        estimated = True
    if total_tokens is None:
        total_tokens = int(input_tokens) + int(output_tokens)
    price = lookup_price(model, prices)
    cost = estimate_cost_usd(int(input_tokens), int(output_tokens), price)
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
        "estimated_tokens": estimated,
        "input_per_million_usd": price.get("input_per_million"),
        "output_per_million_usd": price.get("output_per_million"),
        "estimated_cost_usd": cost,
        "pricing_source": price.get("source"),
    }
