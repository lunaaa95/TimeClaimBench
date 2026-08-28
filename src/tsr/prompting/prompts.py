"""Prompt templates for explanation generation, claim extraction, and repair."""

from __future__ import annotations


PROMPT_VERSION = "v1"

DIRECT_EXPLANATION_PROMPT = """You are a time-series analyst.
Given the following time series, describe the major temporal patterns.
Focus on trends, peaks, drops, volatility, anomalies, seasonality, and change points.
Time series:
{series}
Question:
{question}
Answer with a concise explanation."""

COT_PROMPT = """You are a time-series analyst.
Analyze the time series step by step.
First inspect the overall trend.
Then inspect local peaks and drops.
Then inspect volatility and anomalies.
Finally provide a concise explanation.
Time series:
{series}
Question:
{question}"""

STRUCTURED_PROMPT = """You are a time-series analyst.
Return a concise bullet list of grounded temporal patterns.
Only include claims that can be checked against the series.
Time series:
{series}
Question:
{question}"""

EVENT_GROUNDED_PROMPT = """You are a time-series analyst.
You are given a time series and automatically extracted temporal events.
Write an explanation using only the listed events.
Do not mention patterns that are not supported by the event list.
Cite event IDs in brackets.
Time series:
{series}
Extracted events:
{events}
Question:
{question}
Output format:
Explanation:
- [E?] ...
- [E?] ..."""

CLAIM_EXTRACTION_PROMPT = """Decompose the following time-series explanation into atomic claims.
Each claim should describe exactly one temporal pattern.
Explanation:
{explanation}
Return JSON:
[
  {{
    "claim_id": "C1",
    "text": "...",
    "claim_type": "...",
    "direction": "...",
    "interval_text": "...",
    "strength": "..."
  }}
]"""

REPAIR_PROMPT = """You are repairing an unfaithful time-series explanation.
Original explanation:
{explanation}
Claim-level verification:
{verification_results}
Supported temporal events:
{events}
Instructions:
1. Remove unsupported claims.
2. Correct wrong directions.
3. Correct wrong intervals.
4. Preserve supported claims.
5. Do not add new claims.
6. Return only the repaired explanation."""

PROMPTS = {
    "direct": DIRECT_EXPLANATION_PROMPT,
    "cot": COT_PROMPT,
    "structured": STRUCTURED_PROMPT,
    "event_grounded": EVENT_GROUNDED_PROMPT,
    "claim_extraction": CLAIM_EXTRACTION_PROMPT,
    "repair": REPAIR_PROMPT,
}
