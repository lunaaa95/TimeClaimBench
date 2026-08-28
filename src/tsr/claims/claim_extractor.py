"""Rule-based and LLM-backed claim extraction."""

from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional, Tuple

from tsr.claims.claim_schema import Claim


CLAIM_PATTERNS = [
    ("sharp_drop", [r"sharp drop", r"plunge", r"sudden fall", r"steep decline", r"sharp decline"]),
    ("sharp_rise", [r"sharp rise", r"spike upward", r"sudden increase", r"steep increase", r"jumps?"]),
    ("upward_trend", [r"upward", r"increase", r"rise", r"rising", r"growing", r"positive trend"]),
    ("downward_trend", [r"downward", r"decrease", r"decline", r"falling", r"negative trend"]),
    ("local_peak", [r"peak", r"maximum", r"highest point", r"local high"]),
    ("local_trough", [r"trough", r"minimum", r"lowest point", r"local low"]),
    ("volatility_shift", [r"volatil", r"fluctuation", r"unstable", r"variability"]),
    ("seasonality", [r"seasonal", r"periodic", r"repeating", r"cycle", r"cyclic"]),
    ("anomaly", [r"anomaly", r"outlier", r"unusual", r"abnormal"]),
    ("change_point", [r"change point", r"structural break", r"regime change", r"level changes?", r"shift"]),
    ("causal_claim", [r"because", r"causes?", r"leads? to", r"due to", r"drives?"]),
    ("chronology_claim", [r"foreshadows?", r"predicts?", r"before the future", r"future"]),
]


def split_claim_text(explanation: str) -> List[str]:
    text = explanation.strip()
    text = re.sub(r"^\s*Explanation:\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s*", "", text, flags=re.MULTILINE)
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
    claims = []
    for part in parts:
        cleaned = part.strip(" \t\r\n-")
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"(?:\[[A-Za-z0-9_-]+\]\s*,?\s*)+", "", cleaned)
        cleaned = cleaned.strip(" \t\r\n-,:")
        if cleaned:
            claims.append(cleaned)
    return claims


def infer_claim_type(text: str) -> str:
    lowered = text.lower()
    for claim_type, patterns in CLAIM_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, lowered):
                return claim_type
    return "unknown"


def infer_direction(text: str, claim_type: str) -> Optional[str]:
    lowered = text.lower()
    if claim_type in {"upward_trend", "sharp_rise"}:
        return "up"
    if claim_type in {"downward_trend", "sharp_drop"}:
        return "down"
    if "increase" in lowered or "more volatile" in lowered:
        return "increase"
    if "decrease" in lowered or "less volatile" in lowered:
        return "decrease"
    if "upward" in lowered or "higher" in lowered:
        return "up"
    if "downward" in lowered or "lower" in lowered:
        return "down"
    return None


def infer_strength(text: str) -> Optional[str]:
    lowered = text.lower()
    if re.search(r"\b(weak|slight|mild|small)\b", lowered):
        return "weak"
    if re.search(r"\b(moderate|clear|noticeable)\b", lowered):
        return "moderate"
    if re.search(r"\b(strong|sharp|steep|dramatic|large|major)\b", lowered):
        return "strong"
    return None


def parse_interval_text(text: str, series_length: Optional[int] = None) -> Optional[Tuple[int, int]]:
    lowered = text.lower()
    explicit_patterns = [
        r"\bt\s*=?\s*(\d+)\s*(?:to|-|through|and)\s*t?\s*=?\s*(\d+)\b",
        r"\bindex\s*(\d+)\s*(?:to|-|through|and)\s*(?:index\s*)?(\d+)\b",
        r"\bindices\s*(\d+)\s*(?:to|-|through|and)\s*(\d+)\b",
        r"\bfrom\s+(?:index|indices)\s*(\d+)\s*(?:to|-|through|and|continuing through)\s*(?:index\s*)?(\d+)\b",
        r"\bstarting\s+(?:at|around)\s+(?:index|t)\s*=?\s*(\d+).*?\b(?:to|through|until|continuing through)\s+(?:index|t)?\s*=?\s*(\d+)\b",
    ]
    explicit = None
    for pattern in explicit_patterns:
        explicit = re.search(pattern, lowered)
        if explicit:
            break
    if explicit:
        a = int(explicit.group(1))
        b = int(explicit.group(2))
        return (min(a, b), max(a, b))

    single = re.search(r"\b(?:at|around|near)\s+(?:t|index)\s*=?\s*(\d+)\b", lowered)
    if single:
        t = int(single.group(1))
        return (t, t)

    if series_length is None:
        return None

    n = max(1, series_length)
    start_to = re.search(r"\bfrom\s+(?:the\s+)?(?:start|beginning)\s+(?:to|through|until)\s+(?:index|t)\s*=?\s*(\d+)\b", lowered)
    if start_to:
        return (0, min(int(start_to.group(1)), n - 1))

    after = re.search(r"\bafter\s+(?:t|index)\s*=?\s*(\d+)\b", lowered)
    if after:
        t = min(int(after.group(1)), n - 1)
        return (t, n - 1)

    before = re.search(r"\bbefore\s+(?:t|index)\s*=?\s*(\d+)\b", lowered)
    if before:
        t = min(int(before.group(1)), n - 1)
        return (0, t)

    if re.search(r"\bfirst half\b", lowered):
        return (0, max(0, n // 2 - 1))
    if re.search(r"\bsecond half\b", lowered):
        return (n // 2, n - 1)
    if re.search(r"\b(beginning|early|initial)\b", lowered):
        return (0, max(0, n // 3 - 1))
    if re.search(r"\b(middle|mid)\b", lowered):
        return (n // 3, max(n // 3, 2 * n // 3 - 1))
    if re.search(r"\b(late|end|final)\b", lowered):
        return (2 * n // 3, n - 1)

    return None


def extract_numeric_value(text: str) -> Optional[float]:
    matches = re.findall(r"(?<!t=)(?<!t)\b-?\d+(?:\.\d+)?\b", text.lower())
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def rule_extract_claims(explanation: str, series_length: Optional[int] = None) -> List[Claim]:
    claims: List[Claim] = []
    for i, text in enumerate(split_claim_text(explanation), start=1):
        claim_type = infer_claim_type(text)
        claims.append(
            Claim(
                claim_id=f"C{i}",
                text=text,
                claim_type=claim_type,
                direction=infer_direction(text, claim_type),
                interval_text=text if parse_interval_text(text, series_length) is not None else None,
                strength=infer_strength(text),
                numeric_value=extract_numeric_value(text),
                support_interval=parse_interval_text(text, series_length),
            )
        )
    return claims


def _validate_claims(data: Iterable[dict]) -> List[Claim]:
    claims = []
    for i, item in enumerate(data, start=1):
        item = dict(item)
        item.setdefault("claim_id", f"C{i}")
        item.setdefault("claim_type", infer_claim_type(item.get("text", "")))
        claims.append(Claim(**item))
    return claims


def extract_claims(explanation: str, mode: str = "rule", llm_response: Optional[str] = None) -> List[Claim]:
    """Extract atomic claims.

    The LLM path expects strict JSON and falls back to the rule parser on failure.
    """

    if mode == "rule":
        return rule_extract_claims(explanation)
    if mode == "llm" and llm_response:
        try:
            data = json.loads(llm_response)
            if isinstance(data, list):
                return _validate_claims(data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return rule_extract_claims(explanation)
