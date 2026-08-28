"""Claim and verification result schemas."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class Claim(BaseModel):
    claim_id: str
    text: str
    claim_type: str
    direction: Optional[str] = None
    interval_text: Optional[str] = None
    strength: Optional[str] = None
    numeric_value: Optional[float] = None
    label: Optional[str] = None
    support_events: List[str] = Field(default_factory=list)
    support_interval: Optional[Tuple[int, int]] = None
    error_type: Optional[str] = None


class VerificationResult(BaseModel):
    claim_id: str
    label: Literal["supported", "partial", "unsupported"]
    support_events: List[str] = Field(default_factory=list)
    support_interval: Optional[Tuple[int, int]] = None
    score: float
    error_type: Optional[str] = None
    rationale: Optional[str] = None
