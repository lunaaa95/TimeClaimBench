"""Serializable schemas for time-series samples and events."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class Event(BaseModel):
    event_id: str
    type: str
    start: int
    end: int
    score: float
    attributes: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TimeSeriesMetadata(BaseModel):
    domain: str = "synthetic"
    length: int
    pattern_types: List[str] = Field(default_factory=list)
    decision_time: Optional[int] = None
    seed: Optional[int] = None
    split: Optional[str] = None
    source: Optional[str] = None
    source_file: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    timestamps: List[str] = Field(default_factory=list)
    original_values: List[float] = Field(default_factory=list)
    original_mean: Optional[float] = None
    original_std: Optional[float] = None
    label_source: Optional[str] = None


class ReferenceClaim(BaseModel):
    claim_id: str
    text: str
    claim_type: str
    label: str = "supported"
    support_events: List[str] = Field(default_factory=list)
    support_interval: Optional[Tuple[int, int]] = None
    error_type: Optional[str] = None


class TimeSeriesSample(BaseModel):
    id: str
    series: List[float]
    metadata: TimeSeriesMetadata
    events: List[Event] = Field(default_factory=list)
    question: str = "Describe the major temporal patterns in the series."
    reference_claims: List[ReferenceClaim] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
