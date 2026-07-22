"""
Unit 2.11 -- Embedding domain observability schemas.

EmbeddingResult and EmbedderHealth are ADDITIVE: they do not change
BaseEmbedder's frozen contract (embed_query/embed_batch/dimensions still
return bare vectors, exactly as Unit 2.4 approved). They are produced by
a separate, optional capability (see result_builder.py) that any
BaseEmbedder implementation may offer alongside the required interface.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EmbeddingResult(BaseModel):
    """
    Business output remains the vector; everything else here is
    observability metadata for dashboards/telemetry -- same
    business/observability split established in Unit 2.9's
    VerifiedEvidence / VerificationDiagnostics.
    """

    vector: list[float]
    provider: str
    model_name: str
    embedding_dimensions: int
    generation_latency_ms: float = Field(ge=0.0)
    model_version: Optional[str] = None


class EmbedderHealthState(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class EmbedderHealth(BaseModel):
    """Lightweight -- current state plus an optional human-readable reason."""

    state: EmbedderHealthState
    detail: Optional[str] = None
