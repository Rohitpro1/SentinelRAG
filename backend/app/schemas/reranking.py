"""
Unit 2.12 -- Reranking domain observability schemas.

RerankResult and RerankerCapabilities are ADDITIVE: they do not change
BaseReranker's frozen contract (rerank() still returns bare
list[RankedChunk], exactly as Unit 2.5 approved). Same business/
observability split as Unit 2.9 (VerifiedEvidence/VerificationDiagnostics)
and Unit 2.11 (EmbeddingResult/EmbedderHealth).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.retrieval_domain import RankedChunk


class RerankResult(BaseModel):
    """Business output remains ranked_chunks; the rest is observability metadata."""

    ranked_chunks: list[RankedChunk]
    provider: str
    model_name: str
    model_version: Optional[str] = None
    rerank_latency_ms: float = Field(ge=0.0)


class RerankerCapabilities(BaseModel):
    """
    Lightweight, CONFIGURATION-DRIVEN capability metadata (instruction 2)
    -- these values come from RerankingSettings, set based on the
    provider's documented limits. Never queried from the provider at
    runtime (no "call the API to ask what it supports" logic anywhere).
    """

    supports_batching: bool
    max_batch_size: int
    max_input_tokens: int
    model_dimensions: Optional[int] = None  # not applicable to most cross-encoders (scalar relevance output)
