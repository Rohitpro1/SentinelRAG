"""
Unit 2.12 -- shared RerankResult construction.

Same pattern as embedding/result_builder.py (Unit 2.11): NOT a method on
BaseReranker (would change the frozen interface). A plain function any
BaseReranker implementation calls internally to build a consistent
RerankResult -- used by both DeterministicReranker and
CrossEncoderReranker, so the metadata-wrapping logic exists exactly once.
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable, Optional

from app.schemas.reranking import RerankResult
from app.schemas.retrieval_domain import RankedChunk


async def build_rerank_result(
    rerank_fn: Callable[[], Awaitable[list[RankedChunk]]],
    *,
    provider: str,
    model_name: str,
    model_version: Optional[str] = None,
) -> RerankResult:
    start = time.perf_counter()
    ranked_chunks = await rerank_fn()
    latency_ms = (time.perf_counter() - start) * 1000
    return RerankResult(
        ranked_chunks=ranked_chunks,
        provider=provider,
        model_name=model_name,
        model_version=model_version,
        rerank_latency_ms=round(latency_ms, 3),
    )
