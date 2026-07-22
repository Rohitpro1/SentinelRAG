"""
Unit 2.11 -- shared EmbeddingResult construction.

Deliberately NOT a method added to BaseEmbedder (that would change the
frozen interface, Unit 2.4). Instead, this is a plain function any
BaseEmbedder implementation can call internally to build a consistent
EmbeddingResult -- used by both DeterministicEmbedder and OpenAIEmbedder
below, so the metadata-wrapping logic exists exactly once.
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable, Optional

from app.schemas.embedding import EmbeddingResult


async def build_embedding_result(
    embed_fn: Callable[[], Awaitable[list[float]]],
    *,
    provider: str,
    model_name: str,
    model_version: Optional[str] = None,
) -> EmbeddingResult:
    start = time.perf_counter()
    vector = await embed_fn()
    latency_ms = (time.perf_counter() - start) * 1000
    return EmbeddingResult(
        vector=vector,
        provider=provider,
        model_name=model_name,
        embedding_dimensions=len(vector),
        generation_latency_ms=round(latency_ms, 3),
        model_version=model_version,
    )
