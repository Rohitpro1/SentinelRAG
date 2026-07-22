"""
Unit 2.13 -- shared NLIResult construction. Same pattern as
embedding/result_builder.py and reranking/result_builder.py.
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable, Optional

from app.schemas.nli import NLIResult
from app.schemas.retrieval import NLIRelation


async def build_nli_result(
    verify_fn: Callable[[], Awaitable[tuple[NLIRelation, float]]],
    *,
    provider: str,
    model_name: str,
    model_version: Optional[str] = None,
) -> NLIResult:
    start = time.perf_counter()
    label, confidence = await verify_fn()
    latency_ms = (time.perf_counter() - start) * 1000
    return NLIResult(
        label=label, provider=provider, model_name=model_name, model_version=model_version,
        latency_ms=round(latency_ms, 3), confidence=confidence,
    )
