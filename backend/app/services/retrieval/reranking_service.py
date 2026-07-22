"""
Unit 2.6 -- RerankingService.

Owns: calling BaseReranker with a timeout, and -- per the frozen design's
explicit non-fatal-reranker-failure policy (Retrieval Domain Design,
Section 1) -- degrading to similarity-ranked results with rerank_score=None
on ANY failure (timeout or exception) rather than propagating an error.
This is the one place in the retrieval chain that intentionally never
raises to its caller.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.core.logging import get_logger, log_event
from app.core.settings.retrieval import RetrievalSettings
from app.schemas.retrieval import RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk
from app.services.reranking.base import BaseReranker


class RerankingService:
    def __init__(self, reranker: BaseReranker, settings: RetrievalSettings, logger: Optional[logging.Logger] = None):
        self._reranker = reranker
        self._settings = settings
        self._logger = logger or get_logger(__name__)

    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RankedChunk]:
        timeout_s = self._settings.rerank_timeout_ms / 1000
        try:
            return await asyncio.wait_for(self._reranker.rerank(query, candidates, top_n), timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 -- intentional: this stage must never raise
            log_event(self._logger, "reranker_degraded", level=logging.WARNING, error=str(exc))
            return self._degrade(candidates, top_n)

    @staticmethod
    def _degrade(candidates: list[RetrievedChunk], top_n: int) -> list[RankedChunk]:
        ordered = sorted(candidates, key=lambda rc: rc.similarity_score, reverse=True)[:top_n]
        return [RankedChunk(retrieved_chunk=rc, rerank_score=None, rank=i) for i, rc in enumerate(ordered)]
