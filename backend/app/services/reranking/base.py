"""
Unit 2.5 -- BaseReranker interface.

Owned alongside the Retrieval Domain (reranking is a retrieval-quality
concern per the frozen design). RerankingService (Unit 2.6) depends on
this interface, never on a concrete reranker.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.retrieval import RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RankedChunk]:
        """
        Return up to top_n RankedChunk, ordered best-first by rerank_score.
        Must raise RerankError on failure -- callers (RerankingService) are
        responsible for catching it and degrading gracefully, per the
        frozen design's non-fatal-reranker-failure policy.
        """
        raise NotImplementedError
