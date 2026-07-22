from __future__ import annotations

from app.providers.base.reranker_provider import BaseRerankerProvider
from app.schemas.retrieval import RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk
from app.services.reranking.deterministic import DeterministicReranker


class DeterministicRerankerProvider(BaseRerankerProvider):
    """
    Deterministic reranker provider wrapping DeterministicReranker.
    Used for local testing and CI.
    """

    def __init__(self) -> None:
        self._reranker = DeterministicReranker()

    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RankedChunk]:
        return await self._reranker.rerank(query, candidates, top_n)
