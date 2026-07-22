from __future__ import annotations

from app.providers.base.embedding_provider import BaseEmbeddingProvider
from app.services.embedding.deterministic import DeterministicEmbedder


class DeterministicEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic embedding provider wrapping DeterministicEmbedder.
    Used for local testing and CI.
    """

    def __init__(self, dimensions: int = 16) -> None:
        self._embedder = DeterministicEmbedder(dimensions=dimensions)

    async def embed_query(self, text: str) -> list[float]:
        return await self._embedder.embed_query(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._embedder.embed_batch(texts)

    @property
    def dimensions(self) -> int:
        return self._embedder.dimensions
