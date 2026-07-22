"""
Unit 2.4 -- BaseEmbedder interface.

Owned by the Embedding domain (sibling to Retrieval), per the frozen
dependency graph (Retrieval Domain Design, Section 5): RetrieverAgent
depends on this interface, never on a concrete embedding provider.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Always batch size 1 (Design, Section 8)."""
        raise NotImplementedError

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts at ingestion time (batch size per EmbeddingSettings)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output vector dimensionality -- callers use this to validate against EmbeddingSettings.dimensions."""
        raise NotImplementedError
