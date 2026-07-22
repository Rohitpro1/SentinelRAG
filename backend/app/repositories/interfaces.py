"""
Unit 2.2 — Repository interfaces (ABCs only, no real implementations).

Per the frozen Retrieval Domain Design (Section 4) and the reaffirmed
independence rule (Addendum, Section 6): these interfaces live in the
domain layer. Concrete implementations (real Qdrant, real Postgres, real
Redis — Units 2.10+) will depend on these ABCs; these ABCs never depend on
any concrete implementation or infrastructure library. Note there is no
`import qdrant_client` / `import redis` / `import asyncpg` anywhere in this
file, by design.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.schemas.retrieval import Chunk, RetrievedChunk
from app.schemas.retrieval_domain import SearchResponse


class VectorRepository(ABC):
    """The only interface permitted to represent access to the vector store."""

    @abstractmethod
    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Write/update chunks and their embeddings. len(chunks) == len(embeddings)."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self, query_embedding: list[float], top_k: int, document_filter: Optional[dict] = None
    ) -> list[RetrievedChunk]:
        """Return up to top_k RetrievedChunk, similarity-scored, optionally filtered."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, document_id: str) -> None:
        """Remove all chunks belonging to document_id."""
        raise NotImplementedError


class MetadataRepository(ABC):
    """Document-level metadata, separate from chunk-level vector storage."""

    @abstractmethod
    async def get_document_metadata(self, document_id: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    async def save_document_metadata(self, document_id: str, metadata: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_ingestion_status(self, document_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def find_by_fingerprint(self, fingerprint: str) -> Optional[str]:
        """
        Returns an existing document_id if one with this SHA-256 fingerprint
        already exists (Architecture Enhancements, Section 4 — duplicate
        detection), else None.
        """
        raise NotImplementedError


class CacheRepository(ABC):
    """Semantic cache for SearchResponse, keyed by a caller-derived cache key."""

    @abstractmethod
    async def get(self, cache_key: str) -> Optional[SearchResponse]:
        raise NotImplementedError

    @abstractmethod
    async def set(self, cache_key: str, response: SearchResponse, ttl_seconds: int) -> None:
        raise NotImplementedError

    @abstractmethod
    async def invalidate(self, document_id: str) -> None:
        """
        Invalidate any cached SearchResponse that references document_id.
        Required whenever a document is re-ingested or deleted, so stale
        cached results referencing changed/removed chunks are never served.
        """
        raise NotImplementedError


class FeedbackRepository(ABC):
    """
    Deliberately not a dependency of RetrieverAgent (Retrieval Domain
    Design, Section 4) — written by the API layer after a query completes,
    never read on the query path.
    """

    @abstractmethod
    async def record_feedback(self, query_id: str, rating: int, comment: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_feedback_for_query(self, query_id: str) -> list[dict]:
        raise NotImplementedError
