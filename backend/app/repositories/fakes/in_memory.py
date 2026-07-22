"""
Unit 2.2 — In-memory fake implementations of the repository interfaces.

Purpose: let every later Retrieval Domain unit (2.4 onward: embedder,
reranker, RetrieverAgent, VerificationAgent) be built and tested WITHOUT
Qdrant/Postgres/Redis running, per the fake-first sequencing in the
approved Milestone 2 plan. These are test/dev doubles, not a production
storage backend -- no persistence across process restarts, no concurrency
guarantees beyond what asyncio's single-threaded event loop gives for free.
"""
from __future__ import annotations

import math
from typing import Optional

from app.repositories.interfaces import CacheRepository, FeedbackRepository, MetadataRepository, VectorRepository
from app.schemas.retrieval import Chunk, RetrievedChunk
from app.schemas.retrieval_domain import SearchResponse


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    # Clamp to [0, 1]: RetrievedChunk.similarity_score requires this range,
    # but raw cosine similarity can be negative for arbitrary vectors.
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


class InMemoryVectorRepository(VectorRepository):
    def __init__(self) -> None:
        self._store: dict[str, tuple[Chunk, list[float]]] = {}

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        for chunk, embedding in zip(chunks, embeddings):
            self._store[chunk.chunk_id] = (chunk, embedding)

    async def search(
        self, query_embedding: list[float], top_k: int, document_filter: Optional[dict] = None
    ) -> list[RetrievedChunk]:
        candidates = list(self._store.values())
        candidates = [(c, e) for c, e in candidates if len(e) == len(query_embedding)]
        if document_filter and "document_id" in document_filter:
            candidates = [(c, e) for c, e in candidates if c.document_id == document_filter["document_id"]]

        scored = [
            RetrievedChunk(chunk=chunk, similarity_score=_cosine_similarity(query_embedding, embedding))
            for chunk, embedding in candidates
        ]
        scored.sort(key=lambda rc: rc.similarity_score, reverse=True)
        return scored[:top_k]

    async def delete(self, document_id: str) -> None:
        to_remove = [cid for cid, (chunk, _) in self._store.items() if chunk.document_id == document_id]
        for cid in to_remove:
            del self._store[cid]

    # Test-only helper, not part of the VectorRepository interface.
    def _count(self) -> int:
        return len(self._store)


class InMemoryMetadataRepository(MetadataRepository):
    def __init__(self) -> None:
        self._metadata: dict[str, dict] = {}
        self._fingerprints: dict[str, str] = {}  # fingerprint -> document_id
        self._status: dict[str, str] = {}

    async def get_document_metadata(self, document_id: str) -> dict:
        return self._metadata.get(document_id, {})

    async def save_document_metadata(self, document_id: str, metadata: dict) -> None:
        self._metadata[document_id] = metadata
        fingerprint = metadata.get("fingerprint")
        if fingerprint:
            self._fingerprints[fingerprint] = document_id
        self._status.setdefault(document_id, "pending")

    async def get_ingestion_status(self, document_id: str) -> str:
        return self._status.get(document_id, "unknown")

    async def find_by_fingerprint(self, fingerprint: str) -> Optional[str]:
        return self._fingerprints.get(fingerprint)

    # Test-only helper.
    def set_status(self, document_id: str, status: str) -> None:
        self._status[document_id] = status


class InMemoryCacheRepository(CacheRepository):
    def __init__(self) -> None:
        self._cache: dict[str, SearchResponse] = {}
        # Track which document_ids each cache_key's results reference, so
        # invalidate(document_id) can find and drop affected entries without
        # needing the caller to know cache-key structure.
        self._key_documents: dict[str, set[str]] = {}

    async def get(self, cache_key: str) -> Optional[SearchResponse]:
        return self._cache.get(cache_key)

    async def set(self, cache_key: str, response: SearchResponse, ttl_seconds: int) -> None:
        # TTL is intentionally not enforced in the fake (no background
        # eviction) -- Unit 2.8's tests exercise invalidate() explicitly
        # instead of relying on wall-clock expiry, which would make tests
        # slow/flaky.
        self._cache[cache_key] = response
        self._key_documents[cache_key] = {
            rc.retrieved_chunk.chunk.document_id for rc in response.ranked_chunks
        }

    async def invalidate(self, document_id: str) -> None:
        stale_keys = [key for key, doc_ids in self._key_documents.items() if document_id in doc_ids]
        for key in stale_keys:
            self._cache.pop(key, None)
            self._key_documents.pop(key, None)


class InMemoryFeedbackRepository(FeedbackRepository):
    def __init__(self) -> None:
        self._feedback: dict[str, list[dict]] = {}

    async def record_feedback(self, query_id: str, rating: int, comment: Optional[str] = None) -> None:
        self._feedback.setdefault(query_id, []).append({"rating": rating, "comment": comment})

    async def get_feedback_for_query(self, query_id: str) -> list[dict]:
        return list(self._feedback.get(query_id, []))
