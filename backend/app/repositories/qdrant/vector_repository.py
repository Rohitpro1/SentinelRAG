"""
Unit 2.10 -- QdrantVectorRepository.

First real-infrastructure implementation in the project. Implements the
domain's VectorRepository interface EXACTLY (same method signatures as
InMemoryVectorRepository, Unit 2.2) -- this file is the only place in the
codebase that imports qdrant_client; the domain layer (VectorRepository,
RetrieverAgent, SearchService, etc.) has zero knowledge Qdrant exists.

Engineering trade-offs are documented inline at each decision point below,
per instruction 8.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.exceptions import RetrievalError
from app.core.logging import get_logger, log_event
from app.core.settings.storage import StorageSettings
from app.repositories.interfaces import VectorRepository
from app.schemas.retrieval import Chunk, RetrievedChunk

# Fixed namespace for deterministic UUID5 point IDs.
#
# TRADE-OFF: Qdrant's server-side point ID validation only accepts unsigned
# integers or UUIDs -- an arbitrary string like "c1" (our Chunk.chunk_id
# format) is rejected at the server, even though the client SDK's type
# hints permit `str` generically. Rather than constrain Chunk.chunk_id's
# format (which would leak a Qdrant-specific constraint into the domain
# schema -- exactly what instruction 2 forbids), this repository derives a
# deterministic UUID5 from chunk_id for the Qdrant point ID, and stores the
# original chunk_id in the payload as the source of truth on read. Same
# chunk_id always maps to the same point ID, so upsert-by-chunk_id remains
# idempotent.
_QDRANT_POINT_ID_NAMESPACE = uuid.UUID("6f9c1b2a-6b1e-4b0a-9c2d-8f3e1a2b3c4d")


def _chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_QDRANT_POINT_ID_NAMESPACE, chunk_id))


class QdrantVectorRepository(VectorRepository):
    def __init__(self, client: AsyncQdrantClient, settings: StorageSettings, logger: Optional[logging.Logger] = None):
        # DI per instruction 7: the client is constructed exclusively by
        # app.infrastructure.qdrant_client_factory.create_qdrant_client()
        # and injected here -- this class never constructs its own client.
        self._client = client
        self._settings = settings
        self._collection = settings.qdrant_collection
        self._logger = logger or get_logger(__name__)

    async def ensure_collection(self, vector_size: Optional[int] = None) -> None:
        """
        Idempotent collection setup. Ensures target collection exists for given vector_size.
        """
        size = vector_size or self._settings.qdrant_vector_size
        collection = f"{self._collection}_{size}"
        try:
            exists = await self._client.collection_exists(collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=collection,
                    vectors_config=qmodels.VectorParams(
                        size=size, distance=qmodels.Distance.COSINE
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            raise self._translate_exception(exc, operation="ensure_collection") from exc

    def _get_collection_name(self, vector: list[float]) -> str:
        size = len(vector) if vector else self._settings.qdrant_vector_size
        return f"{self._collection}_{size}"

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise RetrievalError(
                "chunks and embeddings must be the same length",
                transient=False,
                context={"chunks": len(chunks), "embeddings": len(embeddings)},
            )

        if not chunks:
            return

        vector_size = len(embeddings[0])
        collection_name = self._get_collection_name(embeddings[0])
        await self.ensure_collection(vector_size=vector_size)

        points = [
            qmodels.PointStruct(
                id=_chunk_id_to_point_id(chunk.chunk_id),
                vector=embedding,
                payload=self._chunk_to_payload(chunk, len(embedding)),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        await self._execute_with_single_retry(
            operation="upsert",
            fn=lambda: self._client.upsert(collection_name=collection_name, points=points),
            extra_log_fields={"point_count": len(points), "collection": collection_name},
        )

    async def search(
        self, query_embedding: list[float], top_k: int, document_filter: Optional[dict] = None
    ) -> list[RetrievedChunk]:
        qdrant_filter = self._build_filter(document_filter)
        collection_name = self._get_collection_name(query_embedding)
        await self.ensure_collection(vector_size=len(query_embedding))

        start = time.perf_counter()
        try:
            response = await self._client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._translate_exception(exc, operation="search") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        results = [self._point_to_retrieved_chunk(point) for point in response.points]
        log_event(self._logger, "qdrant_search_succeeded", latency_ms=round(latency_ms, 3), results=len(results), collection=collection_name)
        return results

    async def delete(self, document_id: str) -> None:
        # Same single-bounded-retry trade-off as upsert -- see above.
        await self._execute_with_single_retry(
            operation="delete",
            fn=lambda: self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
                    )
                ),
            ),
            extra_log_fields={"document_id": document_id},
        )

    async def close(self) -> None:
        """
        Explicit, deterministic teardown (instruction 4). Delegates to the
        injected client rather than owning connection lifecycle itself --
        the client was constructed by the factory and may be shared;
        whether close() here or the factory's close_qdrant_client() is the
        canonical shutdown path is an application-bootstrap decision (not
        yet made, since no bootstrap/main.py exists). Exposed here for
        symmetry and so this repository is independently disposable in
        tests without reaching into the factory module.
        """
        await self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_with_single_retry(self, *, operation: str, fn, extra_log_fields: dict) -> None:
        last_exc: Optional[Exception] = None
        for attempt in range(2):  # initial attempt + 1 retry, bounded and hardcoded (see trade-off note above)
            start = time.perf_counter()
            try:
                await fn()
                latency_ms = (time.perf_counter() - start) * 1000
                log_event(
                    self._logger, f"qdrant_{operation}_succeeded",
                    latency_ms=round(latency_ms, 3), attempt=attempt, **extra_log_fields,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                translated = self._translate_exception(exc, operation=operation)
                if not translated.transient or attempt == 1:
                    raise translated from exc
                log_event(self._logger, f"qdrant_{operation}_retrying", level=logging.WARNING, attempt=attempt, error=str(exc))
        # Unreachable in practice (loop always returns or raises), but
        # keeps type checkers satisfied and fails loudly if reached.
        raise self._translate_exception(last_exc or RuntimeError("unknown failure"), operation=operation)

    @staticmethod
    def _build_filter(document_filter: Optional[dict]) -> Optional[qmodels.Filter]:
        if not document_filter or "document_id" not in document_filter:
            return None
        return qmodels.Filter(
            must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_filter["document_id"]))]
        )

    @staticmethod
    def _chunk_to_payload(chunk: Chunk, vector_dimension: int = 768) -> dict[str, Any]:
        meta = dict(chunk.metadata or {})
        meta["vector_dimension"] = vector_dimension
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "source_reliability_score": chunk.source_reliability_score,
            "ocr_confidence": chunk.ocr_confidence,
            "metadata": meta,
        }

    @staticmethod
    def _point_to_retrieved_chunk(point: Any) -> RetrievedChunk:
        payload = point.payload or {}
        chunk = Chunk(
            chunk_id=payload.get("chunk_id", str(point.id)),
            document_id=payload["document_id"],
            text=payload["text"],
            token_count=payload["token_count"],
            source_reliability_score=payload["source_reliability_score"],
            ocr_confidence=payload.get("ocr_confidence"),
            metadata=payload.get("metadata", {}),
        )
        # TRADE-OFF: Qdrant's COSINE distance metric returns a score that
        # is mathematically cosine similarity in [-1, 1] for this client
        # version, but RetrievedChunk.similarity_score is constrained to
        # [0, 1] (Milestone 1 schema, frozen). Clamped rather than
        # rescaled/renormalized -- rescaling (-1,1)->(0,1) would silently
        # change what "0.5 similarity" means versus every other
        # VectorRepository implementation (e.g. InMemoryVectorRepository's
        # cosine similarity, Unit 2.2, is ALSO clamped, not rescaled) --
        # clamping keeps the two implementations' output comparable.
        similarity = max(0.0, min(1.0, float(point.score)))
        return RetrievedChunk(chunk=chunk, similarity_score=similarity)

    def _translate_exception(self, exc: Exception, *, operation: str) -> RetrievalError:
        transient = self._is_transient(exc)
        log_event(
            self._logger, "qdrant_operation_failed", level=logging.WARNING,
            operation=operation, error=str(exc), error_type=type(exc).__name__, transient=transient,
        )
        return RetrievalError(f"Qdrant {operation} failed: {exc}", transient=transient, context={"operation": operation})

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        # Network-level failures and timeouts: always transient -- these
        # are exactly the "temporary outage" / "network failure" /
        # "timeout" cases instruction 5 asks this class to be designed for.
        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout, TimeoutError, ConnectionError)):
            return True
        if isinstance(exc, UnexpectedResponse):
            # 5xx = server-side issue, worth retrying. 4xx (e.g. malformed
            # filter, wrong vector dimension) = caller error, retrying
            # would just fail identically every time -- not transient.
            status = exc.status_code or 0
            return status >= 500
        # Unknown exception type: default to transient. Rationale: the
        # cost of an unnecessary retry (one extra attempt, bounded) is
        # much lower than the cost of treating a genuinely-recoverable
        # failure as permanent and giving up immediately -- "design for
        # partial failures" (instruction 5) favors the safer default here.
        return True
