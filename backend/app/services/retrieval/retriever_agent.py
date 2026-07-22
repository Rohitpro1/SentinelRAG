"""
Unit 2.6 -- RetrieverAgent.

Deliberately small: this class ONLY orchestrates the five-stage chain
(cache check -> embed -> search -> fuse -> rerank -> cache write) and
emits one summary telemetry event. All actual logic (timeouts, retries,
degradation, fusion rules) lives in the dedicated services it composes,
per the Unit 2.6 review instruction to keep the agent from becoming a
God Object.

Boundary note: per the frozen Retrieval Domain Design's dependency graph
(Section 5), RetrieverAgent and VerificationAgent are siblings -- this
agent does NOT depend on verification. Its output (SearchResponse) is
handed to VerificationAgent (Unit 2.9) by the caller, not by this class.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Optional

from app.core.logging import get_logger, log_event
from app.core.settings.retrieval import RetrievalSettings
from app.repositories.interfaces import CacheRepository
from app.schemas.retrieval_domain import SearchRequest, SearchResponse
from app.services.retrieval.embedding_service import EmbeddingService
from app.services.retrieval.fusion_service import FusionService
from app.services.retrieval.reranking_service import RerankingService
from app.services.retrieval.search_service import SearchService


class RetrieverAgent:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        search_service: SearchService,
        fusion_service: FusionService,
        reranking_service: RerankingService,
        settings: RetrievalSettings,
        cache_repository: Optional[CacheRepository] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._embedding_service = embedding_service
        self._search_service = search_service
        self._fusion_service = fusion_service
        self._reranking_service = reranking_service
        self._settings = settings
        self._cache_repository = cache_repository
        self._logger = logger or get_logger(__name__)

    async def search(self, request: SearchRequest) -> SearchResponse:
        cache_key = self._cache_key(request)

        if self._cache_repository is not None:
            cached = await self._cache_repository.get(cache_key)
            if cached is not None:
                log_event(
                    self._logger, "retrieval_completed",
                    request_id=request.request_id, trace_id=request.trace_id,
                    cache_hit=True, candidates_returned=len(cached.ranked_chunks),
                    retry_count=request.retry_count,
                )
                return cached.model_copy(update={"cache_hit": True})

        stage_latencies: dict[str, float] = {}

        t0 = time.perf_counter()
        query_embedding = await self._embedding_service.embed_query(request.query)
        stage_latencies["embedding"] = round((time.perf_counter() - t0) * 1000, 3)

        t1 = time.perf_counter()
        semantic_results = await self._search_service.search(query_embedding, request.top_k, request.document_filter)
        stage_latencies["vector_search"] = round((time.perf_counter() - t1) * 1000, 3)

        fused_results = await self._fusion_service.fuse(semantic_results)

        t2 = time.perf_counter()
        ranked_chunks = await self._reranking_service.rerank(request.query, fused_results, request.rerank_top_n)
        stage_latencies["rerank"] = round((time.perf_counter() - t2) * 1000, 3)

        response = SearchResponse(
            request=request, ranked_chunks=ranked_chunks, cache_hit=False, stage_latencies_ms=stage_latencies
        )

        if self._cache_repository is not None:
            await self._cache_repository.set(cache_key, response, ttl_seconds=self._settings.cache_ttl_seconds)

        log_event(
            self._logger, "retrieval_completed",
            request_id=request.request_id, trace_id=request.trace_id,
            cache_hit=False, candidates_returned=len(ranked_chunks),
            latency_ms=round(sum(stage_latencies.values()), 3), retry_count=request.retry_count,
        )
        return response

    @staticmethod
    def _cache_key(request: SearchRequest) -> str:
        payload = json.dumps(
            {
                "query": request.query,
                "top_k": request.top_k,
                "rerank_top_n": request.rerank_top_n,
                "document_filter": request.document_filter,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
