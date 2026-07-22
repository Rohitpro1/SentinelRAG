"""
Unit 2.12 -- CrossEncoderReranker: first real reranking provider.

Named per the sibling already documented in DeterministicReranker
(Unit 2.5): "CrossEncoderReranker (e.g. a sentence-transformers
cross-encoder model)". Implements a REST rerank API shape --
POST {base_url}/rerank {"model", "query", "documents", "top_n"} ->
{"results": [{"index": i, "relevance_score": s}, ...], "model": ...} --
matching the Cohere/Jina-style rerank API convention that several hosted
and self-hosted cross-encoder servers expose, so this one class covers
multiple providers via RerankingSettings.api_base_url, same scope
decision as Unit 2.11's OpenAIEmbedder.

NETWORK NOTE: identical situation to Unit 2.11 -- this sandbox has no
route to any reranking provider. All logic here is tested via
httpx.MockTransport; the isolated integration test skips cleanly without
a configured live endpoint.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.core.exceptions import RerankError
from app.core.logging import get_logger, log_event
from app.core.settings.reranking import RerankingSettings
from app.schemas.reranking import RerankerCapabilities, RerankResult
from app.schemas.retrieval import RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk
from app.services.reranking.base import BaseReranker
from app.services.reranking.result_builder import build_rerank_result


class CrossEncoderReranker(BaseReranker):
    def __init__(self, client: httpx.AsyncClient, settings: RerankingSettings, logger: Optional[logging.Logger] = None):
        # DI per the established pattern (Units 2.10/2.11): client is
        # constructed exclusively by
        # app.infrastructure.reranking_client_factory.create_reranking_http_client()
        # and injected here.
        self._client = client
        self._settings = settings
        self._logger = logger or get_logger(__name__)

    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RankedChunk]:
        if not candidates:
            return []

        # No internal retry, and deliberately so -- more strongly true
        # here than Units 2.10/2.11's read paths: RerankingService (Unit
        # 2.6) already catches ANY exception from this method and
        # degrades to similarity-ranked results immediately. A retry loop
        # here would only add latency to a path explicitly designed to
        # fail fast and gracefully -- retrying delays the degradation
        # without changing the outcome.
        try:
            batches = self._split_into_batches(candidates)
            scored_pairs: list[tuple[float, RetrievedChunk]] = []
            for batch in batches:
                scored_pairs.extend(await self._rerank_batch(query, batch))
        except RerankError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RerankError(f"Cross-encoder rerank failed: {exc}") from exc

        scored_pairs.sort(key=lambda pair: pair[0], reverse=True)
        top = scored_pairs[:top_n]
        return [
            RankedChunk(retrieved_chunk=rc, rerank_score=score, rank=idx)
            for idx, (score, rc) in enumerate(top)
        ]

    async def rerank_with_result(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> RerankResult:
        return await build_rerank_result(
            lambda: self.rerank(query, candidates, top_n),
            provider="cross_encoder_http",
            model_name=self._settings.model_name,
        )

    def capabilities(self) -> RerankerCapabilities:
        """Configuration-driven only -- no runtime provider discovery (instruction 2)."""
        return RerankerCapabilities(
            supports_batching=self._settings.supports_batching,
            max_batch_size=self._settings.max_batch_size,
            max_input_tokens=self._settings.max_input_tokens,
            model_dimensions=self._settings.model_dimensions,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_into_batches(self, candidates: list[RetrievedChunk]) -> list[list[RetrievedChunk]]:
        """
        Batches according to RerankingSettings.max_batch_size -- a
        configured limit, not a discovered one (instruction 2). If
        supports_batching is False, every candidate becomes its own
        single-item batch (one request per candidate) rather than
        raising, so a provider without batching support still works,
        just with more requests -- a graceful capability degradation,
        not a hard failure.
        """
        batch_size = self._settings.max_batch_size if self._settings.supports_batching else 1
        return [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]

    async def _rerank_batch(self, query: str, batch: list[RetrievedChunk]) -> list[tuple[float, RetrievedChunk]]:
        documents = [rc.chunk.text for rc in batch]
        try:
            response = await self._client.post(
                "/rerank",
                json={"model": self._settings.model_name, "query": query, "documents": documents},
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            log_event(
                self._logger, "rerank_request_failed", level=logging.WARNING,
                error=str(exc), error_type=type(exc).__name__, batch_size=len(batch),
            )
            raise RerankError(f"Rerank request failed: {exc}", context={"batch_size": len(batch)}) from exc

        try:
            payload = response.json()
            results = payload["results"]
        except (KeyError, ValueError) as exc:
            raise RerankError(f"Malformed rerank response: {exc}", context={"batch_size": len(batch)}) from exc

        log_event(self._logger, "rerank_request_succeeded", batch_size=len(batch), model=self._settings.model_name)
        return [(item["relevance_score"], batch[item["index"]]) for item in results]
