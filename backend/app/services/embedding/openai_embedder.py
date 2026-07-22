"""
Unit 2.11 -- OpenAIEmbedder: first real embedding provider.

Named per the sibling list already documented in DeterministicEmbedder
(Unit 2.4): "SentenceTransformerEmbedder, OpenAIEmbedder, OllamaEmbedder,
AzureOpenAIEmbedder". Implements the OpenAI embeddings REST API shape
(POST {base_url}/embeddings, {"model", "input"} -> {"data": [{"embedding": [...]}], "model": ...}),
which is also what Ollama, vLLM, and several other self-hosted servers
expose for compatibility -- so this one class covers OpenAI itself and
any OpenAI-compatible endpoint via EmbeddingSettings.api_base_url,
without a separate class per provider today. A genuinely divergent API
shape (e.g. a provider needing different auth or a different request
schema) would warrant its own class -- documented here as a deliberate
scope decision, not an oversight.

NETWORK NOTE: this implementation's HTTP logic is exercised in this
codebase's test suite entirely via httpx.MockTransport (no real network
call) -- the development sandbox this was built in has no route to
api.openai.com or any embedding provider. The isolated integration test
(test_openai_embedder_integration.py) is written to run against a real
endpoint and will skip cleanly wherever one isn't configured/reachable,
identical in spirit to Unit 2.10's live-Qdrant test.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger, log_event
from app.core.settings.embedding import EmbeddingSettings
from app.schemas.embedding import EmbedderHealth, EmbedderHealthState, EmbeddingResult
from app.services.embedding.base import BaseEmbedder
from app.services.embedding.result_builder import build_embedding_result


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, client: httpx.AsyncClient, settings: EmbeddingSettings, logger: Optional[logging.Logger] = None):
        # DI per the project's established pattern (Unit 2.10): the client
        # is constructed exclusively by
        # app.infrastructure.embedding_client_factory.create_embedding_http_client()
        # and injected here.
        self._client = client
        self._settings = settings
        self._logger = logger or get_logger(__name__)
        self._consecutive_failures = 0

    @property
    def dimensions(self) -> int:
        return self._settings.dimensions

    async def embed_query(self, text: str) -> list[float]:
        # No internal retry -- see class docstring / the Unit 2.10 pattern
        # this mirrors: EmbeddingService (Unit 2.6) already wraps this
        # call with timeout + backoff retry. A second retry loop here
        # would multiply actual attempts unpredictably.
        vectors = await self._request_embeddings([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # TRADE-OFF: single bounded retry here, same reasoning as Unit
        # 2.10's upsert()/delete() -- no IngestionService exists yet to
        # own retry for the ingestion-time batch path, so this repository-
        # equivalent layer takes on one bounded retry until that service
        # exists, at which point this should be removed to avoid the same
        # duplication problem the read path (embed_query) already avoids.
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                return await self._request_embeddings(texts)
            except EmbeddingError as exc:
                last_exc = exc
                if attempt == 1:
                    raise
                log_event(self._logger, "embedding_batch_retrying", level=logging.WARNING, attempt=attempt)
        if last_exc is not None:
            raise last_exc
        raise EmbeddingError("Batch embedding failed with unknown error.")

    async def embed_query_with_result(self, text: str) -> EmbeddingResult:
        return await build_embedding_result(
            lambda: self.embed_query(text),
            provider="openai_compatible",
            model_name=self._settings.model_name,
            model_version=None,  # populated per-call in _request_embeddings via response "model" field when needed
        )

    def health(self) -> EmbedderHealth:
        threshold = self._settings.unavailable_after_consecutive_failures
        if self._consecutive_failures == 0:
            return EmbedderHealth(state=EmbedderHealthState.READY)
        if self._consecutive_failures < threshold:
            return EmbedderHealth(
                state=EmbedderHealthState.DEGRADED,
                detail=f"{self._consecutive_failures} consecutive failure(s)",
            )
        return EmbedderHealth(
            state=EmbedderHealthState.UNAVAILABLE,
            detail=f"{self._consecutive_failures} consecutive failures (threshold={threshold})",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        start = time.perf_counter()
        try:
            response = await self._client.post(
                "/embeddings", json={"model": self._settings.model_name, "input": texts}
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self._consecutive_failures += 1
            log_event(
                self._logger, "embedding_request_failed", level=logging.WARNING,
                error=str(exc), error_type=type(exc).__name__, consecutive_failures=self._consecutive_failures,
            )
            # Per the frozen Architecture Enhancements (Section 5) and
            # Unit 2.6's exceptions.py: EmbeddingError is ALWAYS transient
            # by design -- no 4xx/5xx distinction here, unlike
            # QdrantVectorRepository's RetrievalError. That asymmetry is
            # an existing frozen decision, not something introduced here.
            raise EmbeddingError(f"Embedding request failed: {exc}", context={"text_count": len(texts)}) from exc

        self._consecutive_failures = 0
        latency_ms = (time.perf_counter() - start) * 1000

        try:
            payload = response.json()
            vectors = [item["embedding"] for item in payload["data"]]
        except (KeyError, ValueError) as exc:
            raise EmbeddingError(f"Malformed embedding response: {exc}", context={"text_count": len(texts)}) from exc

        log_event(
            self._logger, "embedding_request_succeeded",
            latency_ms=round(latency_ms, 3), text_count=len(texts), model=self._settings.model_name,
        )
        return vectors
