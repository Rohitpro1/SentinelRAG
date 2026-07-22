"""
Unit 2.6 -- SearchService.

Owns: calling VectorRepository.search with a timeout and agent-level
retry on transient failures. Never handles non-transient errors -- a
malformed SearchRequest is already rejected by Pydantic validation
(Unit 2.1) before this service is ever called.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from app.core.exceptions import RetrievalError
from app.core.logging import get_logger, log_event
from app.core.settings.retrieval import RetrievalSettings
from app.repositories.interfaces import VectorRepository
from app.schemas.retrieval import RetrievedChunk


class SearchService:
    def __init__(self, vector_repository: VectorRepository, settings: RetrievalSettings, logger: Optional[logging.Logger] = None):
        self._vector_repository = vector_repository
        self._settings = settings
        self._logger = logger or get_logger(__name__)

    async def search(
        self, query_embedding: list[float], top_k: int, document_filter: Optional[dict] = None
    ) -> list[RetrievedChunk]:
        timeout_s = self._settings.vector_search_timeout_ms / 1000
        backoff_ms = self._settings.backoff_schedule_ms()
        last_exc: Optional[Exception] = None

        for attempt in range(self._settings.max_transient_retries + 1):
            start = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    self._vector_repository.search(query_embedding, top_k, document_filter), timeout=timeout_s
                )
                latency_ms = (time.perf_counter() - start) * 1000
                log_event(self._logger, "vector_search_succeeded", latency_ms=round(latency_ms, 3), attempt=attempt, results=len(result))
                return result
            except asyncio.TimeoutError as exc:
                last_exc = exc
                log_event(self._logger, "vector_search_timeout", level=logging.WARNING, attempt=attempt)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log_event(self._logger, "vector_search_failed", level=logging.WARNING, attempt=attempt, error=str(exc))

            if attempt < len(backoff_ms):
                await asyncio.sleep(backoff_ms[attempt] / 1000)

        raise RetrievalError(
            f"Vector search failed after {self._settings.max_transient_retries} retries",
            transient=True,
            context={"top_k": top_k},
        ) from last_exc
