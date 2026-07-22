"""
Unit 2.6 -- EmbeddingService.

Owns: calling BaseEmbedder with a timeout, and agent-level retry on
TRANSIENT failures only (Retrieval Domain Design, Section 1). This is
deliberately separate from RetrieverAgent so the agent stays a thin
orchestrator (per the Unit 2.6 review instruction) and so this timeout
/retry logic is unit-testable in isolation from search/fusion/rerank.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger, log_event
from app.core.settings.retrieval import RetrievalSettings
from app.services.embedding.base import BaseEmbedder


class EmbeddingService:
    def __init__(self, embedder: BaseEmbedder, settings: RetrievalSettings, logger: Optional[logging.Logger] = None):
        self._embedder = embedder
        self._settings = settings
        self._logger = logger or get_logger(__name__)

    async def embed_query(self, text: str) -> list[float]:
        timeout_s = self._settings.embedding_timeout_ms / 1000
        backoff_ms = self._settings.backoff_schedule_ms()
        last_exc: Optional[Exception] = None

        for attempt in range(self._settings.max_transient_retries + 1):
            start = time.perf_counter()
            try:
                result = await asyncio.wait_for(self._embedder.embed_query(text), timeout=timeout_s)
                latency_ms = (time.perf_counter() - start) * 1000
                log_event(self._logger, "embedding_succeeded", latency_ms=round(latency_ms, 3), attempt=attempt)
                return result
            except asyncio.TimeoutError as exc:
                last_exc = exc
                log_event(self._logger, "embedding_timeout", level=logging.WARNING, attempt=attempt)
            except Exception as exc:  # noqa: BLE001 -- intentionally broad, translated below
                last_exc = exc
                log_event(self._logger, "embedding_failed", level=logging.WARNING, attempt=attempt, error=str(exc))

            if attempt < len(backoff_ms):
                await asyncio.sleep(backoff_ms[attempt] / 1000)

        raise EmbeddingError(
            f"Embedding failed after {self._settings.max_transient_retries} retries",
            context={"text_length": len(text)},
        ) from last_exc
