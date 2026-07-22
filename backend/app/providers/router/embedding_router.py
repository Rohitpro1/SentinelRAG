from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from app.core.exceptions import SentinelRAGError, EmbeddingError
from app.providers.base.embedding_provider import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


def is_recoverable_error(exc: Exception) -> bool:
    """
    Determines if an exception is a recoverable infrastructure failure
    (HTTP 429 rate limit, 5xx server error, transport timeout, connection error).
    Unrecoverable auth errors (401/403) or programming errors return False.
    """
    if isinstance(exc, SentinelRAGError):
        msg = str(exc).lower()
        if "auth" in msg or "401" in msg or "403" in msg or "requires a non-empty api_key" in msg:
            return False

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403, 400, 422):
            return False
        if status == 429 or status >= 500:
            return True

    if isinstance(exc, (EmbeddingError, httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
        return True

    return True


class EmbeddingRouterProvider(BaseEmbeddingProvider):
    """
    Intelligent Embedding Router & Failover Provider.
    Primary: Gemini Embedding Provider
    Secondary: Voyage AI Embedding Provider
    Automatically fails over to secondary on recoverable infrastructure errors (429, timeouts, 5xx).
    """

    def __init__(
        self,
        primary: BaseEmbeddingProvider,
        secondary: Optional[BaseEmbeddingProvider] = None,
        primary_name: str = "gemini",
        secondary_name: str = "voyage",
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.primary_name = primary_name
        self.secondary_name = secondary_name
        self.active_provider_name = primary_name

        # Metrics Tracking
        self._metrics: dict[str, Any] = {
            "gemini_success_count": 0,
            "gemini_failure_count": 0,
            "voyage_usage_count": 0,
            "failover_count": 0,
            "retry_count": 0,
            "provider_latencies": {
                primary_name: [],
                secondary_name: [],
            },
        }

    @property
    def dimensions(self) -> int:
        # Return dimensions of current active provider
        if self.active_provider_name == self.secondary_name and self.secondary:
            return self.secondary.dimensions
        return self.primary.dimensions

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    async def embed_query(self, text: str) -> list[float]:
        start_time = time.monotonic()
        try:
            res = await self.primary.embed_query(text)
            elapsed = time.monotonic() - start_time
            self._metrics["gemini_success_count"] += 1
            self._metrics["provider_latencies"][self.primary_name].append(elapsed)
            self.active_provider_name = self.primary_name
            return res
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            self._metrics["gemini_failure_count"] += 1
            self._metrics["provider_latencies"][self.primary_name].append(elapsed)

            if not is_recoverable_error(exc) or not self.secondary:
                logger.error("Primary embedding provider (%s) failed with unrecoverable error: %s", self.primary_name, exc)
                raise

            logger.warning(
                "Embedding Provider Switch: Primary (%s) failed due to recoverable error (%s). Switching to Secondary (%s).",
                self.primary_name,
                exc,
                self.secondary_name,
            )
            self._metrics["failover_count"] += 1
            self._metrics["voyage_usage_count"] += 1
            self.active_provider_name = self.secondary_name

            sec_start = time.monotonic()
            sec_res = await self.secondary.embed_query(text)
            sec_elapsed = time.monotonic() - sec_start
            self._metrics["provider_latencies"][self.secondary_name].append(sec_elapsed)
            return sec_res

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        start_time = time.monotonic()
        try:
            res = await self.primary.embed_batch(texts)
            elapsed = time.monotonic() - start_time
            self._metrics["gemini_success_count"] += len(texts)
            self._metrics["provider_latencies"][self.primary_name].append(elapsed)
            self.active_provider_name = self.primary_name
            return res
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            self._metrics["gemini_failure_count"] += 1
            self._metrics["provider_latencies"][self.primary_name].append(elapsed)

            if not is_recoverable_error(exc) or not self.secondary:
                logger.error("Primary embedding provider (%s) failed batch with unrecoverable error: %s", self.primary_name, exc)
                raise

            logger.warning(
                "Embedding Provider Switch: Primary (%s) batch failed due to recoverable error (%s). Switching to Secondary (%s).",
                self.primary_name,
                exc,
                self.secondary_name,
            )
            self._metrics["failover_count"] += 1
            self._metrics["voyage_usage_count"] += len(texts)
            self.active_provider_name = self.secondary_name

            sec_start = time.monotonic()
            sec_res = await self.secondary.embed_batch(texts)
            sec_elapsed = time.monotonic() - sec_start
            self._metrics["provider_latencies"][self.secondary_name].append(sec_elapsed)
            return sec_res
