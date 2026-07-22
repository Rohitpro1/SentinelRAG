"""Unit 2.6 tests -- EmbeddingService: timeout + retry-on-transient-failure behavior."""
import asyncio

import pytest

from app.core.exceptions import EmbeddingError
from app.core.settings.retrieval import RetrievalSettings
from app.services.embedding.base import BaseEmbedder
from app.services.embedding.deterministic import DeterministicEmbedder
from app.services.retrieval.embedding_service import EmbeddingService


class FlakyEmbedder(BaseEmbedder):
    """Fails N times then succeeds -- for exercising the retry path."""

    def __init__(self, fail_times: int, dimensions: int = 8):
        self._fail_times = fail_times
        self._calls = 0
        self._dimensions = dimensions
        self._inner = DeterministicEmbedder(dimensions=dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_query(self, text: str) -> list[float]:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise ConnectionError("simulated transient failure")
        return await self._inner.embed_query(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class AlwaysTimingOutEmbedder(BaseEmbedder):
    dimensions = 8

    async def embed_query(self, text: str) -> list[float]:
        await asyncio.sleep(10)
        return [0.0] * 8

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@pytest.fixture
def fast_settings():
    # Small timeouts/backoff so tests run quickly.
    return RetrievalSettings(embedding_timeout_ms=50, max_transient_retries=2, retry_backoff_base_ms=1)


@pytest.mark.asyncio
async def test_succeeds_on_first_try(fast_settings):
    service = EmbeddingService(DeterministicEmbedder(dimensions=8), fast_settings)
    result = await service.embed_query("hello")
    assert len(result) == 8


@pytest.mark.asyncio
async def test_recovers_after_transient_failures_within_retry_budget(fast_settings):
    embedder = FlakyEmbedder(fail_times=2)  # fails twice, succeeds on 3rd (== max_transient_retries + 1 attempts)
    service = EmbeddingService(embedder, fast_settings)
    result = await service.embed_query("hello")
    assert len(result) == 8
    assert embedder._calls == 3


@pytest.mark.asyncio
async def test_raises_embedding_error_after_exhausting_retries(fast_settings):
    embedder = FlakyEmbedder(fail_times=10)  # never succeeds within budget
    service = EmbeddingService(embedder, fast_settings)
    with pytest.raises(EmbeddingError):
        await service.embed_query("hello")


@pytest.mark.asyncio
async def test_timeout_is_treated_as_transient_and_retried_then_raises(fast_settings):
    service = EmbeddingService(AlwaysTimingOutEmbedder(), fast_settings)
    with pytest.raises(EmbeddingError):
        await service.embed_query("hello")
