"""Unit 2.6 tests -- SearchService: timeout + retry-on-transient-failure behavior."""
import asyncio

import pytest

from app.core.exceptions import RetrievalError
from app.core.settings.retrieval import RetrievalSettings
from app.repositories.fakes.in_memory import InMemoryVectorRepository
from app.repositories.interfaces import VectorRepository
from app.schemas.retrieval import Chunk
from app.services.retrieval.search_service import SearchService


class FlakyVectorRepository(VectorRepository):
    def __init__(self, fail_times: int):
        self._fail_times = fail_times
        self._calls = 0
        self._inner = InMemoryVectorRepository()

    async def upsert(self, chunks, embeddings):
        await self._inner.upsert(chunks, embeddings)

    async def search(self, query_embedding, top_k, document_filter=None):
        self._calls += 1
        if self._calls <= self._fail_times:
            raise ConnectionError("simulated transient failure")
        return await self._inner.search(query_embedding, top_k, document_filter)

    async def delete(self, document_id):
        await self._inner.delete(document_id)


class AlwaysTimingOutVectorRepository(VectorRepository):
    async def upsert(self, chunks, embeddings):
        pass

    async def search(self, query_embedding, top_k, document_filter=None):
        await asyncio.sleep(10)
        return []

    async def delete(self, document_id):
        pass


@pytest.fixture
def fast_settings():
    return RetrievalSettings(vector_search_timeout_ms=50, max_transient_retries=2, retry_backoff_base_ms=1)


@pytest.mark.asyncio
async def test_succeeds_on_first_try(fast_settings):
    repo = InMemoryVectorRepository()
    chunk = Chunk(chunk_id="c1", document_id="doc-1", text="x", token_count=5, source_reliability_score=0.9)
    await repo.upsert([chunk], [[1.0, 0.0]])
    service = SearchService(repo, fast_settings)
    results = await service.search([1.0, 0.0], top_k=5)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_recovers_after_transient_failures_within_retry_budget(fast_settings):
    repo = FlakyVectorRepository(fail_times=2)
    service = SearchService(repo, fast_settings)
    results = await service.search([1.0, 0.0], top_k=5)
    assert results == []  # empty repo, but no exception raised
    assert repo._calls == 3


@pytest.mark.asyncio
async def test_raises_retrieval_error_transient_after_exhausting_retries(fast_settings):
    repo = FlakyVectorRepository(fail_times=10)
    service = SearchService(repo, fast_settings)
    with pytest.raises(RetrievalError) as exc_info:
        await service.search([1.0, 0.0], top_k=5)
    assert exc_info.value.transient is True


@pytest.mark.asyncio
async def test_timeout_eventually_raises_retrieval_error(fast_settings):
    service = SearchService(AlwaysTimingOutVectorRepository(), fast_settings)
    with pytest.raises(RetrievalError):
        await service.search([1.0, 0.0], top_k=5)
