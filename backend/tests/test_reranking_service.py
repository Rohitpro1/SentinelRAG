"""
Unit 2.6 tests -- RerankingService: the graceful-degradation path is the
one most important to exercise explicitly (per the frozen design's
non-fatal-reranker-failure policy).
"""
import asyncio

import pytest

from app.core.settings.retrieval import RetrievalSettings
from app.schemas.retrieval import Chunk, RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk
from app.services.reranking.base import BaseReranker
from app.services.reranking.deterministic import DeterministicReranker
from app.services.retrieval.reranking_service import RerankingService


def make_rc(chunk_id, similarity):
    chunk = Chunk(chunk_id=chunk_id, document_id="doc-1", text="x", token_count=5, source_reliability_score=0.9)
    return RetrievedChunk(chunk=chunk, similarity_score=similarity)


class AlwaysFailingReranker(BaseReranker):
    async def rerank(self, query, candidates, top_n):
        raise RuntimeError("cross-encoder is down")


class AlwaysTimingOutReranker(BaseReranker):
    async def rerank(self, query, candidates, top_n):
        await asyncio.sleep(10)
        return []


@pytest.fixture
def fast_settings():
    return RetrievalSettings(rerank_timeout_ms=50)


@pytest.mark.asyncio
async def test_successful_rerank_passes_through(fast_settings):
    service = RerankingService(DeterministicReranker(), fast_settings)
    candidates = [make_rc("c1", 0.5), make_rc("c2", 0.6)]
    result = await service.rerank("q", candidates, top_n=2)
    assert len(result) == 2
    assert all(r.rerank_score is not None for r in result)


@pytest.mark.asyncio
async def test_degrades_gracefully_on_reranker_exception(fast_settings):
    service = RerankingService(AlwaysFailingReranker(), fast_settings)
    candidates = [make_rc("c1", 0.5), make_rc("c2", 0.9)]
    result = await service.rerank("q", candidates, top_n=2)
    assert len(result) == 2
    assert all(r.rerank_score is None for r in result)
    # Degraded mode still ranks by original similarity, descending.
    assert result[0].retrieved_chunk.chunk.chunk_id == "c2"


@pytest.mark.asyncio
async def test_degrades_gracefully_on_timeout(fast_settings):
    service = RerankingService(AlwaysTimingOutReranker(), fast_settings)
    candidates = [make_rc("c1", 0.9)]
    result = await service.rerank("q", candidates, top_n=1)
    assert result[0].rerank_score is None


@pytest.mark.asyncio
async def test_degraded_mode_respects_top_n(fast_settings):
    service = RerankingService(AlwaysFailingReranker(), fast_settings)
    candidates = [make_rc(f"c{i}", 0.1 * i) for i in range(10)]
    result = await service.rerank("q", candidates, top_n=3)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_degraded_mode_assigns_sequential_ranks(fast_settings):
    service = RerankingService(AlwaysFailingReranker(), fast_settings)
    candidates = [make_rc(f"c{i}", 0.1 * i) for i in range(4)]
    result = await service.rerank("q", candidates, top_n=4)
    assert [r.rank for r in result] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_never_raises_regardless_of_reranker_failure(fast_settings):
    # This is the core invariant: RerankingService must never propagate
    # an exception to its caller, no matter what the underlying reranker does.
    service = RerankingService(AlwaysFailingReranker(), fast_settings)
    try:
        await service.rerank("q", [make_rc("c1", 0.5)], top_n=1)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"RerankingService must never raise, but raised: {exc}")
