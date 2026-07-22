"""Unit 2.5 tests -- BaseReranker contract compliance for DeterministicReranker."""
import pytest

from app.schemas.retrieval import Chunk, RetrievedChunk
from app.services.reranking.base import BaseReranker
from app.services.reranking.deterministic import DeterministicReranker


def make_retrieved_chunk(chunk_id, similarity):
    chunk = Chunk(
        chunk_id=chunk_id, document_id="doc-1", text=f"content {chunk_id}",
        token_count=10, source_reliability_score=0.9,
    )
    return RetrievedChunk(chunk=chunk, similarity_score=similarity)


@pytest.mark.asyncio
async def test_is_a_valid_base_reranker():
    assert isinstance(DeterministicReranker(), BaseReranker)


@pytest.mark.asyncio
async def test_rerank_respects_top_n():
    reranker = DeterministicReranker()
    candidates = [make_retrieved_chunk(f"c{i}", 0.5) for i in range(10)]
    result = await reranker.rerank("query", candidates, top_n=3)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_rerank_is_deterministic_across_calls():
    reranker = DeterministicReranker()
    candidates = [make_retrieved_chunk("c1", 0.5), make_retrieved_chunk("c2", 0.6)]
    result1 = await reranker.rerank("query", candidates, top_n=2)
    result2 = await reranker.rerank("query", candidates, top_n=2)
    assert [r.rerank_score for r in result1] == [r.rerank_score for r in result2]
    assert [r.retrieved_chunk.chunk.chunk_id for r in result1] == [
        r.retrieved_chunk.chunk.chunk_id for r in result2
    ]


@pytest.mark.asyncio
async def test_rerank_scores_are_bounded():
    reranker = DeterministicReranker()
    candidates = [make_retrieved_chunk("c1", 0.99), make_retrieved_chunk("c2", 0.01)]
    result = await reranker.rerank("query", candidates, top_n=2)
    for ranked in result:
        assert 0.0 <= ranked.rerank_score <= 1.0


@pytest.mark.asyncio
async def test_rerank_assigns_sequential_zero_indexed_ranks():
    reranker = DeterministicReranker()
    candidates = [make_retrieved_chunk(f"c{i}", 0.5) for i in range(5)]
    result = await reranker.rerank("query", candidates, top_n=5)
    assert [r.rank for r in result] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_rerank_empty_candidates():
    reranker = DeterministicReranker()
    result = await reranker.rerank("query", [], top_n=5)
    assert result == []


@pytest.mark.asyncio
async def test_rerank_order_is_by_descending_score():
    reranker = DeterministicReranker()
    candidates = [make_retrieved_chunk(f"c{i}", 0.5) for i in range(8)]
    result = await reranker.rerank("some query", candidates, top_n=8)
    scores = [r.rerank_score for r in result]
    assert scores == sorted(scores, reverse=True)


# --- Unit 2.12 additions: RerankResult / RerankerCapabilities ---

@pytest.mark.asyncio
async def test_rerank_with_result_shape():
    reranker = DeterministicReranker()
    candidates = [make_retrieved_chunk("c1", 0.5), make_retrieved_chunk("c2", 0.6)]
    result = await reranker.rerank_with_result("q", candidates, top_n=2)
    assert len(result.ranked_chunks) == 2
    assert result.provider == "deterministic"
    assert result.model_name == "deterministic-sha256-perturbation"
    assert result.model_version == "v1"
    assert result.rerank_latency_ms >= 0


@pytest.mark.asyncio
async def test_rerank_with_result_matches_plain_rerank():
    reranker = DeterministicReranker()
    candidates = [make_retrieved_chunk("c1", 0.5), make_retrieved_chunk("c2", 0.6)]
    direct = await reranker.rerank("q", candidates, top_n=2)
    via_result = (await reranker.rerank_with_result("q", candidates, top_n=2)).ranked_chunks
    assert [r.rerank_score for r in direct] == [r.rerank_score for r in via_result]


def test_capabilities_returns_unconstrained_values():
    caps = DeterministicReranker().capabilities()
    assert caps.supports_batching is True
    assert caps.max_batch_size > 0
    assert caps.max_input_tokens > 0
