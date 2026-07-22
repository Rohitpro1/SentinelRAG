"""Unit 2.9 tests -- CoverageAnalyzer."""
import pytest

from app.schemas.retrieval import Chunk, RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk
from app.services.verification.coverage_analyzer import CoverageAnalyzer


def make_ranked_chunk(chunk_id, rerank_score):
    chunk = Chunk(chunk_id=chunk_id, document_id="doc-1", text="content", token_count=10, source_reliability_score=0.9)
    return RankedChunk(retrieved_chunk=RetrievedChunk(chunk=chunk, similarity_score=0.9), rerank_score=rerank_score, rank=0)


@pytest.mark.asyncio
async def test_full_coverage_when_all_evidence_valid():
    chunks = [make_ranked_chunk("c1", 0.8), make_ranked_chunk("c2", 0.6)]
    coverage, _ = await CoverageAnalyzer().analyze(chunks, total_retrieved=2)
    assert coverage == 1.0


@pytest.mark.asyncio
async def test_partial_coverage_when_some_evidence_dropped():
    chunks = [make_ranked_chunk("c1", 0.8)]
    coverage, _ = await CoverageAnalyzer().analyze(chunks, total_retrieved=4)
    assert coverage == 0.25


@pytest.mark.asyncio
async def test_zero_total_retrieved_yields_zero_coverage_no_division_error():
    coverage, _ = await CoverageAnalyzer().analyze([], total_retrieved=0)
    assert coverage == 0.0


@pytest.mark.asyncio
async def test_reranker_confidence_averages_available_scores():
    chunks = [make_ranked_chunk("c1", 0.8), make_ranked_chunk("c2", 0.4)]
    _, reranker_confidence = await CoverageAnalyzer().analyze(chunks, total_retrieved=2)
    assert abs(reranker_confidence - 0.6) < 1e-9


@pytest.mark.asyncio
async def test_reranker_confidence_zero_when_all_degraded():
    chunks = [make_ranked_chunk("c1", None), make_ranked_chunk("c2", None)]
    _, reranker_confidence = await CoverageAnalyzer().analyze(chunks, total_retrieved=2)
    assert reranker_confidence == 0.0


@pytest.mark.asyncio
async def test_reranker_confidence_ignores_none_scores_in_average():
    chunks = [make_ranked_chunk("c1", 0.8), make_ranked_chunk("c2", None)]
    _, reranker_confidence = await CoverageAnalyzer().analyze(chunks, total_retrieved=2)
    assert reranker_confidence == 0.8  # only the non-None score counted
