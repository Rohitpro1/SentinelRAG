"""
Unit 2.1 tests — schema validation and the VerificationOutput -> VerificationReport
adapter, which is the one allowed seam into frozen Milestone 1 code.
"""
import pytest
from pydantic import ValidationError

from app.schemas.retrieval import Chunk, NLIRelation, PairwiseNLIResult, RetrievedChunk
from app.schemas.retrieval_domain import (
    RankedChunk,
    SearchRequest,
    SearchResponse,
    VerificationInput,
    VerificationOutput,
)


def make_chunk(chunk_id="c1", reliability=0.9):
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=f"Content for {chunk_id}",
        token_count=50,
        source_reliability_score=reliability,
    )


def make_retrieved_chunk(chunk_id="c1", similarity=0.9):
    return RetrievedChunk(chunk=make_chunk(chunk_id), similarity_score=similarity)


def make_ranked_chunk(chunk_id="c1", similarity=0.9, rerank_score=0.8, rank=0):
    return RankedChunk(
        retrieved_chunk=make_retrieved_chunk(chunk_id, similarity),
        rerank_score=rerank_score,
        rank=rank,
    )


# --- SearchRequest ---

def test_search_request_defaults():
    req = SearchRequest(query="what is the refund policy?")
    assert req.top_k == 20
    assert req.rerank_top_n == 5
    assert req.retry_count == 0


def test_search_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        SearchRequest(query="")


def test_search_request_rejects_negative_retry_count():
    with pytest.raises(ValidationError):
        SearchRequest(query="q", retry_count=-1)


# --- RankedChunk ---

def test_ranked_chunk_allows_none_rerank_score_for_degraded_mode():
    rc = RankedChunk(retrieved_chunk=make_retrieved_chunk(), rerank_score=None, rank=0)
    assert rc.rerank_score is None


def test_ranked_chunk_rejects_out_of_range_rerank_score():
    with pytest.raises(ValidationError):
        RankedChunk(retrieved_chunk=make_retrieved_chunk(), rerank_score=1.5, rank=0)


# --- SearchResponse ---

def test_search_response_defaults_to_empty_and_no_cache_hit():
    resp = SearchResponse(request=SearchRequest(query="q"))
    assert resp.ranked_chunks == []
    assert resp.cache_hit is False
    assert resp.stage_latencies_ms == {}


def test_search_response_carries_stage_latencies():
    resp = SearchResponse(
        request=SearchRequest(query="q"),
        ranked_chunks=[make_ranked_chunk()],
        stage_latencies_ms={"embedding": 42.1, "vector_search": 55.0},
    )
    assert resp.stage_latencies_ms["embedding"] == 42.1
    assert len(resp.ranked_chunks) == 1


# --- VerificationInput ---

def test_verification_input_basic():
    vi = VerificationInput(query="q", ranked_chunks=[make_ranked_chunk()], retry_count=1)
    assert vi.retry_count == 1
    assert len(vi.ranked_chunks) == 1


# --- VerificationOutput + adapter ---

def test_verification_output_to_verification_report_preserves_fields():
    nli = [PairwiseNLIResult(chunk_id_a="c1", chunk_id_b="c2", relation=NLIRelation.NEUTRAL, confidence=0.5)]
    vo = VerificationOutput(
        query="what is the refund policy?",
        retrieved_chunks=[make_retrieved_chunk("c1"), make_retrieved_chunk("c2")],
        nli_results=nli,
        retry_count=1,
    )
    report = vo.to_verification_report()
    assert report.query == vo.query
    assert len(report.retrieved_chunks) == 2
    assert report.nli_results == nli
    assert report.retry_count == 1


def test_verification_output_adapter_feeds_decision_engine_end_to_end():
    """
    This is the cross-domain integration check called out in the Retrieval
    Domain Design: VerificationOutput -> VerificationReport must actually
    work as input to the frozen Milestone 1 DecisionEngine.
    """
    from app.core.settings.decision_engine import DecisionEngineSettings
    from app.services.decision_engine.engine import DecisionEngine

    vo = VerificationOutput(
        query="q",
        retrieved_chunks=[make_retrieved_chunk("c1", similarity=0.9)],
        retry_count=0,
    )
    engine = DecisionEngine(DecisionEngineSettings())
    decision = engine.evaluate(vo.to_verification_report())
    assert decision.action.value in {"proceed", "low_confidence_response"}


def test_verification_output_from_ranked_chunks_flattens_correctly():
    ranked = [make_ranked_chunk("c1", rank=0), make_ranked_chunk("c2", rank=1)]
    vo = VerificationOutput.from_ranked_chunks(query="q", ranked_chunks=ranked, retry_count=2)
    assert len(vo.retrieved_chunks) == 2
    assert vo.retrieved_chunks[0].chunk.chunk_id == "c1"
    assert vo.retry_count == 2
    assert vo.nli_results == []
