import pytest
from app.core.exceptions import DecisionEngineError
from app.core.settings.decision_engine import DecisionEngineSettings
from app.schemas.retrieval import (
    Chunk, Decision, DecisionAction, NLIRelation, PairwiseNLIResult, RetrievedChunk, VerificationReport,
)
from app.services.decision_engine.engine import DecisionEngine


@pytest.fixture
def settings():
    return DecisionEngineSettings(
        min_retrieval_similarity=0.55, contradiction_threshold=0.40,
        low_confidence_threshold=0.60, max_retrieval_retries=2,
    )


@pytest.fixture
def engine(settings):
    return DecisionEngine(settings)


def make_chunk(chunk_id, reliability=0.9, ocr_conf=None):
    return Chunk(
        chunk_id=chunk_id, document_id="doc-1", text=f"Content for {chunk_id}",
        token_count=50, source_reliability_score=reliability, ocr_confidence=ocr_conf,
    )


def test_no_chunks_retries_when_budget_remains(engine):
    report = VerificationReport(query="q", retrieved_chunks=[], retry_count=0)
    assert engine.evaluate(report).action == DecisionAction.RETRY_RETRIEVAL


def test_no_chunks_clarifies_after_max_retries(engine):
    report = VerificationReport(query="q", retrieved_chunks=[], retry_count=2)
    assert engine.evaluate(report).action == DecisionAction.CLARIFY


def test_weak_similarity_triggers_retry(engine):
    report = VerificationReport(
        query="q", retrieved_chunks=[RetrievedChunk(chunk=make_chunk("c1"), similarity_score=0.30)], retry_count=0
    )
    assert engine.evaluate(report).action == DecisionAction.RETRY_RETRIEVAL


def test_weak_similarity_clarifies_after_retries_exhausted(engine):
    report = VerificationReport(
        query="q", retrieved_chunks=[RetrievedChunk(chunk=make_chunk("c1"), similarity_score=0.30)], retry_count=2
    )
    assert engine.evaluate(report).action == DecisionAction.CLARIFY


def test_contradiction_routes_to_human_review(engine):
    report = VerificationReport(
        query="q",
        retrieved_chunks=[
            RetrievedChunk(chunk=make_chunk("c1"), similarity_score=0.9),
            RetrievedChunk(chunk=make_chunk("c2"), similarity_score=0.85),
        ],
        nli_results=[PairwiseNLIResult(chunk_id_a="c1", chunk_id_b="c2", relation=NLIRelation.CONTRADICTION, confidence=0.75)],
        retry_count=0,
    )
    decision = engine.evaluate(report)
    assert decision.action == DecisionAction.HUMAN_REVIEW
    assert "Contradiction" in decision.reasons[0]


def test_contradiction_below_threshold_does_not_trigger_review(engine):
    report = VerificationReport(
        query="q",
        retrieved_chunks=[
            RetrievedChunk(chunk=make_chunk("c1"), similarity_score=0.9),
            RetrievedChunk(chunk=make_chunk("c2"), similarity_score=0.85),
        ],
        nli_results=[PairwiseNLIResult(chunk_id_a="c1", chunk_id_b="c2", relation=NLIRelation.CONTRADICTION, confidence=0.10)],
        retry_count=0,
    )
    assert engine.evaluate(report).action != DecisionAction.HUMAN_REVIEW


def test_high_confidence_proceeds(engine, settings):
    report = VerificationReport(
        query="q", retrieved_chunks=[RetrievedChunk(chunk=make_chunk("c1", reliability=0.95), similarity_score=0.9)], retry_count=0
    )
    decision = engine.evaluate(report)
    assert decision.action == DecisionAction.PROCEED
    assert decision.confidence_score >= settings.low_confidence_threshold


def test_marginal_confidence_returns_low_confidence_response(engine):
    report = VerificationReport(
        query="q", retrieved_chunks=[RetrievedChunk(chunk=make_chunk("c1", reliability=0.2), similarity_score=0.56)], retry_count=0
    )
    assert engine.evaluate(report).action == DecisionAction.LOW_CONFIDENCE_RESPONSE


def test_ocr_confidence_lowers_overall_confidence(engine):
    high_ocr = VerificationReport(query="q", retrieved_chunks=[RetrievedChunk(chunk=make_chunk("c1", ocr_conf=0.95), similarity_score=0.9)])
    low_ocr = VerificationReport(query="q", retrieved_chunks=[RetrievedChunk(chunk=make_chunk("c1", ocr_conf=0.3), similarity_score=0.9)])
    high_conf, _ = engine.compute_confidence(high_ocr)
    low_conf, _ = engine.compute_confidence(low_ocr)
    assert high_conf > low_conf


def test_confidence_score_always_bounded(engine):
    report = VerificationReport(
        query="q",
        retrieved_chunks=[RetrievedChunk(chunk=make_chunk("c1", reliability=1.0), similarity_score=1.0)],
        nli_results=[PairwiseNLIResult(chunk_id_a="c1", chunk_id_b="c1", relation=NLIRelation.CONTRADICTION, confidence=1.0)],
    )
    score, signals = engine.compute_confidence(report)
    assert 0.0 <= score <= 1.0
    assert len(signals) == 4


def test_decision_reasons_are_never_empty(engine):
    report = VerificationReport(query="q", retrieved_chunks=[], retry_count=0)
    assert len(engine.evaluate(report).reasons) > 0


def test_evaluate_accepts_tracing_context_without_error(engine):
    report = VerificationReport(query="q", retrieved_chunks=[], retry_count=0)
    decision = engine.evaluate(report, request_id="req-1", trace_id="trace-1", query_id="q-1")
    assert isinstance(decision, Decision)


def test_explain_returns_explainability_matching_decision(engine):
    report = VerificationReport(
        query="q", retrieved_chunks=[RetrievedChunk(chunk=make_chunk("c1", reliability=0.95), similarity_score=0.9)]
    )
    decision = engine.evaluate(report)
    explainability = engine.explain(decision)
    assert explainability.action == decision.action
    assert explainability.confidence == decision.confidence_score
    assert len(explainability.contributing_signals) == 4
    assert len(explainability.triggered_thresholds) >= 1


def test_explain_raises_on_decision_without_explainability(engine):
    bare_decision = Decision(action=DecisionAction.PROCEED, confidence_score=0.9, reasons=["ok"])
    with pytest.raises(DecisionEngineError):
        engine.explain(bare_decision)


def test_evaluate_wraps_unexpected_internal_errors(engine, monkeypatch):
    monkeypatch.setattr(engine, "_route", lambda report: (_ for _ in ()).throw(RuntimeError("boom")))
    report = VerificationReport(query="q", retrieved_chunks=[])
    with pytest.raises(DecisionEngineError):
        engine.evaluate(report)
