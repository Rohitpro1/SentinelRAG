"""
Unit 3.1 tests -- GraphState: defaults, validation, serialization
round-trip, and the effective_query convenience property.
"""
import pytest
from pydantic import ValidationError

from app.schemas.retrieval import Chunk, Decision, DecisionAction, RetrievedChunk
from app.schemas.retrieval_domain import VerificationDiagnostics, VerifiedEvidence
from app.orchestration.graph_state import GraphState


def test_defaults():
    state = GraphState(original_query="what is the refund policy?")
    assert state.rewritten_query is None
    assert state.retrieval_result is None
    assert state.verification_result is None
    assert state.decision is None
    assert state.retry_count == 0
    assert state.diagnostics is None


def test_original_query_is_required():
    with pytest.raises(ValidationError):
        GraphState()


def test_retry_count_rejects_negative():
    with pytest.raises(ValidationError):
        GraphState(original_query="q", retry_count=-1)


def test_effective_query_falls_back_to_original_when_no_rewrite():
    state = GraphState(original_query="original text")
    assert state.effective_query == "original text"


def test_effective_query_prefers_rewritten_query_when_present():
    state = GraphState(original_query="original text", rewritten_query="rewritten text")
    assert state.effective_query == "rewritten text"


def test_serialization_round_trip_with_populated_nested_objects():
    """
    GraphState is built entirely from existing Milestone 2 schemas
    (instruction 2's implicit requirement) -- this test constructs one
    with real nested objects (Decision, VerifiedEvidence,
    VerificationDiagnostics) and confirms model_dump() -> model_validate()
    preserves everything, including nested Pydantic objects.
    """
    chunk = Chunk(chunk_id="c1", document_id="doc-1", text="refund content", token_count=10, source_reliability_score=0.9)
    evidence = VerifiedEvidence(
        query="q", retrieved_chunks=[RetrievedChunk(chunk=chunk, similarity_score=0.9)], retry_count=1
    )
    decision = Decision(action=DecisionAction.PROCEED, confidence_score=0.8, reasons=["confidence sufficient"])
    diagnostics = VerificationDiagnostics(
        query="q", nli_score=1.0, contradiction_detected=False, evidence_coverage=1.0,
        unsupported_claims=[], reranker_confidence=0.8, verification_latency_ms=5.0,
    )

    original = GraphState(
        original_query="q", rewritten_query="q rewritten", verification_result=evidence,
        decision=decision, retry_count=1, diagnostics=diagnostics,
    )

    dumped = original.model_dump()
    restored = GraphState.model_validate(dumped)

    assert restored == original
    assert restored.decision.action == DecisionAction.PROCEED
    assert restored.verification_result.retrieved_chunks[0].chunk.chunk_id == "c1"


def test_json_round_trip():
    """JSON (not just dict) round-trip, since a real graph checkpointer would serialize to JSON."""
    original = GraphState(original_query="q", retry_count=2)
    restored = GraphState.model_validate_json(original.model_dump_json())
    assert restored == original


def test_reconstructing_from_partial_dict_fills_defaults():
    """
    Documents actual observed LangGraph behavior (see UNIT_3_1.md):
    ainvoke()'s raw output dict omits unset/None-valued fields entirely.
    GraphState(**partial_dict) must still reconstruct a valid, correctly-
    defaulted state from such a partial dict.
    """
    partial = {"original_query": "q", "retry_count": 3}  # simulates ainvoke()'s trimmed output
    state = GraphState(**partial)
    assert state.retry_count == 3
    assert state.rewritten_query is None
    assert state.decision is None
