"""Unit 2.9 tests -- DiagnosticsBuilder."""
import pytest

from app.schemas.retrieval import NLIRelation, PairwiseNLIResult
from app.services.verification.diagnostics_builder import DiagnosticsBuilder


def test_no_contradictions_yields_nli_score_of_one():
    results = [PairwiseNLIResult(chunk_id_a="c1", chunk_id_b="c2", relation=NLIRelation.ENTAILMENT, confidence=0.9)]
    diag = DiagnosticsBuilder().build("q", results, evidence_coverage=1.0, unsupported_claims=[], reranker_confidence=0.8, verification_latency_ms=12.5)
    assert diag.nli_score == 1.0
    assert diag.contradiction_detected is False


def test_contradiction_lowers_nli_score_and_sets_flag():
    results = [PairwiseNLIResult(chunk_id_a="c1", chunk_id_b="c2", relation=NLIRelation.CONTRADICTION, confidence=0.7)]
    diag = DiagnosticsBuilder().build("q", results, evidence_coverage=1.0, unsupported_claims=[], reranker_confidence=0.8, verification_latency_ms=12.5)
    assert diag.contradiction_detected is True
    assert abs(diag.nli_score - 0.3) < 1e-9


def test_multiple_contradictions_uses_max_confidence():
    results = [
        PairwiseNLIResult(chunk_id_a="c1", chunk_id_b="c2", relation=NLIRelation.CONTRADICTION, confidence=0.3),
        PairwiseNLIResult(chunk_id_a="c1", chunk_id_b="c3", relation=NLIRelation.CONTRADICTION, confidence=0.9),
    ]
    diag = DiagnosticsBuilder().build("q", results, evidence_coverage=1.0, unsupported_claims=[], reranker_confidence=0.8, verification_latency_ms=1.0)
    assert abs(diag.nli_score - 0.1) < 1e-9


def test_empty_nli_results_yields_perfect_score_no_contradiction():
    diag = DiagnosticsBuilder().build("q", [], evidence_coverage=0.5, unsupported_claims=["c1"], reranker_confidence=0.0, verification_latency_ms=5.0)
    assert diag.nli_score == 1.0
    assert diag.contradiction_detected is False
    assert diag.unsupported_claims == ["c1"]


def test_all_fields_carried_through_correctly():
    diag = DiagnosticsBuilder().build("what is X?", [], evidence_coverage=0.75, unsupported_claims=["c9"], reranker_confidence=0.42, verification_latency_ms=99.9)
    assert diag.query == "what is X?"
    assert diag.evidence_coverage == 0.75
    assert diag.reranker_confidence == 0.42
    assert diag.verification_latency_ms == 99.9
