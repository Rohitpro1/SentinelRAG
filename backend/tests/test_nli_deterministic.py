"""Unit 2.9 tests -- DeterministicNLIVerifier."""
import pytest

from app.schemas.retrieval import NLIRelation
from app.services.verification.nli_base import BaseNLIVerifier
from app.services.verification.nli_deterministic import DeterministicNLIVerifier


@pytest.mark.asyncio
async def test_is_a_valid_base_nli_verifier():
    assert isinstance(DeterministicNLIVerifier(), BaseNLIVerifier)


@pytest.mark.asyncio
async def test_marker_in_both_texts_triggers_contradiction():
    verifier = DeterministicNLIVerifier(conflict_marker="[CONTRADICTION]")
    relation, confidence = await verifier.verify_pair(
        "Refunds are allowed within 30 days. [CONTRADICTION]",
        "Refunds are never allowed. [CONTRADICTION]",
    )
    assert relation == NLIRelation.CONTRADICTION
    assert 0.0 <= confidence <= 1.0


@pytest.mark.asyncio
async def test_marker_in_only_one_text_does_not_trigger_contradiction():
    verifier = DeterministicNLIVerifier(conflict_marker="[CONTRADICTION]")
    relation, _ = await verifier.verify_pair("Normal text. [CONTRADICTION]", "Other normal text.")
    assert relation != NLIRelation.CONTRADICTION


@pytest.mark.asyncio
async def test_result_is_deterministic_across_calls():
    verifier = DeterministicNLIVerifier()
    r1 = await verifier.verify_pair("text a", "text b")
    r2 = await verifier.verify_pair("text a", "text b")
    assert r1 == r2


@pytest.mark.asyncio
async def test_result_is_order_independent():
    verifier = DeterministicNLIVerifier()
    r1 = await verifier.verify_pair("alpha", "beta")
    r2 = await verifier.verify_pair("beta", "alpha")
    assert r1 == r2


@pytest.mark.asyncio
async def test_no_marker_produces_entailment_or_neutral_only():
    verifier = DeterministicNLIVerifier()
    relation, _ = await verifier.verify_pair("some text", "some other text")
    assert relation in (NLIRelation.ENTAILMENT, NLIRelation.NEUTRAL)


# --- Unit 2.13 additions: NLIResult / EmbedderHealth (reused) ---

@pytest.mark.asyncio
async def test_verify_pair_with_result_shape():
    from app.schemas.embedding import EmbedderHealthState

    verifier = DeterministicNLIVerifier()
    result = await verifier.verify_pair_with_result("text a", "text b")
    assert result.provider == "deterministic"
    assert result.model_name == "deterministic-sha256-marker"
    assert result.model_version == "v1"
    assert result.latency_ms >= 0
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_verify_pair_with_result_label_matches_plain_verify_pair():
    verifier = DeterministicNLIVerifier(conflict_marker="[X]")
    direct_relation, direct_confidence = await verifier.verify_pair("a [X]", "b [X]")
    result = await verifier.verify_pair_with_result("a [X]", "b [X]")
    assert result.label == direct_relation
    assert result.confidence == direct_confidence


def test_health_always_ready():
    from app.schemas.embedding import EmbedderHealthState

    verifier = DeterministicNLIVerifier()
    assert verifier.health().state == EmbedderHealthState.READY
