"""
Unit 2.13 tests -- LLMBasedNLIVerifier's request/response/error logic,
tested WITHOUT any real network call via httpx.MockTransport.
"""
import json

import httpx
import pytest

from app.core.exceptions import VerificationError
from app.core.settings.nli import NLISettings
from app.schemas.embedding import EmbedderHealthState
from app.schemas.retrieval import NLIRelation
from app.services.verification.llm_nli_verifier import LLMBasedNLIVerifier
from app.services.verification.nli_base import BaseNLIVerifier


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="https://fake-llm.test/v1", transport=httpx.MockTransport(handler))


def chat_response(label: str, confidence: float, model="test-model") -> httpx.Response:
    content = json.dumps({"label": label, "confidence": confidence})
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}], "model": model},
    )


def make_handler(label, confidence):
    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response(label, confidence)
    return handler


def make_verifier(handler, settings=None):
    settings = settings or NLISettings(model_name="test-model")
    return LLMBasedNLIVerifier(make_client(handler), settings)


def test_is_a_valid_base_nli_verifier():
    assert issubclass(LLMBasedNLIVerifier, BaseNLIVerifier)


@pytest.mark.asyncio
async def test_entailment_label_parsed_correctly():
    verifier = make_verifier(make_handler("entailment", 0.85))
    relation, confidence = await verifier.verify_pair("a", "b")
    assert relation == NLIRelation.ENTAILMENT
    assert confidence == 0.85


@pytest.mark.asyncio
async def test_contradiction_label_parsed_correctly():
    verifier = make_verifier(make_handler("contradiction", 0.7))
    relation, confidence = await verifier.verify_pair("a", "b")
    assert relation == NLIRelation.CONTRADICTION


@pytest.mark.asyncio
async def test_neutral_label_parsed_correctly():
    verifier = make_verifier(make_handler("neutral", 0.4))
    relation, _ = await verifier.verify_pair("a", "b")
    assert relation == NLIRelation.NEUTRAL


@pytest.mark.asyncio
async def test_label_parsing_is_case_insensitive():
    verifier = make_verifier(make_handler("CONTRADICTION", 0.9))
    relation, _ = await verifier.verify_pair("a", "b")
    assert relation == NLIRelation.CONTRADICTION


@pytest.mark.asyncio
async def test_confidence_clamped_to_unit_range():
    verifier = make_verifier(make_handler("entailment", 1.5))
    _, confidence = await verifier.verify_pair("a", "b")
    assert confidence == 1.0


# --- Error handling ---

def error_handler_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "internal"})


def unparseable_json_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": "not json at all"}}], "model": "x"})


def unknown_label_handler(request: httpx.Request) -> httpx.Response:
    content = json.dumps({"label": "definitely_maybe", "confidence": 0.5})
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "model": "x"})


def missing_choices_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"unexpected": "shape"})


@pytest.mark.asyncio
async def test_5xx_raises_verification_error():
    verifier = make_verifier(error_handler_500)
    with pytest.raises(VerificationError):
        await verifier.verify_pair("a", "b")


@pytest.mark.asyncio
async def test_unparseable_json_content_raises_verification_error():
    verifier = make_verifier(unparseable_json_handler)
    with pytest.raises(VerificationError):
        await verifier.verify_pair("a", "b")


@pytest.mark.asyncio
async def test_unknown_label_raises_verification_error():
    verifier = make_verifier(unknown_label_handler)
    with pytest.raises(VerificationError):
        await verifier.verify_pair("a", "b")


@pytest.mark.asyncio
async def test_missing_choices_raises_verification_error():
    verifier = make_verifier(missing_choices_handler)
    with pytest.raises(VerificationError):
        await verifier.verify_pair("a", "b")


@pytest.mark.asyncio
async def test_never_retries_internally():
    call_count = {"n": 0}

    def counting_handler(request):
        call_count["n"] += 1
        return error_handler_500(request)

    verifier = make_verifier(counting_handler)
    with pytest.raises(VerificationError):
        await verifier.verify_pair("a", "b")
    assert call_count["n"] == 1  # zero internal retry -- ContradictionDetector owns degradation


# --- Health tracking (reused EmbedderHealth) ---

@pytest.mark.asyncio
async def test_health_starts_ready():
    verifier = make_verifier(make_handler("neutral", 0.5))
    assert verifier.health().state == EmbedderHealthState.READY


@pytest.mark.asyncio
async def test_health_degrades_after_failure_below_threshold():
    settings = NLISettings(model_name="test-model", unavailable_after_consecutive_failures=3)
    verifier = make_verifier(error_handler_500, settings)
    with pytest.raises(VerificationError):
        await verifier.verify_pair("a", "b")
    assert verifier.health().state == EmbedderHealthState.DEGRADED


@pytest.mark.asyncio
async def test_health_becomes_unavailable_after_threshold():
    settings = NLISettings(model_name="test-model", unavailable_after_consecutive_failures=2)
    verifier = make_verifier(error_handler_500, settings)
    for _ in range(2):
        with pytest.raises(VerificationError):
            await verifier.verify_pair("a", "b")
    assert verifier.health().state == EmbedderHealthState.UNAVAILABLE


@pytest.mark.asyncio
async def test_health_recovers_after_success_following_failure():
    call_count = {"n": 0}

    def flaky_handler(request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return error_handler_500(request)
        return chat_response("neutral", 0.5)

    verifier = make_verifier(flaky_handler)
    with pytest.raises(VerificationError):
        await verifier.verify_pair("a", "b")
    assert verifier.health().state == EmbedderHealthState.DEGRADED
    await verifier.verify_pair("a", "b")
    assert verifier.health().state == EmbedderHealthState.READY


# --- NLIResult ---

@pytest.mark.asyncio
async def test_verify_pair_with_result_shape():
    verifier = make_verifier(make_handler("entailment", 0.8))
    result = await verifier.verify_pair_with_result("a", "b")
    assert result.label == NLIRelation.ENTAILMENT
    assert result.provider == "llm_chat_completions"
    assert result.model_name == "test-model"
    assert result.confidence == 0.8
    assert result.latency_ms >= 0
