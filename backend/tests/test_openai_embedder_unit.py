"""
Unit 2.11 tests -- OpenAIEmbedder's request/response/error-handling logic,
tested WITHOUT any real network call via httpx.MockTransport, per the
project's deterministic-testing-by-default principle (instruction 4:
comprehensive tests, real network isolated to the integration test file).
"""
import json

import httpx
import pytest

from app.core.exceptions import EmbeddingError
from app.core.settings.embedding import EmbeddingSettings
from app.schemas.embedding import EmbedderHealthState
from app.services.embedding.base import BaseEmbedder
from app.services.embedding.openai_embedder import OpenAIEmbedder


def make_client(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(base_url="https://fake-provider.test/v1", transport=transport)


def success_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    texts = body["input"]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": i} for i in range(len(texts))],
            "model": body["model"],
        },
    )


def make_embedder(handler, settings=None):
    settings = settings or EmbeddingSettings(model_name="test-model", dimensions=3)
    return OpenAIEmbedder(make_client(handler), settings)


def test_is_a_valid_base_embedder():
    assert issubclass(OpenAIEmbedder, BaseEmbedder)


@pytest.mark.asyncio
async def test_embed_query_returns_vector_from_response():
    embedder = make_embedder(success_handler)
    vector = await embedder.embed_query("hello world")
    assert vector == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_batch_returns_one_vector_per_text():
    embedder = make_embedder(success_handler)
    vectors = await embedder.embed_batch(["a", "b", "c"])
    assert len(vectors) == 3


@pytest.mark.asyncio
async def test_embed_batch_empty_list_short_circuits_no_request():
    calls = []

    def handler(request):
        calls.append(request)
        return success_handler(request)

    embedder = make_embedder(handler)
    result = await embedder.embed_batch([])
    assert result == []
    assert calls == []


@pytest.mark.asyncio
async def test_dimensions_reflects_settings():
    settings = EmbeddingSettings(model_name="test-model", dimensions=1536)
    embedder = make_embedder(success_handler, settings)
    assert embedder.dimensions == 1536


# --- Error handling ---

def error_handler_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "internal server error"})


def error_handler_401(request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json={"error": "invalid api key"})


def malformed_response_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"unexpected": "shape"})


@pytest.mark.asyncio
async def test_5xx_response_raises_embedding_error():
    embedder = make_embedder(error_handler_500)
    with pytest.raises(EmbeddingError):
        await embedder.embed_query("hello")


@pytest.mark.asyncio
async def test_401_response_raises_embedding_error():
    embedder = make_embedder(error_handler_401)
    with pytest.raises(EmbeddingError):
        await embedder.embed_query("hello")


@pytest.mark.asyncio
async def test_embedding_error_is_always_transient_per_frozen_design():
    embedder = make_embedder(error_handler_401)
    try:
        await embedder.embed_query("hello")
        pytest.fail("expected EmbeddingError")
    except EmbeddingError as exc:
        assert exc.transient is True  # frozen decision from Unit 2.6 -- no 4xx/5xx distinction


@pytest.mark.asyncio
async def test_malformed_response_raises_embedding_error():
    embedder = make_embedder(malformed_response_handler)
    with pytest.raises(EmbeddingError):
        await embedder.embed_query("hello")


@pytest.mark.asyncio
async def test_embed_query_never_retries_internally():
    call_count = {"n": 0}

    def counting_error_handler(request):
        call_count["n"] += 1
        return error_handler_500(request)

    embedder = make_embedder(counting_error_handler)
    with pytest.raises(EmbeddingError):
        await embedder.embed_query("hello")
    assert call_count["n"] == 1  # zero internal retry, mirrors Unit 2.10's search()


@pytest.mark.asyncio
async def test_embed_batch_retries_once_then_succeeds():
    call_count = {"n": 0}

    def flaky_handler(request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return error_handler_500(request)
        return success_handler(request)

    embedder = make_embedder(flaky_handler)
    vectors = await embedder.embed_batch(["a", "b"])
    assert len(vectors) == 2
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_embed_batch_gives_up_after_bounded_retry():
    call_count = {"n": 0}

    def always_failing_handler(request):
        call_count["n"] += 1
        return error_handler_500(request)

    embedder = make_embedder(always_failing_handler)
    with pytest.raises(EmbeddingError):
        await embedder.embed_batch(["a"])
    assert call_count["n"] == 2  # initial + 1 bounded retry, then gives up


# --- Health tracking ---

@pytest.mark.asyncio
async def test_health_starts_ready():
    embedder = make_embedder(success_handler)
    assert embedder.health().state == EmbedderHealthState.READY


@pytest.mark.asyncio
async def test_health_degrades_after_failures_below_threshold():
    settings = EmbeddingSettings(model_name="test-model", dimensions=3, unavailable_after_consecutive_failures=3)
    embedder = make_embedder(error_handler_500, settings)
    with pytest.raises(EmbeddingError):
        await embedder.embed_query("x")
    assert embedder.health().state == EmbedderHealthState.DEGRADED


@pytest.mark.asyncio
async def test_health_becomes_unavailable_after_threshold_reached():
    settings = EmbeddingSettings(model_name="test-model", dimensions=3, unavailable_after_consecutive_failures=2)
    embedder = make_embedder(error_handler_500, settings)
    for _ in range(2):
        with pytest.raises(EmbeddingError):
            await embedder.embed_query("x")
    assert embedder.health().state == EmbedderHealthState.UNAVAILABLE


@pytest.mark.asyncio
async def test_health_recovers_to_ready_after_success_following_failure():
    call_count = {"n": 0}

    def flaky_handler(request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return error_handler_500(request)
        return success_handler(request)

    embedder = make_embedder(flaky_handler)
    with pytest.raises(EmbeddingError):
        await embedder.embed_query("x")
    assert embedder.health().state == EmbedderHealthState.DEGRADED

    await embedder.embed_query("y")
    assert embedder.health().state == EmbedderHealthState.READY


# --- EmbeddingResult ---

@pytest.mark.asyncio
async def test_embed_query_with_result_shape():
    embedder = make_embedder(success_handler)
    result = await embedder.embed_query_with_result("hello")
    assert result.vector == [0.1, 0.2, 0.3]
    assert result.provider == "openai_compatible"
    assert result.model_name == "test-model"
    assert result.embedding_dimensions == 3
    assert result.generation_latency_ms >= 0
