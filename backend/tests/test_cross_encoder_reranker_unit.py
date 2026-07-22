"""
Unit 2.12 tests -- CrossEncoderReranker's request/response/batching/error
logic, tested WITHOUT any real network call via httpx.MockTransport.
"""
import json

import httpx
import pytest

from app.core.exceptions import RerankError
from app.core.settings.reranking import RerankingSettings
from app.schemas.retrieval import Chunk, RetrievedChunk
from app.services.reranking.base import BaseReranker
from app.services.reranking.cross_encoder_reranker import CrossEncoderReranker


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="https://fake-reranker.test/v1", transport=httpx.MockTransport(handler))


def make_candidate(chunk_id, text, similarity=0.5):
    chunk = Chunk(chunk_id=chunk_id, document_id="doc-1", text=text, token_count=10, source_reliability_score=0.9)
    return RetrievedChunk(chunk=chunk, similarity_score=similarity)


def success_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    documents = body["documents"]
    # Return in reverse order with descending scores, so tests can
    # distinguish "provider reordered these" from "passthrough order".
    results = [
        {"index": i, "relevance_score": 1.0 - (i * 0.1)}
        for i in reversed(range(len(documents)))
    ]
    return httpx.Response(200, json={"results": results, "model": body["model"]})


def make_reranker(handler, settings=None):
    settings = settings or RerankingSettings(model_name="test-rerank-model")
    return CrossEncoderReranker(make_client(handler), settings)


def test_is_a_valid_base_reranker():
    assert issubclass(CrossEncoderReranker, BaseReranker)


@pytest.mark.asyncio
async def test_empty_candidates_returns_empty_no_request():
    calls = []

    def handler(request):
        calls.append(request)
        return success_handler(request)

    reranker = make_reranker(handler)
    result = await reranker.rerank("q", [], top_n=5)
    assert result == []
    assert calls == []


@pytest.mark.asyncio
async def test_rerank_respects_top_n():
    reranker = make_reranker(success_handler)
    candidates = [make_candidate(f"c{i}", f"text {i}") for i in range(10)]
    result = await reranker.rerank("q", candidates, top_n=3)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_rerank_scores_come_from_provider_response():
    reranker = make_reranker(success_handler)
    candidates = [make_candidate("c0", "a"), make_candidate("c1", "b")]
    result = await reranker.rerank("q", candidates, top_n=2)
    assert result[0].rerank_score == 1.0
    assert result[1].rerank_score == 0.9


@pytest.mark.asyncio
async def test_rerank_assigns_sequential_ranks():
    reranker = make_reranker(success_handler)
    candidates = [make_candidate(f"c{i}", f"text {i}") for i in range(4)]
    result = await reranker.rerank("q", candidates, top_n=4)
    assert [r.rank for r in result] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_rerank_sorted_descending_by_relevance():
    reranker = make_reranker(success_handler)
    candidates = [make_candidate(f"c{i}", f"text {i}") for i in range(5)]
    result = await reranker.rerank("q", candidates, top_n=5)
    scores = [r.rerank_score for r in result]
    assert scores == sorted(scores, reverse=True)


# --- Batching (configuration-driven, instruction 2) ---

@pytest.mark.asyncio
async def test_batches_split_according_to_max_batch_size():
    call_sizes = []

    def counting_handler(request):
        body = json.loads(request.content)
        call_sizes.append(len(body["documents"]))
        return success_handler(request)

    settings = RerankingSettings(model_name="test-model", max_batch_size=3)
    reranker = CrossEncoderReranker(make_client(counting_handler), settings)
    candidates = [make_candidate(f"c{i}", f"text {i}") for i in range(7)]
    await reranker.rerank("q", candidates, top_n=7)
    assert call_sizes == [3, 3, 1]  # 7 candidates split into batches of 3


@pytest.mark.asyncio
async def test_supports_batching_false_sends_one_request_per_candidate():
    call_count = {"n": 0}

    def counting_handler(request):
        call_count["n"] += 1
        return success_handler(request)

    settings = RerankingSettings(model_name="test-model", supports_batching=False)
    reranker = CrossEncoderReranker(make_client(counting_handler), settings)
    candidates = [make_candidate(f"c{i}", f"text {i}") for i in range(3)]
    await reranker.rerank("q", candidates, top_n=3)
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_results_merged_correctly_across_batches():
    settings = RerankingSettings(model_name="test-model", max_batch_size=2)
    reranker = CrossEncoderReranker(make_client(success_handler), settings)
    candidates = [make_candidate(f"c{i}", f"text {i}") for i in range(5)]
    result = await reranker.rerank("q", candidates, top_n=5)
    assert len(result) == 5
    assert len({r.retrieved_chunk.chunk.chunk_id for r in result}) == 5  # no duplicates/dropped items


# --- Error handling ---

def error_handler_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "internal error"})


def malformed_response_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"unexpected": "shape"})


@pytest.mark.asyncio
async def test_5xx_raises_rerank_error():
    reranker = make_reranker(error_handler_500)
    with pytest.raises(RerankError):
        await reranker.rerank("q", [make_candidate("c1", "x")], top_n=1)


@pytest.mark.asyncio
async def test_malformed_response_raises_rerank_error():
    reranker = make_reranker(malformed_response_handler)
    with pytest.raises(RerankError):
        await reranker.rerank("q", [make_candidate("c1", "x")], top_n=1)


@pytest.mark.asyncio
async def test_rerank_never_retries_internally():
    call_count = {"n": 0}

    def counting_error_handler(request):
        call_count["n"] += 1
        return error_handler_500(request)

    reranker = make_reranker(counting_error_handler)
    with pytest.raises(RerankError):
        await reranker.rerank("q", [make_candidate("c1", "x")], top_n=1)
    assert call_count["n"] == 1  # zero internal retry -- RerankingService owns degradation


# --- Capabilities (configuration-driven, instruction 2) ---

def test_capabilities_reflects_settings_not_hardcoded():
    settings = RerankingSettings(
        model_name="test-model", supports_batching=False, max_batch_size=7, max_input_tokens=256, model_dimensions=384,
    )
    reranker = CrossEncoderReranker(make_client(success_handler), settings)
    caps = reranker.capabilities()
    assert caps.supports_batching is False
    assert caps.max_batch_size == 7
    assert caps.max_input_tokens == 256
    assert caps.model_dimensions == 384


def test_capabilities_never_makes_a_network_call():
    calls = []

    def handler(request):
        calls.append(request)
        return success_handler(request)

    reranker = CrossEncoderReranker(make_client(handler), RerankingSettings(model_name="test-model"))
    reranker.capabilities()
    assert calls == []  # purely configuration-driven, no runtime discovery request


# --- RerankResult ---

@pytest.mark.asyncio
async def test_rerank_with_result_shape():
    reranker = make_reranker(success_handler)
    result = await reranker.rerank_with_result("q", [make_candidate("c1", "x"), make_candidate("c2", "y")], top_n=2)
    assert len(result.ranked_chunks) == 2
    assert result.provider == "cross_encoder_http"
    assert result.model_name == "test-rerank-model"
    assert result.rerank_latency_ms >= 0
