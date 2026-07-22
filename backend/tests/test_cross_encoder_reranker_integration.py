"""
Unit 2.12 -- REAL network integration test for CrossEncoderReranker.

Isolated and optional, identical pattern to Units 2.10/2.11's live tests:
marked @pytest.mark.integration, excluded from the default run, skips
cleanly (not fails) without a reachable, authenticated endpoint.

Run explicitly, e.g. against Cohere's rerank API:

    RERANKING__API_BASE_URL=https://api.cohere.ai/v1 \\
    RERANKING__API_KEY=... \\
    RERANKING__MODEL_NAME=rerank-english-v3.0 \\
    PYTHONPATH=. pytest tests/test_cross_encoder_reranker_integration.py -m integration -v

This sandbox has no network route to any reranking provider, so this
file has only ever been run in its skip path here.
"""
import pytest

from app.core.settings.reranking import RerankingSettings
from app.infrastructure.reranking_client_factory import close_reranking_http_client, create_reranking_http_client
from app.schemas.retrieval import Chunk, RetrievedChunk
from app.services.reranking.cross_encoder_reranker import CrossEncoderReranker

pytestmark = pytest.mark.integration


def make_candidate(chunk_id, text):
    chunk = Chunk(chunk_id=chunk_id, document_id="doc-1", text=text, token_count=10, source_reliability_score=0.9)
    return RetrievedChunk(chunk=chunk, similarity_score=0.5)


@pytest.fixture
async def live_reranker():
    settings = RerankingSettings()
    if not settings.api_key and "localhost" not in settings.api_base_url and "127.0.0.1" not in settings.api_base_url:
        pytest.skip("No RERANKING__API_KEY configured and api_base_url is not a local endpoint -- skipping live test.")

    client = create_reranking_http_client(settings)
    reranker = CrossEncoderReranker(client, settings)

    try:
        await reranker.rerank("connectivity check", [make_candidate("c1", "test document")], top_n=1)
    except Exception as exc:  # noqa: BLE001
        await close_reranking_http_client(client)
        pytest.skip(f"Configured reranking endpoint not reachable: {exc}")

    yield reranker
    await close_reranking_http_client(client)


@pytest.mark.asyncio
async def test_rerank_returns_scored_results(live_reranker):
    candidates = [
        make_candidate("c1", "The refund policy allows returns within 30 days."),
        make_candidate("c2", "Our office is located in downtown Seattle."),
    ]
    result = await live_reranker.rerank("what is the refund policy?", candidates, top_n=2)
    assert len(result) == 2
    assert result[0].rerank_score is not None


@pytest.mark.asyncio
async def test_rerank_orders_relevant_document_first(live_reranker):
    candidates = [
        make_candidate("c1", "The weather today is sunny with a chance of rain."),
        make_candidate("c2", "Refunds are processed within 5-7 business days of approval."),
    ]
    result = await live_reranker.rerank("how long do refunds take?", candidates, top_n=2)
    assert result[0].retrieved_chunk.chunk.chunk_id == "c2"
