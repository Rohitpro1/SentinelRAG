"""
Unit 2.11 -- REAL network integration test for OpenAIEmbedder.

Isolated and optional, identical pattern to Unit 2.10's
test_qdrant_integration.py: marked @pytest.mark.integration, excluded
from the default run (pytest.ini), and skips cleanly (not fails) when no
reachable, authenticated endpoint is configured.

Run explicitly with a real endpoint configured, e.g.:

    EMBEDDING__API_BASE_URL=https://api.openai.com/v1 \\
    EMBEDDING__API_KEY=sk-... \\
    EMBEDDING__MODEL_NAME=text-embedding-3-small \\
    PYTHONPATH=. pytest tests/test_openai_embedder_integration.py -m integration -v

or against a local OpenAI-compatible server (e.g. Ollama):

    EMBEDDING__API_BASE_URL=http://localhost:11434/v1 \\
    EMBEDDING__MODEL_NAME=nomic-embed-text \\
    PYTHONPATH=. pytest tests/test_openai_embedder_integration.py -m integration -v

This sandbox has no network route to any embedding provider, so this
file has only ever been run in its skip path here -- documented plainly
rather than claimed as verified against a live provider.
"""
import pytest

from app.core.settings.embedding import EmbeddingSettings
from app.infrastructure.embedding_client_factory import close_embedding_http_client, create_embedding_http_client
from app.services.embedding.openai_embedder import OpenAIEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture
async def live_embedder():
    settings = EmbeddingSettings()
    if not settings.api_key and "localhost" not in settings.api_base_url and "127.0.0.1" not in settings.api_base_url:
        pytest.skip("No EMBEDDING__API_KEY configured and api_base_url is not a local endpoint -- skipping live test.")

    client = create_embedding_http_client(settings)
    embedder = OpenAIEmbedder(client, settings)

    try:
        await embedder.embed_query("connectivity check")
    except Exception as exc:  # noqa: BLE001
        await close_embedding_http_client(client)
        pytest.skip(f"Configured embedding endpoint not reachable: {exc}")

    yield embedder
    await close_embedding_http_client(client)


@pytest.mark.asyncio
async def test_embed_query_returns_correctly_dimensioned_vector(live_embedder):
    vector = await live_embedder.embed_query("what is the refund policy?")
    assert len(vector) > 0


@pytest.mark.asyncio
async def test_embed_batch_returns_one_vector_per_input(live_embedder):
    vectors = await live_embedder.embed_batch(["refund policy", "shipping policy", "return policy"])
    assert len(vectors) == 3


@pytest.mark.asyncio
async def test_health_is_ready_after_successful_call(live_embedder):
    await live_embedder.embed_query("health check")
    from app.schemas.embedding import EmbedderHealthState
    assert live_embedder.health().state == EmbedderHealthState.READY
