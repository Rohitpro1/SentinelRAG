import pytest
from unittest.mock import AsyncMock, patch

from app.core.exceptions import SentinelRAGError, EmbeddingError
from app.providers.voyage.voyage_embedding import VoyageEmbeddingProvider


def test_voyage_provider_initialization():
    provider = VoyageEmbeddingProvider(api_key="test_voyage_key", model="voyage-3-large")
    assert provider.model == "voyage-3-large"
    assert provider.dimensions == 1024
    assert "voyage-3-large" in provider.name


def test_voyage_provider_missing_key():
    with pytest.raises(SentinelRAGError, match="requires a non-empty api_key"):
        VoyageEmbeddingProvider(api_key=None)


@pytest.mark.asyncio
async def test_voyage_embed_query_success():
    provider = VoyageEmbeddingProvider(api_key="test_voyage_key")

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"embedding": [0.1] * 1024}
        ]
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        vec = await provider.embed_query("test query")
        assert len(vec) == 1024
        assert vec[0] == 0.1


@pytest.mark.asyncio
async def test_voyage_embed_batch_success():
    provider = VoyageEmbeddingProvider(api_key="test_voyage_key")

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"embedding": [0.1] * 1024},
            {"embedding": [0.2] * 1024},
        ]
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        vecs = await provider.embed_batch(["t1", "t2"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 1024
        assert vecs[1][0] == 0.2
