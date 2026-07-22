import pytest
from unittest.mock import AsyncMock, patch

from app.core.settings.ai_provider import AISettings
from app.providers.factory import AIProviderFactory
from app.providers.deterministic.deterministic_embedding import DeterministicEmbeddingProvider
from app.providers.deterministic.deterministic_llm import DeterministicLLMProvider
from app.providers.deterministic.deterministic_reranker import DeterministicRerankerProvider
from app.providers.gemini.gemini_embedding import GeminiEmbeddingProvider
from app.providers.gemini.gemini_llm import GeminiLLMProvider
from app.schemas.retrieval import Decision, DecisionAction, NLIRelation


def test_factory_deterministic():
    settings = AISettings(provider="deterministic")
    factory = AIProviderFactory(settings)

    embedder = factory.create_embedding_provider()
    llm = factory.create_llm_provider()
    reranker = factory.create_reranker_provider()

    assert isinstance(embedder, DeterministicEmbeddingProvider)
    assert isinstance(llm, DeterministicLLMProvider)
    assert isinstance(reranker, DeterministicRerankerProvider)


def test_factory_gemini():
    settings = AISettings(provider="gemini", gemini_api_key="test_key")
    factory = AIProviderFactory(settings)

    embedder = factory.create_embedding_provider()
    llm = factory.create_llm_provider()

    assert isinstance(embedder, GeminiEmbeddingProvider)
    assert isinstance(llm, GeminiLLMProvider)


@pytest.mark.asyncio
async def test_deterministic_provider_execution():
    provider = DeterministicLLMProvider()

    decision = Decision(action=DecisionAction.PROCEED, reasons=["All good"], confidence=0.95)
    answer = await provider.generate(decision=decision, query="test")
    assert isinstance(answer, str)
    assert "Based on verified evidence" in answer

    relation, conf = await provider.verify_pair("Premise text", "Hypothesis text")
    assert isinstance(relation, NLIRelation)
    assert 0.0 <= conf <= 1.0


from app.core.exceptions import SentinelRAGError


def test_factory_gemini_missing_key_raises_error():
    settings = AISettings(provider="gemini", gemini_api_key=None)
    factory = AIProviderFactory(settings)

    with pytest.raises(SentinelRAGError, match="GEMINI_API_KEY is missing"):
        factory.create_embedding_provider()

    with pytest.raises(SentinelRAGError, match="GEMINI_API_KEY is missing"):
        factory.create_llm_provider()


def test_gemini_embedding_provider_missing_key_raises_error():
    with pytest.raises(SentinelRAGError, match="requires a non-empty api_key"):
        GeminiEmbeddingProvider(api_key=None)


def test_gemini_llm_provider_missing_key_raises_error():
    with pytest.raises(SentinelRAGError, match="requires a non-empty api_key"):
        GeminiLLMProvider(api_key=None)


def test_gemini_model_name_normalization():
    provider_llm = GeminiLLMProvider(api_key="test_key", model="models/gemini-1.5-flash")
    assert provider_llm.model == "gemini-1.5-flash"
    assert provider_llm._endpoint == "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    provider_embed = GeminiEmbeddingProvider(api_key="test_key", model="models/text-embedding-004")
    assert provider_embed.model == "text-embedding-004"
    assert provider_embed._endpoint == "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"


def test_factory_gemini_invalid_model_prevalidation():
    settings = AISettings(provider="gemini", gemini_api_key="test_key", gemini_model="nonexistent-model")
    factory = AIProviderFactory(settings)

    mock_resp = patch("httpx.Client.get")
    with mock_resp as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]
        }
        llm = factory.create_llm_provider()
        assert llm.model == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_gemini_batch_embed_contents_success():
    provider = GeminiEmbeddingProvider(api_key="test_key")

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "embeddings": [
            {"values": [0.1, 0.2, 0.3]},
            {"values": [0.4, 0.5, 0.6]},
        ]
    }

    with patch("httpx.AsyncClient.post", return_value=mock_resp):
        res = await provider.embed_batch(["text 1", "text 2"])
        assert len(res) == 2
        assert res[0] == [0.1, 0.2, 0.3]
        assert res[1] == [0.4, 0.5, 0.6]


def test_gemini_2_5_flash_direct_support():
    provider = GeminiLLMProvider(api_key="test_key", model="gemini-2.5-flash")
    assert provider.model == "gemini-2.5-flash"
    assert provider._endpoint == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
