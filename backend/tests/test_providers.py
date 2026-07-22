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
