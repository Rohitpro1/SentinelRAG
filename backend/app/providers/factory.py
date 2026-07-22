from __future__ import annotations

import logging
from typing import Optional

from app.core.settings.ai_provider import AISettings
from app.providers.base.embedding_provider import BaseEmbeddingProvider
from app.providers.base.llm_provider import BaseLLMProvider
from app.providers.base.reranker_provider import BaseRerankerProvider
from app.providers.deterministic.deterministic_embedding import DeterministicEmbeddingProvider
from app.providers.deterministic.deterministic_llm import DeterministicLLMProvider
from app.providers.deterministic.deterministic_reranker import DeterministicRerankerProvider
from app.providers.gemini.gemini_embedding import GeminiEmbeddingProvider
from app.providers.gemini.gemini_llm import GeminiLLMProvider

logger = logging.getLogger(__name__)


class AIProviderFactory:
    """
    Factory for instantiating AI Providers based on AISettings configuration.
    Supports AI_PROVIDER values: "gemini", "deterministic".
    """

    def __init__(self, settings: Optional[AISettings] = None) -> None:
        self.settings = settings or AISettings()

    def create_embedding_provider(self) -> BaseEmbeddingProvider:
        provider_name = self.settings.provider.lower().strip()
        if provider_name == "gemini":
            logger.info("Initializing Gemini Embedding Provider (%s)", self.settings.gemini_embedding_model)
            return GeminiEmbeddingProvider(
                api_key=self.settings.gemini_api_key,
                model=self.settings.gemini_embedding_model,
            )
        return DeterministicEmbeddingProvider()

    def create_llm_provider(self) -> BaseLLMProvider:
        provider_name = self.settings.provider.lower().strip()
        if provider_name == "gemini":
            logger.info("Initializing Gemini LLM Provider (%s)", self.settings.gemini_model)
            return GeminiLLMProvider(
                api_key=self.settings.gemini_api_key,
                model=self.settings.gemini_model,
            )
        return DeterministicLLMProvider()

    def create_reranker_provider(self) -> BaseRerankerProvider:
        return DeterministicRerankerProvider()
