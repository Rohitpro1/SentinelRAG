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

import os
from app.core.exceptions import SentinelRAGError
from app.providers.voyage.voyage_embedding import VoyageEmbeddingProvider
from app.providers.router.embedding_router import EmbeddingRouterProvider

logger = logging.getLogger(__name__)


class AIProviderFactory:
    """
    Factory for instantiating AI Providers based on AISettings configuration.
    Supports AI_PROVIDER values: "gemini", "deterministic".
    """

    def __init__(self, settings: Optional[AISettings] = None) -> None:
        self.settings = settings or AISettings()
        self._log_startup_configuration()

    def _resolve_gemini_key_and_source(self) -> tuple[Optional[str], str]:
        if self.settings.gemini_api_key and self.settings.gemini_api_key.strip():
            return self.settings.gemini_api_key.strip(), "AISettings"
        
        env_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("AI_GEMINI_API_KEY")
        if env_gemini and env_gemini.strip():
            source = "GEMINI_API_KEY" if os.getenv("GEMINI_API_KEY") else "AI_GEMINI_API_KEY"
            return env_gemini.strip(), f"env:{source}"
        
        return None, "missing"

    def _log_startup_configuration(self) -> None:
        provider_name = self.settings.provider.lower().strip()
        key, source = self._resolve_gemini_key_and_source()
        if provider_name == "gemini":
            if key:
                logger.info(
                    "AIProviderFactory: Configured AI_PROVIDER='gemini'. Gemini API key present (source: %s, length: %d chars).",
                    source,
                    len(key),
                )
            else:
                logger.error(
                    "AIProviderFactory: Configured AI_PROVIDER='gemini' but no GEMINI_API_KEY found in environment or settings."
                )
        else:
            logger.info("AIProviderFactory: Configured AI_PROVIDER='%s'. Using deterministic providers.", provider_name)

    def _validate_gemini_model(self, api_key: str, model_name: str, expected_method: str) -> None:
        import httpx
        from app.providers.gemini.gemini_llm import resolve_gemini_model_name

        clean_name = resolve_gemini_model_name(model_name)
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url)
                if resp.status_code != 200:
                    logger.warning("Could not pre-validate Gemini model via GET /models (HTTP %d). Proceeding.", resp.status_code)
                    return
                data = resp.json()
                models = data.get("models", [])
                available_models = [m.get("name", "").removeprefix("models/") for m in models]
                if available_models and clean_name not in available_models:
                    logger.error("Configured Gemini model '%s' not found in available models: %s", clean_name, available_models)
                    raise SentinelRAGError(
                        f"Configuration error: Configured Gemini model '{clean_name}' is unavailable for your API key. "
                        f"Available models: {', '.join(available_models[:10])}"
                    )
                target = next((m for m in models if m.get("name", "").removeprefix("models/") == clean_name), None)
                if target and expected_method:
                    methods = target.get("supportedGenerationMethods", [])
                    if expected_method not in methods:
                        raise SentinelRAGError(
                            f"Configuration error: Gemini model '{clean_name}' does not support method '{expected_method}'. "
                            f"Supported methods: {', '.join(methods)}"
                        )
                logger.info("Gemini model validation passed: '%s' is available and supports '%s'.", clean_name, expected_method)
        except SentinelRAGError:
            raise
        except Exception as exc:
            logger.warning("Gemini model validation encountered error (%s). Proceeding.", exc)

    def _resolve_voyage_key_and_source(self) -> tuple[Optional[str], str]:
        if self.settings.voyage_api_key and self.settings.voyage_api_key.strip():
            return self.settings.voyage_api_key.strip(), "AISettings"
        
        env_voyage = os.getenv("VOYAGE_API_KEY") or os.getenv("AI_VOYAGE_API_KEY")
        if env_voyage and env_voyage.strip():
            source = "VOYAGE_API_KEY" if os.getenv("VOYAGE_API_KEY") else "AI_VOYAGE_API_KEY"
            return env_voyage.strip(), f"env:{source}"
        
        return None, "missing"

    def create_embedding_provider(self) -> BaseEmbeddingProvider:
        provider_name = self.settings.provider.lower().strip()
        if provider_name == "gemini":
            key, source = self._resolve_gemini_key_and_source()
            if not key:
                raise SentinelRAGError(
                    "Configuration error: AI_PROVIDER is set to 'gemini' but GEMINI_API_KEY is missing or empty."
                )
            self._validate_gemini_model(key, self.settings.gemini_embedding_model, "embedContent")
            logger.info("Creating Gemini Embedding Provider (%s, key source: %s, key length: %d)", self.settings.gemini_embedding_model, source, len(key))
            
            gemini_provider = GeminiEmbeddingProvider(
                api_key=key,
                model=self.settings.gemini_embedding_model,
            )

            voyage_key, voyage_source = self._resolve_voyage_key_and_source()
            if self.settings.embedding_failover_enabled and voyage_key:
                logger.info("Enabling Embedding Failover: Primary=Gemini, Secondary=Voyage (%s, key source: %s)", self.settings.voyage_embedding_model, voyage_source)
                voyage_provider = VoyageEmbeddingProvider(
                    api_key=voyage_key,
                    model=self.settings.voyage_embedding_model,
                )
                return EmbeddingRouterProvider(
                    primary=gemini_provider,
                    secondary=voyage_provider,
                    primary_name="gemini",
                    secondary_name="voyage",
                )
            
            return EmbeddingRouterProvider(
                primary=gemini_provider,
                secondary=None,
                primary_name="gemini",
                secondary_name="voyage",
            )
        return DeterministicEmbeddingProvider()

    def create_llm_provider(self) -> BaseLLMProvider:
        provider_name = self.settings.provider.lower().strip()
        if provider_name == "gemini":
            key, source = self._resolve_gemini_key_and_source()
            if not key:
                raise SentinelRAGError(
                    "Configuration error: AI_PROVIDER is set to 'gemini' but GEMINI_API_KEY is missing or empty."
                )
            self._validate_gemini_model(key, self.settings.gemini_model, "generateContent")
            logger.info("Creating Gemini LLM Provider (%s, key source: %s, key length: %d)", self.settings.gemini_model, source, len(key))
            return GeminiLLMProvider(
                api_key=key,
                model=self.settings.gemini_model,
            )
        return DeterministicLLMProvider()

    def create_reranker_provider(self) -> BaseRerankerProvider:
        return DeterministicRerankerProvider()
