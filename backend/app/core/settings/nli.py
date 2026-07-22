"""
Unit 2.13 -- NLISettings.

Same pattern as EmbeddingSettings/RerankingSettings. The real NLI
provider (LLMBasedNLIVerifier) uses an OpenAI-compatible chat-completions
endpoint as its transport -- see that class's docstring for why no
dedicated single-vendor NLI REST API convention exists to mirror instead.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class NLISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NLI__", env_file=".env", extra="ignore")

    model_name: str = "gpt-4o-mini"
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    request_timeout_ms: int = 5000
    unavailable_after_consecutive_failures: int = 3
