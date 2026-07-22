from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Embedding domain settings, including real-provider connection config (Unit 2.11)."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING__", env_file=".env", extra="ignore")

    model_name: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64

    # Real provider (OpenAIEmbedder, Unit 2.11) connection config.
    # base_url defaults to OpenAI's API but is deliberately overridable --
    # any OpenAI-compatible embeddings endpoint (Ollama, vLLM, Azure OpenAI
    # with the right path) works without a new client class, per the
    # trade-off documented in openai_embedder.py.
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    request_timeout_ms: int = 5000
    unavailable_after_consecutive_failures: int = 3
