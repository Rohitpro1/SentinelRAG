from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    provider: str = "deterministic"
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("AI_GEMINI_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash-lite",
        validation_alias=AliasChoices("AI_GEMINI_MODEL", "GEMINI_MODEL"),
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        validation_alias=AliasChoices("AI_GEMINI_EMBEDDING_MODEL", "GEMINI_EMBEDDING_MODEL"),
    )

    voyage_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("AI_VOYAGE_API_KEY", "VOYAGE_API_KEY"),
    )
    voyage_embedding_model: str = Field(
        default="voyage-3-large",
        validation_alias=AliasChoices("AI_VOYAGE_EMBEDDING_MODEL", "VOYAGE_EMBEDDING_MODEL"),
    )

    embedding_failover_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("AI_EMBEDDING_FAILOVER_ENABLED", "EMBEDDING_FAILOVER_ENABLED"),
    )
    embedding_primary_provider: str = Field(
        default="gemini",
        validation_alias=AliasChoices("AI_EMBEDDING_PRIMARY_PROVIDER", "EMBEDDING_PRIMARY_PROVIDER"),
    )
    embedding_secondary_provider: str = Field(
        default="voyage",
        validation_alias=AliasChoices("AI_EMBEDDING_SECONDARY_PROVIDER", "EMBEDDING_SECONDARY_PROVIDER"),
    )
