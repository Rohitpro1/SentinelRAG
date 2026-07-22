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
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("AI_GEMINI_MODEL", "GEMINI_MODEL"),
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias=AliasChoices("AI_GEMINI_EMBEDDING_MODEL", "GEMINI_EMBEDDING_MODEL"),
    )
