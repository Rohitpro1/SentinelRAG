from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecuritySettings(BaseSettings):
    """Placeholder domain for Milestone 3 (Auth0, prompt-injection detection, PII redaction)."""

    model_config = SettingsConfigDict(env_prefix="SECURITY__", env_file=".env", extra="ignore")

    auth0_domain: str = ""
    auth0_audience: str = ""
    jwt_algorithm: str = "RS256"
    prompt_injection_detection_enabled: bool = True
    pii_redaction_enabled: bool = True
