from app.core.settings.ai_provider import AISettings


def test_constructor_credentials_are_preserved():
    settings = AISettings(_env_file=None, gemini_api_key="local-test-key", voyage_api_key="local-voyage-key")
    assert settings.gemini_api_key == "local-test-key"
    assert settings.voyage_api_key == "local-voyage-key"


def test_environment_aliases_are_supported(monkeypatch):
    monkeypatch.setenv("AI_GEMINI_API_KEY", "prefixed-test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-test-key")
    settings = AISettings(_env_file=None)
    assert settings.gemini_api_key == "prefixed-test-key"


def test_explicit_constructor_value_overrides_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "environment-test-key")
    settings = AISettings(_env_file=None, gemini_api_key="constructor-test-key")
    assert settings.gemini_api_key == "constructor-test-key"
