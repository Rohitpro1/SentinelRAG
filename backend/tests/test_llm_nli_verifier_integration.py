"""
Unit 2.13 -- REAL network integration test for LLMBasedNLIVerifier.

Isolated and optional, identical pattern to Units 2.10/2.11/2.12's live
tests: marked @pytest.mark.integration, excluded from the default run,
skips cleanly without a reachable, authenticated endpoint.

Run explicitly, e.g. against OpenAI:

    NLI__API_BASE_URL=https://api.openai.com/v1 \\
    NLI__API_KEY=sk-... \\
    NLI__MODEL_NAME=gpt-4o-mini \\
    PYTHONPATH=. pytest tests/test_llm_nli_verifier_integration.py -m integration -v

This sandbox has no network route to any LLM provider, so this file has
only ever been run in its skip path here.
"""
import pytest

from app.core.settings.nli import NLISettings
from app.infrastructure.nli_client_factory import close_nli_http_client, create_nli_http_client
from app.schemas.retrieval import NLIRelation
from app.services.verification.llm_nli_verifier import LLMBasedNLIVerifier

pytestmark = pytest.mark.integration


@pytest.fixture
async def live_verifier():
    settings = NLISettings()
    if not settings.api_key and "localhost" not in settings.api_base_url and "127.0.0.1" not in settings.api_base_url:
        pytest.skip("No NLI__API_KEY configured and api_base_url is not a local endpoint -- skipping live test.")

    client = create_nli_http_client(settings)
    verifier = LLMBasedNLIVerifier(client, settings)

    try:
        await verifier.verify_pair("connectivity check A", "connectivity check B")
    except Exception as exc:  # noqa: BLE001
        await close_nli_http_client(client)
        pytest.skip(f"Configured NLI endpoint not reachable: {exc}")

    yield verifier
    await close_nli_http_client(client)


@pytest.mark.asyncio
async def test_detects_obvious_contradiction(live_verifier):
    relation, confidence = await live_verifier.verify_pair(
        "Refunds are allowed within 30 days of purchase.",
        "Refunds are never allowed under any circumstances.",
    )
    assert relation == NLIRelation.CONTRADICTION
    assert confidence > 0.5


@pytest.mark.asyncio
async def test_detects_unrelated_texts_as_neutral(live_verifier):
    relation, _ = await live_verifier.verify_pair(
        "The refund policy allows returns within 30 days.",
        "The office building has five floors and a parking garage.",
    )
    assert relation == NLIRelation.NEUTRAL
