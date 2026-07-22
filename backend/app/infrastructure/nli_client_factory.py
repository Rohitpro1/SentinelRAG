"""
Unit 2.13 -- HTTP client factory for the real NLI provider.

Same DI discipline as Units 2.10/2.11/2.12's factories.
"""
from __future__ import annotations

import httpx

from app.core.settings.nli import NLISettings


def create_nli_http_client(settings: NLISettings) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
    return httpx.AsyncClient(
        base_url=settings.api_base_url,
        timeout=settings.request_timeout_ms / 1000,
        headers=headers,
    )


async def close_nli_http_client(client: httpx.AsyncClient) -> None:
    await client.aclose()
