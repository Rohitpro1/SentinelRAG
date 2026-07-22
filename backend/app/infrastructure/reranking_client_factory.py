"""
Unit 2.12 -- HTTP client factory for the real cross-encoder reranker.

Same DI discipline as Units 2.10/2.11's factories: the only place
permitted to construct the httpx.AsyncClient used by CrossEncoderReranker.
"""
from __future__ import annotations

import httpx

from app.core.settings.reranking import RerankingSettings


def create_reranking_http_client(settings: RerankingSettings) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
    return httpx.AsyncClient(
        base_url=settings.api_base_url,
        timeout=settings.request_timeout_ms / 1000,
        headers=headers,
    )


async def close_reranking_http_client(client: httpx.AsyncClient) -> None:
    await client.aclose()
