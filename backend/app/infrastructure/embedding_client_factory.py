"""
Unit 2.11 -- HTTP client factory for the real embedding provider.

Same DI discipline as Unit 2.10's Qdrant client factory: this is the
ONLY place permitted to construct the httpx.AsyncClient used by
OpenAIEmbedder. The embedder receives an already-constructed client via
its constructor and never builds its own.
"""
from __future__ import annotations

import httpx

from app.core.settings.embedding import EmbeddingSettings


def create_embedding_http_client(settings: EmbeddingSettings) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
    return httpx.AsyncClient(
        base_url=settings.api_base_url,
        timeout=settings.request_timeout_ms / 1000,
        headers=headers,
    )


async def close_embedding_http_client(client: httpx.AsyncClient) -> None:
    await client.aclose()
