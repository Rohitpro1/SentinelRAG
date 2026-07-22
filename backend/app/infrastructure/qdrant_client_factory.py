"""
Unit 2.10 -- Qdrant client factory.

Per instruction 7 (complete DI, no client instantiation inside business
logic): this is the ONLY place in the codebase permitted to construct a
qdrant_client.AsyncQdrantClient. QdrantVectorRepository receives an
already-constructed client via its constructor; it never calls this
factory itself. Application bootstrap (main.py, not yet built) is
responsible for calling this once at startup and passing the client to
whatever DI container/wiring constructs QdrantVectorRepository, then
calling close_qdrant_client() at shutdown.

This module is infrastructure, not domain -- it imports qdrant_client
directly (the domain layer, VectorRepository and everything above it,
never does). This is the one intentional, explicit dependency-direction
boundary: infrastructure depends on the domain interface it implements
(via QdrantVectorRepository implementing VectorRepository), and this
factory depends on the concrete qdrant_client SDK -- the domain never
imports either.
"""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from app.core.settings.storage import StorageSettings


def create_qdrant_client(settings: StorageSettings) -> AsyncQdrantClient:
    """
    Constructs a new AsyncQdrantClient. Supports both local and cloud Qdrant (via api_key).
    """
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=int(settings.qdrant_timeout_ms / 1000),
    )


async def close_qdrant_client(client: AsyncQdrantClient) -> None:
    """
    Explicit resource cleanup, per instruction 4 (deterministic lifecycle).
    Application bootstrap must call this at shutdown (e.g. a FastAPI
    lifespan handler, once one exists) -- there is no finalizer/GC-based
    cleanup relied on here, since that would make connection teardown
    non-deterministic.
    """
    await client.close()
