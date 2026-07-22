"""
Unit 2.10 -- REAL Qdrant integration test.

Per instruction 6: isolated and optional. Marked @pytest.mark.integration,
excluded from the default test run (pytest.ini: addopts = -m "not
integration"). Run explicitly with:

    PYTHONPATH=. pytest tests/test_qdrant_integration.py -m integration -v

Requires a real Qdrant instance reachable at StorageSettings().qdrant_url
(default http://localhost:6333, e.g. `docker run -p 6333:6333 qdrant/qdrant`).
If unreachable, every test in this module is skipped (not failed) via the
module-level fixture below -- CI without Qdrant available stays green,
and this file only ever exercises real behavior when real infrastructure
is actually present.
"""
import uuid

import pytest

from app.core.settings.storage import StorageSettings
from app.infrastructure.qdrant_client_factory import close_qdrant_client, create_qdrant_client
from app.repositories.qdrant.vector_repository import QdrantVectorRepository
from app.schemas.retrieval import Chunk

pytestmark = pytest.mark.integration


@pytest.fixture
async def live_repo():
    settings = StorageSettings(qdrant_collection=f"sentinelrag_test_{uuid.uuid4().hex[:8]}", qdrant_vector_size=8)
    client = create_qdrant_client(settings)
    try:
        await client.get_collections()
    except Exception as exc:  # noqa: BLE001
        await close_qdrant_client(client)
        pytest.skip(f"Live Qdrant not reachable at {settings.qdrant_url}: {exc}")

    repo = QdrantVectorRepository(client, settings)
    await repo.ensure_collection()
    yield repo

    # Deterministic teardown: drop the test-only collection, then close
    # the connection -- both explicit, per instruction 4.
    await client.delete_collection(settings.qdrant_collection)
    await close_qdrant_client(client)


def make_chunk(chunk_id, document_id="doc-1", text="refund policy detail"):
    return Chunk(chunk_id=chunk_id, document_id=document_id, text=text, token_count=10, source_reliability_score=0.9)


@pytest.mark.asyncio
async def test_upsert_then_search_round_trip(live_repo):
    chunks = [make_chunk("c1"), make_chunk("c2")]
    embeddings = [[1.0] + [0.0] * 7, [0.0] * 7 + [1.0]]
    await live_repo.upsert(chunks, embeddings)

    results = await live_repo.search(query_embedding=[1.0] + [0.0] * 7, top_k=5)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c1"  # most similar to the query vector


@pytest.mark.asyncio
async def test_search_respects_document_filter(live_repo):
    await live_repo.upsert(
        [make_chunk("c1", document_id="doc-A"), make_chunk("c2", document_id="doc-B")],
        [[1.0] + [0.0] * 7, [1.0] + [0.0] * 7],
    )
    results = await live_repo.search([1.0] + [0.0] * 7, top_k=5, document_filter={"document_id": "doc-A"})
    assert len(results) == 1
    assert results[0].chunk.document_id == "doc-A"


@pytest.mark.asyncio
async def test_delete_removes_only_target_document(live_repo):
    await live_repo.upsert(
        [make_chunk("c1", document_id="doc-A"), make_chunk("c2", document_id="doc-B")],
        [[1.0] + [0.0] * 7, [1.0] + [0.0] * 7],
    )
    await live_repo.delete("doc-A")
    results = await live_repo.search([1.0] + [0.0] * 7, top_k=5)
    assert len(results) == 1
    assert results[0].chunk.document_id == "doc-B"


@pytest.mark.asyncio
async def test_upsert_is_idempotent_for_same_chunk_id(live_repo):
    chunk = make_chunk("c1", text="version 1")
    await live_repo.upsert([chunk], [[1.0] + [0.0] * 7])

    updated_chunk = make_chunk("c1", text="version 2")
    await live_repo.upsert([updated_chunk], [[1.0] + [0.0] * 7])

    results = await live_repo.search([1.0] + [0.0] * 7, top_k=5)
    assert len(results) == 1  # same chunk_id -> same point ID -> overwrite, not duplicate
    assert results[0].chunk.text == "version 2"
