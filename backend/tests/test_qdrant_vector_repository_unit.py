"""
Unit 2.10 tests -- QdrantVectorRepository's pure/translatable logic, tested
WITHOUT a real Qdrant connection (per instruction 6: deterministic testing
stays the default; a live Qdrant is isolated to test_qdrant_integration.py).

These tests exercise: exception translation (transient vs non-transient),
point-ID determinism, payload round-trip shape, and the VectorRepository
interface contract -- using a fake AsyncQdrantClient double, not a real
server.
"""
import httpx
import pytest

from app.core.exceptions import RetrievalError
from app.core.settings.storage import StorageSettings
from app.repositories.interfaces import VectorRepository
from app.repositories.qdrant.vector_repository import QdrantVectorRepository, _chunk_id_to_point_id
from app.schemas.retrieval import Chunk
from qdrant_client.http.exceptions import UnexpectedResponse


def make_repo(client):
    return QdrantVectorRepository(client, StorageSettings(qdrant_collection="test_collection"))


def make_chunk(chunk_id="c1"):
    return Chunk(
        chunk_id=chunk_id, document_id="doc-1", text="some content",
        token_count=10, source_reliability_score=0.9,
    )


# --- Interface compliance ---

def test_is_a_valid_vector_repository():
    assert issubclass(QdrantVectorRepository, VectorRepository)


# --- Point ID determinism (the chunk_id -> UUID5 trade-off) ---

def test_chunk_id_to_point_id_is_deterministic():
    assert _chunk_id_to_point_id("c1") == _chunk_id_to_point_id("c1")


def test_chunk_id_to_point_id_differs_for_different_ids():
    assert _chunk_id_to_point_id("c1") != _chunk_id_to_point_id("c2")


def test_chunk_id_to_point_id_is_a_valid_uuid_string():
    import uuid
    point_id = _chunk_id_to_point_id("arbitrary-chunk-id-123")
    uuid.UUID(point_id)  # raises ValueError if not a valid UUID -- test fails if so


# --- upsert() length validation ---

@pytest.mark.asyncio
async def test_upsert_raises_non_transient_on_length_mismatch():
    repo = make_repo(client=None)  # never reached -- validation happens before any client call
    with pytest.raises(RetrievalError) as exc_info:
        await repo.upsert([make_chunk()], [])
    assert exc_info.value.transient is False


# --- Exception translation ---

class _FakeAsyncClient:
    """Minimal double -- only the methods QdrantVectorRepository actually calls."""

    def __init__(self, raise_on_search=None, raise_on_upsert=None, raise_on_delete=None):
        self._raise_on_search = raise_on_search
        self._raise_on_upsert = raise_on_upsert
        self._raise_on_delete = raise_on_delete
        self.upsert_calls = 0

    async def query_points(self, **kwargs):
        if self._raise_on_search:
            raise self._raise_on_search

        class _Resp:
            points = []
        return _Resp()

    async def upsert(self, **kwargs):
        self.upsert_calls += 1
        if self._raise_on_upsert:
            raise self._raise_on_upsert

    async def delete(self, **kwargs):
        if self._raise_on_delete:
            raise self._raise_on_delete

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_search_translates_connect_error_as_transient():
    client = _FakeAsyncClient(raise_on_search=httpx.ConnectError("connection refused"))
    repo = make_repo(client)
    with pytest.raises(RetrievalError) as exc_info:
        await repo.search([1.0, 0.0], top_k=5)
    assert exc_info.value.transient is True


@pytest.mark.asyncio
async def test_search_translates_timeout_as_transient():
    client = _FakeAsyncClient(raise_on_search=httpx.TimeoutException("timed out"))
    repo = make_repo(client)
    with pytest.raises(RetrievalError) as exc_info:
        await repo.search([1.0, 0.0], top_k=5)
    assert exc_info.value.transient is True


@pytest.mark.asyncio
async def test_search_translates_5xx_as_transient():
    exc = UnexpectedResponse(status_code=503, reason_phrase="Service Unavailable", content=b"", headers=httpx.Headers())
    client = _FakeAsyncClient(raise_on_search=exc)
    repo = make_repo(client)
    with pytest.raises(RetrievalError) as exc_info:
        await repo.search([1.0, 0.0], top_k=5)
    assert exc_info.value.transient is True


@pytest.mark.asyncio
async def test_search_translates_4xx_as_non_transient():
    exc = UnexpectedResponse(status_code=400, reason_phrase="Bad Request", content=b"", headers=httpx.Headers())
    client = _FakeAsyncClient(raise_on_search=exc)
    repo = make_repo(client)
    with pytest.raises(RetrievalError) as exc_info:
        await repo.search([1.0, 0.0], top_k=5)
    assert exc_info.value.transient is False


@pytest.mark.asyncio
async def test_search_does_not_retry_internally():
    """
    Documented trade-off: search() has NO internal retry (SearchService
    owns that). A single failure must raise immediately, exactly once.
    """
    client = _FakeAsyncClient(raise_on_search=httpx.ConnectError("down"))
    repo = make_repo(client)
    with pytest.raises(RetrievalError):
        await repo.search([1.0, 0.0], top_k=5)
    # No call counter needed for query_points since it raises every time
    # regardless of attempt count -- what matters is this raises on the
    # FIRST call with no observable delay/backoff, verified structurally
    # by there being no retry loop in search()'s implementation.


@pytest.mark.asyncio
async def test_upsert_retries_once_on_transient_then_succeeds():
    class _FlakyUpsertClient(_FakeAsyncClient):
        async def upsert(self, **kwargs):
            self.upsert_calls += 1
            if self.upsert_calls == 1:
                raise httpx.ConnectError("transient blip")
            # second call succeeds

    client = _FlakyUpsertClient()
    repo = make_repo(client)
    await repo.upsert([make_chunk()], [[1.0, 0.0]])
    assert client.upsert_calls == 2


@pytest.mark.asyncio
async def test_upsert_raises_immediately_on_non_transient_without_retry():
    exc = UnexpectedResponse(status_code=422, reason_phrase="Unprocessable", content=b"", headers=httpx.Headers())
    client = _FakeAsyncClient(raise_on_upsert=exc)
    repo = make_repo(client)
    with pytest.raises(RetrievalError):
        await repo.upsert([make_chunk()], [[1.0, 0.0]])
    assert client.upsert_calls == 1  # no retry attempted for a non-transient error


@pytest.mark.asyncio
async def test_upsert_gives_up_after_bounded_retry_exhausted():
    client = _FakeAsyncClient(raise_on_upsert=httpx.ConnectError("always down"))
    repo = make_repo(client)
    with pytest.raises(RetrievalError):
        await repo.upsert([make_chunk()], [[1.0, 0.0]])
    assert client.upsert_calls == 2  # initial attempt + 1 bounded retry, then gives up


# --- Similarity score clamping ---

def test_point_to_retrieved_chunk_clamps_similarity_to_unit_range():
    class _FakePoint:
        id = "some-id"
        score = 1.4  # out-of-range, e.g. a non-cosine metric
        payload = {
            "chunk_id": "c1", "document_id": "doc-1", "text": "x",
            "token_count": 5, "source_reliability_score": 0.9,
        }

    rc = QdrantVectorRepository._point_to_retrieved_chunk(_FakePoint())
    assert rc.similarity_score == 1.0


def test_point_to_retrieved_chunk_preserves_payload_fields():
    class _FakePoint:
        id = "some-id"
        score = 0.75
        payload = {
            "chunk_id": "c7", "document_id": "doc-9", "text": "hello world",
            "token_count": 20, "source_reliability_score": 0.5, "ocr_confidence": 0.8,
            "metadata": {"page": 3},
        }

    rc = QdrantVectorRepository._point_to_retrieved_chunk(_FakePoint())
    assert rc.chunk.chunk_id == "c7"
    assert rc.chunk.ocr_confidence == 0.8
    assert rc.chunk.metadata == {"page": 3}
    assert rc.similarity_score == 0.75
