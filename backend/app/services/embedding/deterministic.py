"""
Unit 2.4 (renamed per Unit 2.5 review) -- DeterministicEmbedder.

Treated as a valid, first-class implementation of BaseEmbedder -- not a
test-only stub. Uses SHA-256 of the input text to produce a fully
deterministic, unit-normalized pseudo-embedding: same text always
produces the same vector, different texts produce different vectors,
with zero model-loading cost. Useful for local dev, CI, and any
environment where a real semantic embedding model is unavailable or
undesired -- not merely "for tests."

Sibling implementations (to be added later, same BaseEmbedder interface):
SentenceTransformerEmbedder, OpenAIEmbedder, OllamaEmbedder, AzureOpenAIEmbedder.

Deliberately NOT semantically meaningful -- do not use where semantic
similarity quality matters.
"""
from __future__ import annotations

import hashlib
import math

from app.schemas.embedding import EmbedderHealth, EmbedderHealthState, EmbeddingResult
from app.services.embedding.base import BaseEmbedder
from app.services.embedding.result_builder import build_embedding_result


class DeterministicEmbedder(BaseEmbedder):
    def __init__(self, dimensions: int = 16):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_query(self, text: str) -> list[float]:
        return self._deterministic_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._deterministic_vector(t) for t in texts]

    async def embed_query_with_result(self, text: str) -> EmbeddingResult:
        """
        Unit 2.11 -- additive capability, not part of BaseEmbedder.
        Deterministic-first: this is where EmbeddingResult's shape gets
        exercised in tests BEFORE any real provider exists, per the
        deterministic-first development principle.
        """
        return await build_embedding_result(
            lambda: self.embed_query(text),
            provider="deterministic",
            model_name="deterministic-sha256",
            model_version="v1",
        )

    def health(self) -> EmbedderHealth:
        """
        The deterministic embedder has no external dependency and cannot
        fail -- always READY. Real providers (Unit 2.11's OpenAIEmbedder)
        override this with actual failure-tracking.
        """
        return EmbedderHealth(state=EmbedderHealthState.READY)

    def _deterministic_vector(self, text: str) -> list[float]:
        # Expand the 32-byte SHA-256 digest to `dimensions` floats by
        # re-hashing with a counter suffix whenever more bytes are needed
        # than a single digest provides.
        raw_bytes = bytearray()
        counter = 0
        while len(raw_bytes) < self._dimensions * 4:
            digest = hashlib.sha256(f"{text}:{counter}".encode("utf-8")).digest()
            raw_bytes.extend(digest)
            counter += 1

        values = [
            int.from_bytes(raw_bytes[i : i + 4], "big") / (2**32) for i in range(0, self._dimensions * 4, 4)
        ]
        # Normalize to unit length so cosine-similarity-based repositories
        # (Unit 2.2's InMemoryVectorRepository) behave sensibly.
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:
            return values
        return [v / norm for v in values]
