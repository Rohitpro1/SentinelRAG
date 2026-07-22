"""
Unit 2.5 -- DeterministicReranker.

A valid, first-class implementation of BaseReranker (same "deterministic,
not fake" treatment as DeterministicEmbedder) -- useful wherever a real
cross-encoder is unavailable or undesired, not merely for tests.

Reranking rule: combines the original similarity_score with a small,
deterministic per-(query, chunk_id) perturbation derived from SHA-256, so
that reranking measurably changes ordering (proving the rerank stage is
actually wired into RetrieverAgent) without requiring a real cross-encoder
model. Not semantically meaningful.

Sibling implementations (to be added later, same BaseReranker interface):
CrossEncoderReranker (e.g. a sentence-transformers cross-encoder model).
"""
from __future__ import annotations

import hashlib

from app.schemas.reranking import RerankerCapabilities, RerankResult
from app.schemas.retrieval import RetrievedChunk
from app.schemas.retrieval_domain import RankedChunk
from app.services.reranking.base import BaseReranker
from app.services.reranking.result_builder import build_rerank_result


class DeterministicReranker(BaseReranker):
    async def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RankedChunk]:
        scored = [(self._score(query, rc), rc) for rc in candidates]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:top_n]
        return [
            RankedChunk(retrieved_chunk=rc, rerank_score=score, rank=idx)
            for idx, (score, rc) in enumerate(top)
        ]

    async def rerank_with_result(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> RerankResult:
        """
        Unit 2.12 -- additive capability, not part of BaseReranker.
        Deterministic-first: proves RerankResult's shape here before any
        real provider exists, same principle as Unit 2.11's
        embed_query_with_result on DeterministicEmbedder.
        """
        return await build_rerank_result(
            lambda: self.rerank(query, candidates, top_n),
            provider="deterministic",
            model_name="deterministic-sha256-perturbation",
            model_version="v1",
        )

    def capabilities(self) -> RerankerCapabilities:
        """
        No real provider limits apply to a local, dependency-free
        computation -- these are nominal/unconstrained values, not
        provider-documented limits (contrast with CrossEncoderReranker's
        capabilities(), which reflects RerankingSettings). Documented
        explicitly rather than left to look like a real constraint.
        """
        return RerankerCapabilities(
            supports_batching=True, max_batch_size=10_000, max_input_tokens=1_000_000, model_dimensions=None
        )

    @staticmethod
    def _score(query: str, retrieved_chunk: RetrievedChunk) -> float:
        digest = hashlib.sha256(f"{query}:{retrieved_chunk.chunk.chunk_id}".encode("utf-8")).digest()
        # Small deterministic perturbation in [-0.05, 0.05], layered on top
        # of the original similarity score so rerank order can plausibly
        # differ from similarity order without being unbounded/nonsensical.
        perturbation = (int.from_bytes(digest[:4], "big") / (2**32) - 0.5) * 0.1
        return max(0.0, min(1.0, retrieved_chunk.similarity_score + perturbation))
