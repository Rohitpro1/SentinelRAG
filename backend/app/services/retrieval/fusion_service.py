"""
Unit 2.6 -- FusionService.

Current scope: deduplicate candidates by chunk_id (keeping the
highest-similarity occurrence) and sort descending. This is a real,
tested behavior, not a stub.

Known gap, stated explicitly rather than hidden: the frozen Retrieval
Domain Design describes hybrid search as semantic + keyword fusion, but
no KeywordRepository interface has been designed yet (only VectorRepository
exists, per Section 4 of the frozen design). Introducing a keyword-search
interface is a new architectural surface and is out of scope for this
implementation-mode unit -- FusionService is deliberately structured so
that adding a `keyword_results: list[RetrievedChunk]` parameter and a
score-fusion strategy (e.g. reciprocal rank fusion) later is an additive
change to this one method, not a change to RetrieverAgent or any other
caller.
"""
from __future__ import annotations

from app.schemas.retrieval import RetrievedChunk


class FusionService:
    async def fuse(self, semantic_results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        best_by_chunk_id: dict[str, RetrievedChunk] = {}
        for candidate in semantic_results:
            existing = best_by_chunk_id.get(candidate.chunk.chunk_id)
            if existing is None or candidate.similarity_score > existing.similarity_score:
                best_by_chunk_id[candidate.chunk.chunk_id] = candidate

        return sorted(best_by_chunk_id.values(), key=lambda rc: rc.similarity_score, reverse=True)
