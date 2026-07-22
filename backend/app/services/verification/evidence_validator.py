"""
Unit 2.9 -- EvidenceValidator.

Owns: structural validation of retrieved evidence BEFORE contradiction
detection or coverage analysis run on it. Current checks are structural
only (non-empty text, positive token_count) -- true semantic claim-level
validation (does this evidence actually support a specific claim in a
draft answer) requires a draft answer from ReasoningAgent/ResponseGenerator,
which do not exist yet (Milestone 3 gap, stated explicitly, not hidden).
"""
from __future__ import annotations

from app.schemas.retrieval_domain import RankedChunk


class EvidenceValidator:
    async def validate(self, ranked_chunks: list[RankedChunk]) -> tuple[list[RankedChunk], list[str]]:
        valid: list[RankedChunk] = []
        unsupported_claims: list[str] = []

        for ranked_chunk in ranked_chunks:
            chunk = ranked_chunk.retrieved_chunk.chunk
            if chunk.text.strip() and chunk.token_count > 0:
                valid.append(ranked_chunk)
            else:
                unsupported_claims.append(chunk.chunk_id)

        return valid, unsupported_claims
