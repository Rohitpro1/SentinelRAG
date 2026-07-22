"""
Unit 2.9 -- CoverageAnalyzer.

Owns: two numeric signals over validated evidence --
  evidence_coverage: fraction of originally retrieved evidence that
    survived EvidenceValidator (i.e. how much of what was retrieved
    turned out to be structurally usable).
  reranker_confidence: average rerank_score across valid evidence with a
    score; 0.0 if every score is None (RerankingService's degraded mode,
    Unit 2.6) -- surfacing degraded reranking as a visibly low confidence
    number rather than silently treating it as neutral.
"""
from __future__ import annotations

from app.schemas.retrieval_domain import RankedChunk


class CoverageAnalyzer:
    async def analyze(
        self, valid_evidence: list[RankedChunk], total_retrieved: int
    ) -> tuple[float, float]:
        evidence_coverage = (len(valid_evidence) / total_retrieved) if total_retrieved > 0 else 0.0
        evidence_coverage = max(0.0, min(1.0, evidence_coverage))

        scored = [rc.rerank_score for rc in valid_evidence if rc.rerank_score is not None]
        reranker_confidence = (sum(scored) / len(scored)) if scored else 0.0

        return evidence_coverage, reranker_confidence
