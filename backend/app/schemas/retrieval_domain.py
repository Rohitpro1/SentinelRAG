"""
Unit 2.1 — Retrieval Domain schemas.

Builds on Milestone 1's app/schemas/retrieval.py (Chunk, RetrievedChunk,
PairwiseNLIResult, VerificationReport) WITHOUT modifying it. Per the frozen
Retrieval Domain design (Section 3), this module owns the new contracts
that flow through RetrieverAgent -> VerificationAgent, and provides the
single allowed adapter into Milestone 1's frozen VerificationReport.
"""
from __future__ import annotations

from typing import Optional, Self

from pydantic import BaseModel, Field

from app.schemas.retrieval import PairwiseNLIResult, RetrievedChunk, VerificationReport


class SearchRequest(BaseModel):
    """Input to RetrieverAgent.search() (Unit 2.6+)."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, description="Candidates fetched before reranking")
    rerank_top_n: int = Field(default=5, ge=1, description="Candidates kept after reranking")
    document_filter: Optional[dict] = None
    retry_count: int = Field(default=0, ge=0)
    request_id: Optional[str] = None
    trace_id: Optional[str] = None


class RankedChunk(BaseModel):
    """A RetrievedChunk after cross-encoder reranking (Unit 2.7)."""

    retrieved_chunk: RetrievedChunk
    rerank_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="None if reranker degraded (see failure modes)"
    )
    rank: int = Field(ge=0, description="Final position after reranking, 0-indexed")


class SearchResponse(BaseModel):
    """Output of RetrieverAgent.search()."""

    request: SearchRequest
    ranked_chunks: list[RankedChunk] = Field(default_factory=list)
    cache_hit: bool = False
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)


class VerificationInput(BaseModel):
    """Input to VerificationAgent (Unit 2.9)."""

    query: str
    ranked_chunks: list[RankedChunk]
    retry_count: int = Field(default=0, ge=0)


class VerificationOutput(BaseModel):
    """
    Output of VerificationAgent. Deliberately a superset of what
    Milestone 1's VerificationReport needs, with an explicit adapter
    method rather than being VerificationReport itself — this is the
    one allowed seam into frozen Milestone 1 code (Retrieval Domain
    Design, Section 3).
    """

    query: str
    retrieved_chunks: list[RetrievedChunk]
    nli_results: list[PairwiseNLIResult] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)

    def to_verification_report(self) -> VerificationReport:
        """
        Adapter into Milestone 1's frozen DecisionEngine input contract.
        Field names/shapes happen to match 1:1 today, but this method
        exists so that if VerificationOutput ever needs to diverge
        (e.g. carrying extra Retrieval-Domain-only fields), the frozen
        VerificationReport contract is unaffected — only this adapter
        changes.
        """
        return VerificationReport(
            query=self.query,
            retrieved_chunks=self.retrieved_chunks,
            nli_results=self.nli_results,
            retry_count=self.retry_count,
        )

    @classmethod
    def from_ranked_chunks(
        cls,
        query: str,
        ranked_chunks: list[RankedChunk],
        nli_results: Optional[list[PairwiseNLIResult]] = None,
        retry_count: int = 0,
    ) -> Self:
        """
        Convenience constructor used by VerificationAgent (Unit 2.9) to
        flatten RankedChunk -> RetrievedChunk, since VerificationReport
        (and everything downstream in the frozen Decision Engine) only
        knows about RetrievedChunk, not the Retrieval Domain's RankedChunk.
        """
        return cls(
            query=query,
            retrieved_chunks=[rc.retrieved_chunk for rc in ranked_chunks],
            nli_results=nli_results or [],
            retry_count=retry_count,
        )


class VerifiedEvidence(VerificationOutput):
    """
    Unit 2.9 -- the BUSINESS output of VerificationAgent. Consumed by
    DecisionEngine (via .to_verification_report(), inherited unchanged),
    and future ReasoningAgent / ResponseGenerator (Milestone 3), which
    need the same verified, de-duplicated evidence set to ground an answer.

    Deliberately a subclass of VerificationOutput (Unit 2.1) rather than a
    parallel, duplicated schema: identical shape, identical adapter
    behavior (to_verification_report(), from_ranked_chunks() both inherited
    and work correctly via `cls(...)` construction) -- this is a semantic
    rename at the type level, made explicit per the Unit 2.9 review's
    business/observability split, without re-deriving or duplicating
    Unit 2.1's already-tested adapter logic.
    """


class VerificationDiagnostics(BaseModel):
    """
    Unit 2.9 -- the OBSERVABILITY output of VerificationAgent. Consumed by
    telemetry, the dashboard, evaluation, and future analytics -- NEVER by
    DecisionEngine, which only ever sees VerifiedEvidence. Keeping these
    as two separate return values (not one merged object) is what lets the
    business path stay exactly as small and stable as VerifiedEvidence's
    inherited adapter, while this object is free to grow new debugging/
    analytics fields over time without touching the Decision Engine seam.
    """

    query: str
    nli_score: float = Field(
        ge=0.0, le=1.0,
        description="Aggregate evidence-consistency score: 1.0 - max_contradiction_confidence across all NLI pairs.",
    )
    contradiction_detected: bool
    evidence_coverage: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of retrieved evidence that passed EvidenceValidator's structural checks.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "chunk_ids flagged by EvidenceValidator as structurally unusable evidence. "
            "NOTE: this is a structural proxy, not semantic claim verification -- true "
            "claim-level support checking requires a draft answer from ReasoningAgent/"
            "ResponseGenerator, which do not exist yet (Milestone 3 gap, stated explicitly)."
        ),
    )
    reranker_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Average rerank_score across evidence with a score; 0.0 if reranking was fully degraded.",
    )
    verification_latency_ms: float = Field(ge=0.0)
