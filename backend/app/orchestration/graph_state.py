"""
Unit 3.1 -- GraphState.

The LangGraph orchestration layer's state model. Composed ENTIRELY from
existing, frozen Milestone 2 domain schemas (SearchResponse,
VerifiedEvidence, Decision, VerificationDiagnostics) -- this file
introduces zero new business types, only a container that carries them
through a graph. Nothing in app/services or app/schemas is modified to
support this; GraphState imports those schemas the same way any other
consumer would.

Unit 3.8 addition: top_k, rerank_top_n, document_filter, request_id,
trace_id. These are NOT new business concepts -- every one already
exists as a QueryService.handle_query() parameter (Unit 2.14) and a
SearchRequest field (Unit 2.1). They were simply never threaded through
GraphState because no unit before 3.8 needed the graph to carry a full
API-equivalent request. Integrating the graph into QueryService
(instruction 2: "construct the initial GraphState") requires these
fields to exist on GraphState, or QueryService's public API would
silently stop honoring top_k/rerank_top_n/document_filter the moment its
internals switched to graph execution -- a regression instruction 3
("preserve the public API") explicitly forbids.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.planning import PlanningMetadata
from app.schemas.retrieval import Decision
from app.schemas.retrieval_domain import SearchResponse, VerificationDiagnostics, VerifiedEvidence


class GraphState(BaseModel):
    """
    Carries a single query's progress through the orchestration graph.

    Field-by-field rationale:
      - original_query: the user's query text, set once at graph entry,
        never mutated by any node.
      - top_k / rerank_top_n / document_filter: request-shaping
        parameters, set once at graph entry from QueryService's incoming
        call (Unit 3.8), read by RetrievalNode when building each
        SearchRequest. Defaults match SearchRequest's own defaults
        exactly, so omitting them reproduces prior behavior precisely.
      - request_id / trace_id: tracing identifiers, set once at graph
        entry, threaded through RetrievalNode and DecisionNode's
        structured-logging calls (Unit 3.8) -- preserves the log
        correlation QueryService's original while-loop implementation
        (Unit 2.14) already provided.
      - planning_metadata: populated by PlannerNode (Unit 3.2) --
        normalized query text, a deterministic classification, and basic
        counts.
      - rewritten_query: populated by PlannerNode on retry (Unit 3.7) --
        a deterministic, stopword-stripped rewrite of original_query.
      - retrieval_result: RetrieverAgent's SearchResponse (Unit 2.6), via
        RetrievalNode (Unit 3.3).
      - verification_result: VerificationAgent's VerifiedEvidence (Unit
        2.9) -- the business output, same object type QueryService
        already consumed pre-graph-integration.
      - decision: DecisionEngine's Decision (Milestone 1, frozen), via
        DecisionNode (Unit 3.5).
      - retry_count: incremented by RetryIncrementNode (Unit 3.6) on each
        RETRY_RETRIEVAL loop pass, bounded by DecisionEngineSettings.
        max_retrieval_retries (Milestone 1, frozen, unmodified).
      - diagnostics: VerificationAgent's VerificationDiagnostics (Unit
        2.9) -- the observability output, kept separate from
        verification_result exactly as that unit's business/observability
        split established.
    """

    original_query: str
    top_k: int = Field(default=20, ge=1)
    rerank_top_n: int = Field(default=5, ge=1)
    document_filter: Optional[dict] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    planning_metadata: Optional[PlanningMetadata] = None
    rewritten_query: Optional[str] = None
    retrieval_result: Optional[SearchResponse] = None
    verification_result: Optional[VerifiedEvidence] = None
    decision: Optional[Decision] = None
    retry_count: int = Field(default=0, ge=0)
    diagnostics: Optional[VerificationDiagnostics] = None
    answer: Optional[str] = None

    @property
    def effective_query(self) -> str:
        """The query a RetrievalNode should actually use -- the rewritten one if present, else the original."""
        return self.rewritten_query or self.original_query
