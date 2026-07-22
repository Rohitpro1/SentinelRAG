"""
Unit 2.14 -- API-layer DTOs.

Deliberately separate from app/schemas (domain layer) -- per "the FastAPI
layer must remain an API adapter only," the HTTP contract is allowed to
evolve independently of internal domain schemas (e.g. renaming a field
for API stability without touching QueryResult), and internal domain
types are never serialized directly to a client.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.query import QueryResult


class QueryRequestBody(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=50)
    document_filter: Optional[dict] = None


class QueryResponseBody(BaseModel):
    action: str
    confidence: float
    reasons: list[str]
    retry_count: int
    contradiction_detected: bool
    evidence_coverage: float
    answer: Optional[str] = None

    @classmethod
    def from_query_result(cls, result: QueryResult) -> "QueryResponseBody":
        """
        The one place API <-> domain mapping happens. Field selection
        (not simply re-exporting QueryResult wholesale) is deliberate --
        e.g. `Explainability.triggered_thresholds` internal detail is not
        exposed here; only what a client needs is. This is data mapping,
        not business logic (no decisions are made here, just field
        selection/renaming).
        """
        return cls(
            action=result.decision.action.value,
            confidence=result.decision.confidence_score,
            reasons=result.decision.reasons,
            retry_count=result.retry_count,
            contradiction_detected=result.diagnostics.contradiction_detected,
            evidence_coverage=result.diagnostics.evidence_coverage,
            answer=result.answer,
        )


class ErrorResponseBody(BaseModel):
    """
    Uniform error shape across every mapped exception (instruction 3).
    `detail` is always a generic, client-safe message -- internal
    exception messages/stack traces are logged server-side (via
    log_event in the exception handlers) but never placed in this field.
    """

    detail: str
