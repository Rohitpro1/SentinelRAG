"""
Shared data contracts between Ingestion, Retriever, Verification, and
the Decision Engine.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    token_count: int
    source_reliability_score: float = Field(ge=0.0, le=1.0)
    ocr_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    similarity_score: float = Field(ge=0.0, le=1.0)
    rerank_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class NLIRelation(str, Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class PairwiseNLIResult(BaseModel):
    chunk_id_a: str
    chunk_id_b: str
    relation: NLIRelation
    confidence: float = Field(ge=0.0, le=1.0)


class VerificationReport(BaseModel):
    query: str
    retrieved_chunks: list[RetrievedChunk]
    nli_results: list[PairwiseNLIResult] = Field(default_factory=list)
    retry_count: int = 0

    @property
    def max_similarity(self) -> float:
        if not self.retrieved_chunks:
            return 0.0
        return max(rc.similarity_score for rc in self.retrieved_chunks)

    @property
    def has_contradiction(self) -> bool:
        return any(r.relation == NLIRelation.CONTRADICTION for r in self.nli_results)

    @property
    def max_contradiction_confidence(self) -> float:
        contras = [r.confidence for r in self.nli_results if r.relation == NLIRelation.CONTRADICTION]
        return max(contras) if contras else 0.0


class DecisionAction(str, Enum):
    PROCEED = "proceed"
    RETRY_RETRIEVAL = "retry_retrieval"
    CLARIFY = "clarify"
    LOW_CONFIDENCE_RESPONSE = "low_confidence_response"
    HUMAN_REVIEW = "human_review"


class TriggeredThreshold(BaseModel):
    name: str
    threshold_value: float
    observed_value: float
    triggered: bool


class ContributingSignal(BaseModel):
    name: str
    value: float
    weight: Optional[float] = None


class Explainability(BaseModel):
    """Full explanation of one Decision Engine routing decision — feeds the dashboard directly."""

    action: DecisionAction
    confidence: float = Field(ge=0.0, le=1.0)
    triggered_thresholds: list[TriggeredThreshold] = Field(default_factory=list)
    contributing_signals: list[ContributingSignal] = Field(default_factory=list)
    human_readable_reasons: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    action: DecisionAction
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    rewritten_query: Optional[str] = None
    explainability: Optional[Explainability] = None
