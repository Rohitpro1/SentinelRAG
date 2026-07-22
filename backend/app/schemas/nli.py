"""
Unit 2.13 -- NLI domain observability schema.

NLIResult is ADDITIVE: does not change BaseNLIVerifier's frozen contract
(verify_pair() still returns tuple[NLIRelation, float], exactly as Unit
2.9 approved).

Per instruction 2, health is NOT a new abstraction here -- EmbedderHealth
/ EmbedderHealthState (app.schemas.embedding, Unit 2.11) are reused
verbatim by the NLI verifier's health() method. The names are historically
embedding-specific (a minor, deliberately-accepted cosmetic debt); a
generic rename to ProviderHealth would be a good non-breaking future
cleanup, but renaming now would touch Unit 2.11's already-approved tests
for no functional gain -- not done here. Reranking (Unit 2.12) did not
add a health() method at all -- if cross-provider consistency there is
wanted later, adding it to CrossEncoderReranker using this same
EmbedderHealth type would be the natural follow-up, out of scope for this
unit since Unit 2.12 is already approved.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.retrieval import NLIRelation


class NLIResult(BaseModel):
    """
    Business output remains `label` (feeds ContradictionDetector's
    contradiction logic exactly as the existing NLIRelation return does).
    Per this unit's explicit instruction, `confidence` is grouped under
    observability metadata here -- a deliberate departure from Units
    2.9/2.11/2.12, where the analogous numeric score (rerank_score,
    similarity_score) was part of the business output. Documented rather
    than silently inconsistent: the categorical label is what
    ContradictionDetector/DecisionEngine act on; confidence here is a
    diagnostic strength signal for dashboards, not a second business value.
    """

    label: NLIRelation
    provider: str
    model_name: str
    model_version: Optional[str] = None
    latency_ms: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
