"""
Unit 3.2 -- Planning domain schemas.

PlanningMetadata is what PlannerNode populates on GraphState. It is
deliberately shaped as data-only (no methods, no logic) -- all
classification logic lives in nodes/planner.py, not here, matching the
project's schema/logic separation everywhere else (e.g. Chunk vs.
SentenceChunker).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QueryClassification(str, Enum):
    """
    Deterministic, rule-based classification (instruction 3: no LLM
    reasoning). Every value is reachable by a simple, explainable rule
    over the normalized query text -- see nodes/planner.py's _classify().
    """

    EMPTY = "empty"
    TOO_SHORT = "too_short"
    QUESTION = "question"
    STATEMENT = "statement"
    MULTI_PART = "multi_part"


class PlanningMetadata(BaseModel):
    normalized_query: str
    classification: QueryClassification
    word_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
