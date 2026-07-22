"""
Unit 3.6 -- conditional routing logic.

Kept as a standalone, pure, easily-unit-tested function rather than a
method buried inside GraphBuilder -- LangGraph's add_conditional_edges
just needs any callable(state) -> str, and testing this function
directly (no graph compilation needed) is the cheapest possible test of
the routing decision itself.

NAMING RECONCILIATION, stated explicitly: the instruction that specified
this unit named four conceptual outcomes -- ANSWER, RETRY_RETRIEVAL,
CLARIFY, FAIL. The frozen DecisionAction enum (Milestone 1,
app/schemas/retrieval.py) has five actual values: PROCEED,
RETRY_RETRIEVAL, CLARIFY, LOW_CONFIDENCE_RESPONSE, HUMAN_REVIEW -- there
is no ANSWER or FAIL member. Rather than add either to a frozen enum
(forbidden) or silently reinterpret the instruction, this function maps
faithfully onto the actual enum:
  - PROCEED and LOW_CONFIDENCE_RESPONSE -> "end" (both are "answer"
    outcomes -- the frozen Architecture Enhancements lifecycle design
    explicitly treats a transparent low-confidence answer as a completed
    response, not a failure)
  - CLARIFY -> "end" (exact match to the instruction's CLARIFY)
  - HUMAN_REVIEW -> "end" (closest analog to the instruction's FAIL --
    a decision the system can't resolve automatically; there is no
    human-review-waiting mechanism built yet, so this terminates the
    graph run rather than looping)
  - RETRY_RETRIEVAL -> "retry" (exact match, the only actual branch)
"""
from __future__ import annotations

from app.orchestration.graph_state import GraphState
from app.schemas.retrieval import DecisionAction


def route_after_decision(state: GraphState) -> str:
    if state.decision is not None and state.decision.action == DecisionAction.RETRY_RETRIEVAL:
        return "retry"
    return "end"
