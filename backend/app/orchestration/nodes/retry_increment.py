"""
Unit 3.6 -- RetryIncrementNode.

Pure orchestration glue: increments GraphState.retry_count by exactly
one. Exists so retry-count bookkeeping lives ENTIRELY in the
orchestration layer (instruction 6: "retry behavior belongs entirely to
the orchestration layer") -- not inside RetrieverAgent, DecisionEngine,
or QueryService, none of which are touched by this unit.

Inserted on the RETRY_RETRIEVAL edge, between DecisionNode's conditional
routing and RetrievalNode's re-entry, so RetrievalNode always sees an
up-to-date retry_count via GraphState.retry_count when it builds its
SearchRequest on the next pass -- the same increment-once-per-loop
behavior QueryService (Unit 2.14) already has, just relocated to a graph
edge instead of a Python while-loop iteration.
"""
from __future__ import annotations

from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.base import BaseGraphNode


class RetryIncrementNode(BaseGraphNode):
    async def __call__(self, state: GraphState) -> GraphState:
        return state.model_copy(update={"retry_count": state.retry_count + 1})
