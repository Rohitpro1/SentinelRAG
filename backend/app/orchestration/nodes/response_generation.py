"""
Unit 3.9 -- ResponseGenerationNode: orchestration wrapper around ResponseGenerator.

Scope, per requirement 2: do NOT place prompting logic inside LangGraph nodes.
The LangGraph node simply extracts state variables (decision, verification_result,
diagnostics, effective_query) and delegates answer generation to ResponseGenerator.
"""
from __future__ import annotations

from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.base import BaseGraphNode
from app.schemas.retrieval import Decision, DecisionAction
from app.services.response_generation.base import BaseResponseGenerator


class ResponseGenerationNode(BaseGraphNode):
    def __init__(self, response_generator: BaseResponseGenerator):
        # Constructor DI per requirement 5
        self._response_generator = response_generator

    async def __call__(self, state: GraphState) -> GraphState:
        # Defensive fallback if decision is missing
        decision = state.decision or Decision(
            action=DecisionAction.CLARIFY,
            confidence_score=0.0,
            reasons=["No decision produced by decision engine."],
        )

        answer = await self._response_generator.generate(
            decision=decision,
            evidence=state.verification_result,
            diagnostics=state.diagnostics,
            query=state.effective_query,
        )

        return state.model_copy(update={"answer": answer})
