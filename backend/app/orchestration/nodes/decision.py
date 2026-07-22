"""
Unit 3.5 -- DecisionNode: thin orchestration adapter around DecisionEngine
(Milestone 1, frozen).

Scope, per explicit instruction: read state, invoke DecisionEngine exactly
once, store the returned Decision, return state. NO branching (this unit
keeps the graph a straight line: planner -> retrieval -> verification ->
decision -> END), NO retry-loop migration from QueryService -- both
explicitly reserved for the next orchestration unit.

CLARIFICATION on "read SearchResponse / VerifiedEvidence /
VerificationDiagnostics": DecisionEngine.evaluate() (Milestone 1, frozen)
takes exactly one input -- a VerificationReport, produced here via
state.verification_result.to_verification_report() (Unit 2.9's adapter,
unchanged). state.retrieval_result (SearchResponse) and state.diagnostics
(VerificationDiagnostics) are available on state and are not discarded,
but DecisionEngine's frozen input contract has no parameter for either --
there is nothing to pass them INTO. This node does not reinterpret or
extend that contract; it uses exactly what DecisionEngine already accepts.
"""
from __future__ import annotations

from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.base import BaseGraphNode
from app.schemas.retrieval import VerificationReport
from app.services.decision_engine.engine import DecisionEngine


class DecisionNode(BaseGraphNode):
    def __init__(self, decision_engine: DecisionEngine):
        # Constructor DI per instruction 4 -- required, matching
        # RetrievalNode/VerificationNode's precedent (Units 3.3/3.4) of
        # never constructing a domain dependency internally.
        self._decision_engine = decision_engine

    async def __call__(self, state: GraphState) -> GraphState:
        # Defensive fallback for standalone testing/invocation without a
        # prior VerificationNode having run -- same pattern
        # VerificationNode (Unit 3.4) used for a missing retrieval_result:
        # treat "no verification_result yet" as "no evidence at all"
        # rather than raising an AttributeError.
        if state.verification_result is not None:
            report = state.verification_result.to_verification_report()
        else:
            report = VerificationReport(query=state.effective_query, retrieved_chunks=[], retry_count=state.retry_count)

        decision = self._decision_engine.evaluate(report, request_id=state.request_id, trace_id=state.trace_id)
        return state.model_copy(update={"decision": decision})
