"""
Unit 3.4 -- VerificationNode: thin orchestration adapter around
VerificationAgent (Unit 2.9, frozen).

Scope, per explicit instruction: read the SearchResponse already in
state, invoke VerificationAgent exactly once, write both its business
output (VerifiedEvidence) and observability output (VerificationDiagnostics)
back into state, return it. NO retries, NO decision logic, NO planner
behavior -- all reserved for other nodes/units.

Same adapter discipline as RetrievalNode (Unit 3.3): VerificationAgent
still owns all of its own internal behavior (EvidenceValidator,
ContradictionDetector's graceful NLI-failure degradation, CoverageAnalyzer,
DiagnosticsBuilder) -- none of that is touched, duplicated, or bypassed
here. Since VerificationAgent.verify() is designed to degrade gracefully
internally (Unit 2.13) rather than raise under normal NLI-provider
failure, this node has nothing extra to catch in the common case; if
verify() is ever made to raise for some other reason, that exception
propagates through this node uncaught, exactly like RetrievalNode lets
RetrievalError propagate.
"""
from __future__ import annotations

from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.base import BaseGraphNode
from app.schemas.retrieval_domain import VerificationInput
from app.services.verification.verification_agent import VerificationAgent


class VerificationNode(BaseGraphNode):
    def __init__(self, verification_agent: VerificationAgent):
        # Constructor DI per instruction 4 -- no default, same reasoning
        # as RetrievalNode (Unit 3.3): VerificationAgent's real
        # dependencies (an NLI verifier, etc.) are a composition-root
        # concern, not something this node or GraphBuilder constructs.
        self._verification_agent = verification_agent

    async def __call__(self, state: GraphState) -> GraphState:
        ranked_chunks = state.retrieval_result.ranked_chunks if state.retrieval_result else []
        verification_input = VerificationInput(
            query=state.effective_query, ranked_chunks=ranked_chunks, retry_count=state.retry_count
        )
        verified_evidence, diagnostics = await self._verification_agent.verify(verification_input)
        return state.model_copy(update={"verification_result": verified_evidence, "diagnostics": diagnostics})
