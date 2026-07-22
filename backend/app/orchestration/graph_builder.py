"""
Unit 3.1 -- GraphBuilder.
Unit 3.2 -- wired PlannerNode in as the graph's first real node.
Unit 3.3 -- optionally wired RetrievalNode in as a second real node.
Unit 3.4 -- optionally wired VerificationNode in as a third real node;
generalized node-chaining to a linear sequence.
Unit 3.5 -- optionally wired DecisionNode in as a fourth real node.
Unit 3.6 -- replaced the straight-line ending after DecisionNode with
CONDITIONAL routing: RETRY_RETRIEVAL loops back (via RetryIncrementNode,
which bumps retry_count -- instruction 6: retry bookkeeping lives
entirely in this orchestration layer); every other Decision.action
terminates the graph. See app/orchestration/routing.py for the exact
action -> edge mapping and its naming reconciliation with the frozen
DecisionAction enum.
Unit 3.7 -- the retry loop's target changed from "retrieval" directly to
"planner": PlannerNode is now retry-aware (generates a deterministic
rewritten_query when retry_count > 0), so retries must pass back through
it before reaching RetrievalNode again, rather than bypassing it.
RetrievalNode is unaffected -- it still just reads
GraphState.effective_query.
Unit 3.9 -- wired ResponseGenerationNode in as the final answer generation
node before END. Every non-retry action routes from DecisionNode to
ResponseGenerationNode to populate GraphState.answer before reaching END.

Assembles the LangGraph workflow. Constructor accepts all node interfaces
via DI. planner_node, decision_node, and response_generation_node default
to real, cheap/pure instances. retrieval_node and verification_node have NO
defaults -- their real dependencies remain a composition-root concern.
retry_increment_node defaults to a real RetryIncrementNode().
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found,import-untyped]
from langgraph.graph.state import CompiledStateGraph  # type: ignore[import-not-found,import-untyped]

from app.core.settings.decision_engine import DecisionEngineSettings
from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.decision import DecisionNode
from app.orchestration.nodes.planner import PlannerNode
from app.orchestration.nodes.response_generation import ResponseGenerationNode
from app.orchestration.nodes.retrieval import RetrievalNode
from app.orchestration.nodes.retry_increment import RetryIncrementNode
from app.orchestration.nodes.verification import VerificationNode
from app.orchestration.routing import route_after_decision
from app.services.decision_engine.engine import DecisionEngine
from app.services.response_generation.service import ResponseGenerator


class GraphBuilder:
    def __init__(
        self,
        planner_node: Optional[PlannerNode] = None,
        retrieval_node: Optional[RetrievalNode] = None,
        verification_node: Optional[VerificationNode] = None,
        decision_node: Optional[DecisionNode] = None,
        retry_increment_node: Optional[RetryIncrementNode] = None,
        response_generation_node: Optional[ResponseGenerationNode] = None,
    ):
        self._planner_node = planner_node or PlannerNode()
        self._retrieval_node = retrieval_node
        self._verification_node = verification_node
        self._decision_node = decision_node or DecisionNode(DecisionEngine(DecisionEngineSettings()))
        self._retry_increment_node = retry_increment_node or RetryIncrementNode()
        self._response_generation_node = response_generation_node or ResponseGenerationNode(ResponseGenerator())

    def build(self) -> CompiledStateGraph:
        """
        Unit 3.9: planner -> [retrieval] -> [verification] -> decision,
        then CONDITIONAL: RETRY_RETRIEVAL -> retry_increment -> planner
        (looping back); every other action -> response_generation -> END.
        """
        graph = StateGraph(GraphState)
        graph.add_node("planner", self._planner_node)
        graph.add_edge(START, "planner")
        last_node = "planner"

        if self._retrieval_node is not None:
            graph.add_node("retrieval", self._retrieval_node)
            graph.add_edge(last_node, "retrieval")
            last_node = "retrieval"

        if self._verification_node is not None:
            graph.add_node("verification", self._verification_node)
            graph.add_edge(last_node, "verification")
            last_node = "verification"

        graph.add_node("decision", self._decision_node)
        graph.add_edge(last_node, "decision")

        # Always include response_generation node
        graph.add_node("response_generation", self._response_generation_node)
        graph.add_edge("response_generation", END)

        if self._retrieval_node is not None:
            graph.add_node("retry_increment", self._retry_increment_node)
            graph.add_edge("retry_increment", "planner")
            graph.add_conditional_edges(
                "decision", route_after_decision, {"retry": "retry_increment", "end": "response_generation"}
            )
        else:
            # Straight-line configuration when no retrieval step exists to retry into
            graph.add_edge("decision", "response_generation")

        return graph.compile()
