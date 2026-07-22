"""
Unit 3.1 tests -- GraphBuilder: compilation, minimal START->END
invocation, state pass-through, and DI acceptance of not-yet-used nodes.
"""
import pytest
from langgraph.graph.state import CompiledStateGraph

from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.decision import DecisionNode
from app.orchestration.nodes.planner import PlannerNode
from app.orchestration.nodes.retrieval import RetrievalNode
from app.orchestration.nodes.verification import VerificationNode


def test_build_returns_a_compiled_state_graph():
    compiled = GraphBuilder().build()
    assert isinstance(compiled, CompiledStateGraph)


def test_constructor_accepts_no_arguments():
    """Default construction (no nodes injected) must work -- Unit 3.1's minimal graph doesn't need them yet."""
    builder = GraphBuilder()
    assert builder is not None


def test_all_four_nodes_are_used_once_all_are_injected_or_defaulted():
    """
    Historical note: this test slot previously covered "nodes injected
    but not yet used" (Units 3.2-3.4, as retrieval/verification/decision
    were added one at a time). As of Unit 3.5, EVERY node interface is
    genuinely implemented and planner_node/decision_node both default to
    real instances -- there is no longer any node that can be injected
    without being used. This test replaces that now-obsolete premise with
    a direct check that all four actually run and populate their
    corresponding state fields.
    """
    from app.orchestration.nodes.retrieval import RetrievalNode
    from app.orchestration.nodes.verification import VerificationNode

    class _StubRetrieverAgent:
        async def search(self, request):
            from app.schemas.retrieval_domain import SearchResponse
            return SearchResponse(request=request, ranked_chunks=[])

    class _StubVerificationAgent:
        async def verify(self, verification_input):
            from app.schemas.retrieval_domain import VerifiedEvidence, VerificationDiagnostics
            evidence = VerifiedEvidence(query=verification_input.query, retrieved_chunks=[], retry_count=0)
            diagnostics = VerificationDiagnostics(
                query=verification_input.query, nli_score=1.0, contradiction_detected=False,
                evidence_coverage=0.0, unsupported_claims=[], reranker_confidence=0.0, verification_latency_ms=0.0,
            )
            return evidence, diagnostics

    builder = GraphBuilder(
        retrieval_node=RetrievalNode(_StubRetrieverAgent()),
        verification_node=VerificationNode(_StubVerificationAgent()),
    )
    compiled = builder.build()
    assert isinstance(compiled, CompiledStateGraph)


@pytest.mark.asyncio
async def test_minimal_graph_returns_state_unchanged():
    compiled = GraphBuilder().build()
    input_state = GraphState(original_query="what is the refund policy?", retry_count=2)
    result = await compiled.ainvoke(input_state)
    reconstructed = GraphState(**result)
    assert reconstructed.original_query == "what is the refund policy?"
    assert reconstructed.retry_count == 2


@pytest.mark.asyncio
async def test_minimal_graph_accepts_a_plain_dict_as_input():
    """LangGraph invocation also accepts a plain dict matching the schema, not only a GraphState instance."""
    compiled = GraphBuilder().build()
    result = await compiled.ainvoke({"original_query": "test query"})
    assert result["original_query"] == "test query"


@pytest.mark.asyncio
async def test_default_graph_recomputes_decision_via_default_decision_node():
    """
    Unit 3.5 update: GraphBuilder() now defaults a real DecisionNode into
    the graph (DecisionEngine is pure/cheap, same reasoning as
    PlannerNode's default). A pre-set input decision is NOT preserved --
    it is intentionally recomputed by the engine based on whatever
    verification_result (or lack thereof) is present, exactly as running
    the real pipeline should behave. With no retrieval_result and no
    verification_result, DecisionNode's defensive fallback (Unit 3.5)
    treats this as "no evidence," and DecisionEngine's own frozen
    threshold logic (Milestone 1) decides the outcome from there.
    """
    from app.schemas.retrieval import Decision, DecisionAction

    compiled = GraphBuilder().build()
    stale_decision = Decision(action=DecisionAction.PROCEED, confidence_score=0.99, reasons=["stale, should be overwritten"])
    input_state = GraphState(original_query="q", decision=stale_decision)
    result = await compiled.ainvoke(input_state)
    reconstructed = GraphState(**result)
    # retry_count=0 (default) with no chunks -> DecisionEngine's frozen
    # logic returns RETRY_RETRIEVAL (Milestone 1, unmodified) -- NOT the
    # stale PROCEED that was passed in.
    assert reconstructed.decision.action == DecisionAction.RETRY_RETRIEVAL
    assert reconstructed.decision.reasons != ["stale, should be overwritten"]


def test_build_can_be_called_multiple_times_independently():
    """Each build() call produces an independently usable compiled graph -- no shared mutable state between calls."""
    builder = GraphBuilder()
    compiled_1 = builder.build()
    compiled_2 = builder.build()
    assert compiled_1 is not compiled_2
