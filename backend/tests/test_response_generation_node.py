"""
Unit 3.9 tests -- ResponseGenerationNode graph node.
"""
import pytest

from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.response_generation import ResponseGenerationNode
from app.schemas.retrieval import Decision, DecisionAction
from app.services.response_generation.service import ResponseGenerator


@pytest.mark.asyncio
async def test_response_generation_node_populates_answer_on_state():
    node = ResponseGenerationNode(ResponseGenerator())
    decision = Decision(action=DecisionAction.PROCEED, confidence_score=0.95, reasons=["Clear evidence"])
    state = GraphState(original_query="how to return item", decision=decision)

    updated_state = await node(state)

    assert updated_state.answer is not None
    assert "how to return item" in updated_state.answer or "verified evidence" in updated_state.answer.lower()


@pytest.mark.asyncio
async def test_response_generation_node_handles_missing_decision():
    node = ResponseGenerationNode(ResponseGenerator())
    state = GraphState(original_query="query without decision")

    updated_state = await node(state)

    assert updated_state.answer is not None
    assert "Clarification required" in updated_state.answer
