"""
Unit 3.1 tests -- node interfaces are correctly shaped: each is a valid
BaseGraphNode subclass, and BaseGraphNode itself cannot be instantiated
directly (it's an ABC).

Unit 3.2 update: PlannerNode is now genuinely implemented (see
test_graph_planner_node.py for its real-behavior tests) and is EXCLUDED
from the "still a placeholder" parametrization below.

Unit 3.3 update: RetrievalNode is now genuinely implemented too (see
test_graph_retrieval_node.py) and requires constructor DI (a
RetrieverAgent) -- it can no longer be zero-arg constructed at all, so
it's also excluded from the "still a placeholder" list below (that list
now covers only VerificationNode and DecisionNode, unchanged from Unit
3.1/3.2).
Unit 3.4 update: VerificationNode is now genuinely implemented too (see
test_graph_verification_node.py) and requires constructor DI (a
VerificationAgent) -- also excluded from the "still a placeholder" list.

Unit 3.5 update: DecisionNode is now genuinely implemented too (see
test_graph_decision_node.py) and requires constructor DI (a
DecisionEngine). This means ALL FOUR node interfaces are now real
implementations -- _STILL_PLACEHOLDER_NODES is deliberately empty
(kept as a named, empty list rather than deleted, so this file's
structure doesn't need to change again if a future node interface is
ever added).
"""
import pytest

from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.base import BaseGraphNode
from app.orchestration.nodes.decision import DecisionNode
from app.orchestration.nodes.planner import PlannerNode
from app.orchestration.nodes.retrieval import RetrievalNode
from app.orchestration.nodes.verification import VerificationNode

_STILL_PLACEHOLDER_NODES: list = []
_ALL_NODES = [PlannerNode, RetrievalNode, VerificationNode, DecisionNode]


@pytest.mark.parametrize("node_cls", _ALL_NODES)
def test_node_is_a_valid_base_graph_node(node_cls):
    assert issubclass(node_cls, BaseGraphNode)


def test_base_graph_node_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseGraphNode()  # ABC with an abstract method -- must fail


@pytest.mark.parametrize("node_cls", _STILL_PLACEHOLDER_NODES)
@pytest.mark.asyncio
async def test_still_placeholder_nodes_raise_not_implemented_when_called(node_cls):
    node = node_cls()
    state = GraphState(original_query="q")
    with pytest.raises(NotImplementedError):
        await node(state)


def test_all_four_required_node_interfaces_exist():
    """Directly checks instruction 4's exact list of four node interfaces."""
    required = {"PlannerNode", "RetrievalNode", "VerificationNode", "DecisionNode"}
    actual = {cls.__name__ for cls in _ALL_NODES}
    assert required == actual
