"""
Unit 3.1 -- BaseGraphNode interface.

Every graph node (Planner, Retrieval, Verification, Decision) implements
this same shape: an async callable taking the current GraphState and
returning the updated GraphState. This uniform interface is what lets
GraphBuilder (graph_builder.py) wire any node into the graph identically,
regardless of what that node eventually does internally.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.orchestration.graph_state import GraphState


class BaseGraphNode(ABC):
    @abstractmethod
    async def __call__(self, state: GraphState) -> GraphState:
        raise NotImplementedError
