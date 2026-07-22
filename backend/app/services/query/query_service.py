"""
Unit 2.14 -- QueryService (original hand-rolled while-loop orchestration).
Unit 3.8 -- REWRITTEN to delegate to the compiled LangGraph instead of
maintaining its own parallel retry loop, per explicit instruction 4
("remove duplicate orchestration logic ... replacing it with graph
execution"). The self-correction loop -- retrieve, verify, decide, and
on RETRY_RETRIEVAL do it again with a rewritten query -- now lives
entirely in the LangGraph built by GraphBuilder (Units 3.1-3.7):
PlannerNode -> RetrievalNode -> VerificationNode -> DecisionNode, with
DecisionNode's conditional routing (Unit 3.6) closing the retry loop
that this class used to implement as a Python `while True`.

PUBLIC API UNCHANGED (instruction 3): handle_query()'s signature and
QueryResult's shape are byte-for-byte identical to Unit 2.14's. This is
what lets the FastAPI route (query_router.py) remain completely
untouched -- it only ever called `query_service.handle_query(...)` and
still does, with no knowledge that the internals changed.

DEPENDENCY INJECTION (instruction 5): QueryService now takes a
CompiledStateGraph via constructor -- it never calls GraphBuilder or
constructs a graph itself. The composition root (app/api/dependencies.py)
is responsible for building the graph (wiring RetrieverAgent,
VerificationAgent, DecisionEngine into RetrievalNode/VerificationNode/
DecisionNode, then GraphBuilder(...).build()) and injecting the compiled
result here.

BOUNDARY, still true: this is orchestration gluing together frozen
Milestone 2 domain components (via the graph nodes that wrap them) --
none of RetrieverAgent, VerificationAgent, DecisionEngine, or their
sub-components were touched to make this integration work.
"""
from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph.state import CompiledStateGraph  # type: ignore[import-not-found,import-untyped]

from app.core.logging import get_logger, log_event
from app.orchestration.graph_state import GraphState
from app.schemas.query import QueryResult


class QueryService:
    def __init__(self, compiled_graph: CompiledStateGraph, logger: Optional[logging.Logger] = None):
        self._compiled_graph = compiled_graph
        self._logger = logger or get_logger(__name__)

    async def handle_query(
        self,
        query: str,
        *,
        top_k: int = 20,
        rerank_top_n: int = 5,
        document_filter: Optional[dict] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> QueryResult:
        initial_state = GraphState(
            original_query=query, top_k=top_k, rerank_top_n=rerank_top_n,
            document_filter=document_filter, request_id=request_id, trace_id=trace_id,
        )

        raw_result = await self._compiled_graph.ainvoke(initial_state)
        # Unit 3.1 documented (and tested) that CompiledStateGraph.ainvoke()
        # on a Pydantic-schema graph returns a dict that omits unset/None
        # fields -- GraphState(**raw_result) is the correct reconstruction,
        # relying on every optional field's default, exactly as every
        # orchestration-layer test since Unit 3.1 has done.
        final_state = GraphState(**raw_result)

        log_event(
            self._logger, "query_completed",
            request_id=request_id, trace_id=trace_id,
            action=final_state.decision.action.value if final_state.decision else None,
            confidence=final_state.decision.confidence_score if final_state.decision else None,
            retry_count=final_state.retry_count,
        )

        if final_state.decision is None or final_state.diagnostics is None:
            raise ValueError("Graph execution completed without producing decision or diagnostics.")

        return QueryResult(
            decision=final_state.decision,
            diagnostics=final_state.diagnostics,
            retry_count=final_state.retry_count,
            answer=final_state.answer,
        )
