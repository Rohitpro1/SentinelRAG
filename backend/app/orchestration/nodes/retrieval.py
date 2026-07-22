"""
Unit 3.3 -- RetrievalNode: first orchestration wrapper around an existing
Milestone 2 component.

Scope, per explicit instruction: choose the effective query (rewritten if
present, else original -- reusing GraphState.effective_query from Unit
3.1, not reimplementing that fallback logic here), invoke RetrieverAgent
exactly once, store the result, return the updated state. NO retries, NO
verification, NO decision logic, NO query rewriting -- all reserved for
later units.

This node is an ADAPTER, not a reimplementation: RetrieverAgent (Unit
2.6, frozen) still owns its own internal timeout/retry/degradation
behavior (EmbeddingService, SearchService, RerankingService) -- none of
that is touched, duplicated, or bypassed here. A RetrievalError raised by
RetrieverAgent.search() is NOT caught here; it propagates to whatever
invokes the compiled graph, exactly as QueryService (Unit 2.14) lets it
propagate to the FastAPI exception handlers today. Retrying at the graph
level (i.e. re-running this node after a RETRY_RETRIEVAL decision) is
explicitly reserved for a later unit once DecisionNode exists.
"""
from __future__ import annotations

from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.base import BaseGraphNode
from app.schemas.retrieval_domain import SearchRequest
from app.services.retrieval.retriever_agent import RetrieverAgent


class RetrievalNode(BaseGraphNode):
    def __init__(self, retriever_agent: RetrieverAgent):
        # Constructor DI per instruction 4 -- no default, unlike
        # PlannerNode's zero-arg-constructable default (Unit 3.2).
        # RetrieverAgent has real dependencies (embedder, vector
        # repository, reranker) that only a composition root should
        # construct -- RetrievalNode never builds one itself.
        self._retriever_agent = retriever_agent

    async def __call__(self, state: GraphState) -> GraphState:
        # Unit 3.8: SearchRequest now carries every request-shaping field
        # GraphState holds, not just query/retry_count -- required so
        # QueryService's top_k/rerank_top_n/document_filter parameters
        # keep working once QueryService delegates to this graph instead
        # of building SearchRequest itself.
        request = SearchRequest(
            query=state.effective_query,
            top_k=state.top_k,
            rerank_top_n=state.rerank_top_n,
            document_filter=state.document_filter,
            retry_count=state.retry_count,
            request_id=state.request_id,
            trace_id=state.trace_id,
        )
        search_response = await self._retriever_agent.search(request)
        return state.model_copy(update={"retrieval_result": search_response})
