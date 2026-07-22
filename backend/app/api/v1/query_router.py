"""
Unit 2.14 -- /query endpoint.

Per instruction 1, this router's ENTIRE responsibility list: request
validation (Pydantic model, automatic), authentication (Depends on the
placeholder), dependency resolution (Depends on get_query_service),
service invocation (one call), and HTTP response generation (one
mapping call). No business logic, no orchestration, no exception
try/except ladder -- exceptions propagate to the handlers registered in
app/main.py (instruction 3's mapping happens there, not here).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_principal, get_query_service
from app.api.v1.schemas import QueryRequestBody, QueryResponseBody
from app.services.query.query_service import QueryService

router = APIRouter()


@router.post("/query", response_model=QueryResponseBody)
async def query(
    body: QueryRequestBody,
    query_service: QueryService = Depends(get_query_service),
    _principal: str = Depends(get_current_principal),
) -> QueryResponseBody:
    result = await query_service.handle_query(
        body.query, top_k=body.top_k, rerank_top_n=body.rerank_top_n, document_filter=body.document_filter,
    )
    return QueryResponseBody.from_query_result(result)
