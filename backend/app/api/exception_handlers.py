"""
Unit 2.14 -- domain exception -> HTTP response mapping (instruction 3).

Registered on the FastAPI app (app/main.py), never inline in the route --
this is what lets the route stay "exception mapping" as a one-line
concern (just letting exceptions propagate) rather than a try/except
ladder duplicated in every endpoint.

Every handler logs the real exception server-side (structured, via
log_event) and returns a GENERIC, client-safe message -- internal
messages, exception types, and stack traces never reach the response
body, per "avoid leaking internal implementation details."
"""
from __future__ import annotations

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.v1.schemas import ErrorResponseBody
from app.core.exceptions import (
    DecisionEngineError,
    EmbeddingError,
    RetrievalError,
    SentinelRAGError,
    VerificationError,
)
from app.core.logging import get_logger, log_event

_logger = get_logger(__name__)


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorResponseBody(detail=detail).model_dump())


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Instruction 3 explicitly maps ValidationError -> 400 (FastAPI's
    # default for RequestValidationError is 422 -- overridden here
    # deliberately to match the specified contract).
    log_event(_logger, "request_validation_failed", level=logging.INFO, path=str(request.url), errors=str(exc.errors()))
    return _error_response(400, "Invalid request.")


async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    log_event(_logger, "retrieval_error", level=logging.ERROR, path=str(request.url), error=exc.message, transient=exc.transient)
    return _error_response(503, "Retrieval service is temporarily unavailable. Please try again.")


async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
    # Grouped with RetrievalError's 503 -- both are retrieval-path
    # infrastructure dependencies from the client's perspective, even
    # though they're distinct exception types internally.
    log_event(_logger, "embedding_error", level=logging.ERROR, path=str(request.url), error=exc.message)
    return _error_response(503, "Retrieval service is temporarily unavailable. Please try again.")


async def verification_error_handler(request: Request, exc: VerificationError) -> JSONResponse:
    log_event(_logger, "verification_error", level=logging.ERROR, path=str(request.url), error=exc.message)
    return _error_response(502, "Verification service returned an invalid response.")


async def decision_engine_error_handler(request: Request, exc: DecisionEngineError) -> JSONResponse:
    log_event(_logger, "decision_engine_error", level=logging.ERROR, path=str(request.url), error=exc.message)
    return _error_response(500, "An internal error occurred while processing the request.")


async def sentinelrag_error_handler(request: Request, exc: SentinelRAGError) -> JSONResponse:
    """Safety net for any SentinelRAGError subclass without a dedicated handler above (e.g. SecurityError, ChunkingError)."""
    log_event(
        _logger, "unhandled_sentinelrag_error", level=logging.ERROR,
        path=str(request.url), error_type=type(exc).__name__, error=exc.message,
    )
    return _error_response(500, "An internal error occurred while processing the request.")


EXCEPTION_HANDLERS = {
    RequestValidationError: validation_error_handler,
    RetrievalError: retrieval_error_handler,
    EmbeddingError: embedding_error_handler,
    VerificationError: verification_error_handler,
    DecisionEngineError: decision_engine_error_handler,
    SentinelRAGError: sentinelrag_error_handler,
}
