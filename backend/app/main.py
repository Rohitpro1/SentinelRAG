"""
Unit 2.14 -- FastAPI application assembly.

This module's only job is wiring: create the app, register the router,
register the exception handlers. No business logic lives here, same as
the router itself.
"""
from __future__ import annotations

from fastapi import FastAPI

from app.api.exception_handlers import EXCEPTION_HANDLERS
from app.api.v1.query_router import router as query_router
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(title="SentinelRAG", version="0.1.0")
app.include_router(query_router, prefix="/api/v1")

for exc_type, handler in EXCEPTION_HANDLERS.items():
    app.add_exception_handler(exc_type, handler)  # type: ignore[arg-type]
