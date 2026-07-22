"""
Unit 2.14 -- FastAPI application assembly.

This module's only job is wiring: create the app, register the router,
register the exception handlers. No business logic lives here, same as
the router itself.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.exception_handlers import EXCEPTION_HANDLERS
from app.api.v1.documents_router import router as documents_router
from app.api.v1.query_router import router as query_router
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(title="SentinelRAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "SentinelRAG", "version": "0.1.0"}


for exc_type, handler in EXCEPTION_HANDLERS.items():
    app.add_exception_handler(exc_type, handler)  # type: ignore[arg-type]
