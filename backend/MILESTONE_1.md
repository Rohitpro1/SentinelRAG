# SentinelRAG — Milestone 1: Core Logic Foundation

## Scope of this milestone
This milestone deliberately contains **no infrastructure** (no Postgres,
Redis, Qdrant, Docker, LangGraph). Per the roadmap principle "no milestone
should depend on unfinished work," this milestone proves out the two
pieces of logic every later layer depends on:

1. **Adaptive Chunking Engine** (`app/services/ingestion/chunker.py`)
2. **Decision Engine** (`app/services/decision_engine/engine.py`)

Both are pure, dependency-light, deterministic Python — no network calls,
no LLM calls — which is why they can be fully unit tested right now
without mocking half the stack.

## Why these two first
- The chunker determines retrieval quality for *everything* downstream.
  Bad chunking cannot be fixed by a smarter agent later.
- The Decision Engine is SentinelRAG's core differentiator (the
  self-correction claim in the PRD). It needed to exist as testable,
  deterministic code before any agent, prompt, or LangGraph node is
  built around it — otherwise "self-correcting" is just a diagram, not
  a system.

## What is explicitly NOT done yet (by design, not oversight)
- OCR, document parsing, table extraction → Milestone 2 (Ingestion Pipeline)
- Retriever Agent, embeddings, Qdrant integration → Milestone 2
- NLI model wiring (real cross-encoder inference) → Milestone 3
  (`PairwiseNLIResult` is currently a schema the engine consumes; the
  actual NLI model call is not implemented — this is a real gap, not
  hidden behind a fake "TODO: implement AI" comment)
- FastAPI endpoints, auth, LangGraph orchestration → Milestone 2/3
- Docker/CI → Milestone 4 (DevOps), once there's a running service worth containerizing

## Confidence formula — explicitly flagged as provisional
`compute_confidence()` uses hand-picked weights (0.45 similarity, 0.20 OCR,
0.20 reliability, -0.35 contradiction penalty). These are a reasonable
starting point, **not** a calibrated result. Milestone 4 (Evaluation Layer)
must run the confidence-calibration curve against the regression/eval
dataset and re-tune these weights — this file exists so nobody mistakes
the current numbers for a finished result.

## Running this milestone
```bash
cd backend
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
```
Expected: 19/19 tests passing (chunker: 7, decision engine: 12... see test files for exact breakdown).

## Next milestone (Milestone 2 — proposed)
1. FastAPI skeleton: `/documents/upload`, `/query` endpoints (schemas only, stub responses)
2. Document parser (Unstructured.io) + OCR (Tesseract) wired to the chunker above
3. Embedding service + Qdrant client
4. Retriever Agent producing real `RetrievedChunk` objects that feed the
   already-tested Decision Engine
5. docker-compose.yml for Qdrant + Postgres + Redis (local dev only, not production k8s yet)

Each of these is independently testable against the schemas already defined
in `app/schemas/retrieval.py`, so Milestone 2 can be built and reviewed
incrementally rather than as one large PR.
