# SentinelRAG — Production-Grade Self-Correcting RAG Framework

SentinelRAG is a production-grade, self-correcting Retrieval-Augmented Generation (RAG) platform powered by **LangGraph**, **FastAPI**, **Qdrant**, **PostgreSQL**, **Redis**, and **Google Gemini AI**.

---

## 🌟 Key Subsystems

1. **Self-Correcting LangGraph State Machine**:
   - `Planner Node` -> `Retrieval Node` -> `Verification Node` -> `Decision Node` -> `Response Generation Node`.
   - Conditional edge retry loops (`RETRY_RETRIEVAL`) with query rewriting on sub-threshold confidence.

2. **Provider-Based AI Layer (Gemini First)**:
   - Default Production AI Provider: Google Gemini (`gemini-2.5-flash` for reasoning & NLI, `gemini-embedding-001` for vector embeddings).
   - Deterministic testing providers for zero-dependency offline execution & CI/CD.

3. **Knowledge Ingestion Pipeline**:
   - Supports **PDF**, **DOCX**, **TXT**, and **Markdown** file uploads via REST API (`POST /api/v1/documents/upload`).
   - SHA-256 duplicate detection fingerprinting (`status: already_exists`).
   - Adaptive sentence-boundary chunking with token budgets and overlaps (`SentenceChunker`).
   - Qdrant Vector DB payload storage & metadata management.
   - Non-blocking asynchronous processing with polling endpoint (`GET /api/v1/documents/{id}/status`).

4. **Observability & Visualizer Frontend**:
   - Built matching the Stitch MCP design system (`#4F46E5` primary, Geist & JetBrains Mono typography, glassmorphism UI).
   - Live query playground, parameter controls (`top_k`, `rerank_top_n`), NLI contradiction stats, evidence coverage telemetry, and vector chunk browser.

---

## 🚀 Quickstart & API Documentation

### Environment Setup (`.env`)
```env
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

STORAGE__QDRANT_URL=https://xxxxxx.cloud.qdrant.io:6333
STORAGE__QDRANT_API_KEY=your_qdrant_cloud_key
STORAGE__POSTGRES_DSN=postgresql+asyncpg://user:pass@ep-xyz.neon.tech/sentinelrag?sslmode=require
STORAGE__REDIS_URL=rediss://default:pass@xyz.upstash.io:6379
```

### Document Ingestion (cURL Example)
```bash
curl -X POST \
  -H "Authorization: Bearer dev-token" \
  -F "file=@knowledge_policy.pdf" \
  http://localhost:8000/api/v1/documents/upload
```

### RAG Query Execution (cURL Example)
```bash
curl -X POST \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k": 20, "rerank_top_n": 5}' \
  http://localhost:8000/api/v1/query
```

---

## 🧪 Testing & Quality Assurance

```bash
# Run pytest test suite (365 tests)
python -m pytest tests

# Run mypy static type checking
python -m mypy app
```

---

## 📚 Reports & Architecture Docs

- [INGESTION_PIPELINE_REPORT.md](file:///c:/Users/ROHIT/Downloads/sentinelrag/docs/INGESTION_PIPELINE_REPORT.md)
- [PROVIDER_MIGRATION_REPORT.md](file:///c:/Users/ROHIT/Downloads/sentinelrag/docs/PROVIDER_MIGRATION_REPORT.md)
- [FINAL_RELEASE_AUDIT_REPORT.md](file:///c:/Users/ROHIT/Downloads/sentinelrag/docs/FINAL_RELEASE_AUDIT_REPORT.md)
