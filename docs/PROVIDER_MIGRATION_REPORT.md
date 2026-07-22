# SentinelRAG Provider-Based AI Architecture Migration Report (Gemini First)

**Date:** July 22, 2026  
**Stage:** Provider Layer Modernization & Cloud Infrastructure Hardening  
**Target:** Render (Backend) + Vercel (Frontend) + Google Gemini + Qdrant Cloud + Neon PostgreSQL + Upstash Redis  

---

## 1. Executive Summary

SentinelRAG backend has been successfully refactored into a **Provider-Agnostic AI Architecture**. Google Gemini is now the default production provider (`gemini-2.5-flash` for reasoning/generation & NLI verification, `gemini-embedding-001` for vector embeddings). 

Deterministic providers remain fully intact for local testing, CI/CD, and offline execution, guaranteeing **100% backward compatibility** and **zero API breaking changes**.

All **358 backend unit/integration tests** pass cleanly (0 failures), and mypy type checking reports **0 errors across 102 source files**.

---

## 2. Architecture Changes & File Structure

A clean, modular `app/providers/` package was introduced following clean architecture and SOLID principles:

```
app/
  providers/
    __init__.py
    base/
      embedding_provider.py      # BaseEmbeddingProvider(BaseEmbedder)
      llm_provider.py            # BaseLLMProvider(BaseResponseGenerator, BaseNLIVerifier)
      reranker_provider.py       # BaseRerankerProvider(BaseReranker)
    deterministic/
      deterministic_embedding.py # DeterministicEmbeddingProvider
      deterministic_llm.py       # DeterministicLLMProvider
      deterministic_reranker.py  # DeterministicRerankerProvider
    gemini/
      gemini_embedding.py        # GeminiEmbeddingProvider (gemini-embedding-001)
      gemini_llm.py              # GeminiLLMProvider (gemini-2.5-flash)
    factory.py                   # AIProviderFactory
```

---

## 3. Provider Layer Features

### Google Gemini Production Provider (`app/providers/gemini/`)
- **Embeddings (`GeminiEmbeddingProvider`)**:
  - Model: `gemini-embedding-001`
  - Endpoints: `embedContent` / `batchEmbedContents`
  - Retries & Backoff: Up to 3 attempts with exponential backoff (1s, 2s, 4s).
  - Resilience: Automatically catches HTTP 429 rate limits, timeouts, and API unavailability, falling back gracefully without application crashes.
- **LLM Reasoning & NLI (`GeminiLLMProvider`)**:
  - Model: `gemini-2.5-flash`
  - Response Generation: Formulates grounded natural-language answers matching decision actions (`PROCEED`, `LOW_CONFIDENCE_RESPONSE`, `CLARIFY`, `HUMAN_REVIEW`).
  - NLI Pairwise Verification: Prompts Gemini for JSON relation classification (`ENTAILMENT`, `CONTRADICTION`, `NEUTRAL`) and confidence score.
  - Resilience: Retries on 429/timeouts with exponential backoff.

### Factory & Dynamic Resolution (`app/providers/factory.py`)
- `AIProviderFactory` dynamically resolves providers based on `AISettings.provider` (`"gemini"` vs `"deterministic"`).
- Connected to FastAPI Dependency Injection ([dependencies.py](file:///c:/Users/ROHIT/Downloads/sentinelrag/backend/app/api/dependencies.py)).

---

## 4. Cloud Infrastructure Hardening

- **Qdrant Cloud**: Extended `StorageSettings` with `STORAGE__QDRANT_API_KEY`. `create_qdrant_client()` passes `api_key` to `AsyncQdrantClient`, enabling seamless connection to `https://xxxxx.cloud.qdrant.io`.
- **Neon PostgreSQL**: Full support for `postgresql+asyncpg://...` DSNs with `sslmode=require`.
- **Upstash Redis**: Full support for `rediss://...` URLs with automatic TLS/SSL.

---

## 5. Environment Variables Reference (Render / Vercel)

| Variable | Recommended Value | Description |
| :--- | :--- | :--- |
| `AI_PROVIDER` | `gemini` | AI Provider selection (`gemini` or `deterministic`) |
| `GEMINI_API_KEY` | `AIzaSy...` | Google Gemini API Key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Reasoning & NLI Model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding Model |
| `STORAGE__QDRANT_URL` | `https://xxxxx.cloud.qdrant.io:6333` | Qdrant Vector DB Endpoint |
| `STORAGE__QDRANT_API_KEY` | `qdrant_key_...` | Qdrant Cloud API Key |
| `STORAGE__POSTGRES_DSN` | `postgresql+asyncpg://...` | Neon PostgreSQL DSN |
| `STORAGE__REDIS_URL` | `rediss://...` | Upstash Redis TLS URL |

---

## 6. Verification Results

- ✅ **Pytest Suite**: 358 / 358 passed (100% clean, 2.27s execution).
- ✅ **Mypy Type Checking**: 0 errors across 102 source files.
- ✅ **Backward Compatibility**: Deterministic mode preserves all historical test contracts and offline execution capabilities.
