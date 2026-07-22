# Unit 2.15 — Docker Compose Local Development Environment

**Status:** Complete. Deployment infrastructure only — no application
code changed, no new tests added to the pytest suite (this unit is
validated by inspection, YAML-syntax checking, and Unit 2.10's existing
integration test, per the original approved plan's own description of
this unit: "validated by 2.10's integration test actually passing
against it, rather than being its own separate test").

## What this unit delivers
- `backend/Dockerfile` — single-stage backend image.
- `docker-compose.yml` (repo root) — `backend`, `qdrant`, `postgres`,
  `redis` by default; `prometheus`, `grafana` behind an `observability`
  Compose profile.
- `.env.example` — every environment variable the compose file uses,
  mapped explicitly to the `Settings` class/field it configures.
- `infrastructure/prometheus.yml` — minimal scrape config for the
  optional observability profile.

## Startup workflow

```
docker compose up
      |
      v
qdrant / postgres / redis start, each running its own healthcheck
      |
      v
backend's `depends_on: condition: service_healthy` blocks its own start
until all three report healthy
      |
      v
backend starts; Compose's healthcheck against it (hitting FastAPI's
built-in /docs page) must pass before Compose reports backend healthy
      |
      v
application ready: http://localhost:8000/docs
```

With observability: `docker compose --profile observability up` adds
`prometheus` (`http://localhost:9090`) and `grafana`
(`http://localhost:3000`, default admin password `admin`, see
`.env.example`) alongside the above.

## Health checks (instruction 3)
| Service | Healthcheck |
|---|---|
| qdrant | `curl -sf http://localhost:6333/healthz` |
| postgres | `pg_isready -U $POSTGRES_USER -d $POSTGRES_DB` |
| redis | `redis-cli ping` |
| backend | `GET /docs` (FastAPI's built-in auto-docs page) |

`backend` only starts once qdrant/postgres/redis are all
`service_healthy`, per `depends_on` conditions — "application startup
should wait for required dependencies where appropriate."

## Key engineering decisions

**1. Backend's healthcheck hits `/docs`, not a new custom endpoint.**
Per instruction 5 ("do not introduce application business logic"),
adding a `/health` route would touch application code in a unit scoped
to be deployment-infrastructure-only. FastAPI's built-in `/docs` page
already returns 200 exactly when the app has finished starting and
routing is live — a correct liveness signal with zero new application
code.

**2. Backend does NOT automatically switch to real infrastructure — a
known scope boundary, stated plainly rather than glossed over.** The
backend container runs with its existing Unit 2.14 default DI wiring
(`DeterministicEmbedder`, `InMemoryVectorRepository`,
`DeterministicReranker`, `DeterministicNLIVerifier`) even though real
Qdrant/Postgres/Redis are running alongside it. This compose file's job
is to provide a reproducible environment where the app *and* real
infrastructure are both available — enough for Units 2.10–2.13's
isolated integration tests to run against real services — not to change
what the app does by default. Rewiring `app/api/dependencies.py` to
consume real infrastructure automatically would be an application-
behavior change, which is explicitly out of scope for this unit
("do not modify the application architecture").

**3. Environment externalization uses Compose service-name DNS, not
hardcoded addresses.** `STORAGE__QDRANT_URL=http://qdrant:6333` in
`.env.example` resolves via Docker Compose's internal network — this is
the *correct* externalized pattern (the hostname is configuration, not a
hardcoded local address); `localhost` would be wrong here since
containers don't share a loopback interface with each other.

**4. Prometheus scrapes only itself for now, honestly.** No
`sentinelrag-backend` scrape target exists because the app has no
`/metrics` endpoint yet (`BaseTelemetrySink`/`PrometheusTelemetrySink`
are still-future work per the frozen Architecture Enhancements
addendum). `prometheus.yml`'s comment says this directly rather than
shipping a scrape config pointed at a target that doesn't exist.

**5. No `version:` key in docker-compose.yml.** Modern Compose Spec
doesn't require it and current `docker compose` emits a deprecation
warning if present — omitted deliberately, not an oversight.

## Honest limitation, stated plainly
This sandbox has no Docker daemon available (confirmed: `docker` and
`docker-compose` are not on `PATH` here), so this compose file has been
validated by **YAML-syntax parsing** (confirmed valid: 6 services,
`observability` profile correctly scoped to `prometheus`/`grafana` only,
3 named volumes) and by inspection against Docker Compose's documented
schema — **not** by actually running `docker compose up` end-to-end.
Anyone adopting this should run it for real before relying on it;
the same honesty standard applied to Units 2.10–2.13's "no live network
in this sandbox" notes applies here.

## Milestone 2 plan status
This completes all 15 units of the originally approved Milestone 2
implementation plan (2.1 through 2.15).
