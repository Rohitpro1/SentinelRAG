# Engineering priorities

The current architecture separates orchestration, providers, domain services, and repository interfaces. The next milestone is to turn those seams into verified runtime contracts.

1. Establish dependency installation and an offline test baseline on the supported Python version. Capture failures before altering behavior.
2. Replace header-presence authentication with validated credentials and explicit failure modes; test malformed, missing, and incorrect credentials. Define tenant isolation before claiming multi-user support.
3. Wire persistent repositories explicitly and prove upload-to-query continuity across process restart. Treat Compose service availability separately from application readiness.
4. Bound document size, queue depth, processing duration, and external provider retries; make failures observable to API clients.
5. Add CI for backend tests and frontend type/build checks. Document integration-test prerequisites separately from offline checks.
6. Measure retrieval quality against a versioned corpus and report methodology, abstention behavior, latency, and failure cases. Avoid performance or production-readiness claims without evidence.

Each change should include a reproducible failing case, a focused regression check, and documentation of remaining limitations.
