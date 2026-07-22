# Unit 3.9 — Response Generation & Reasoning Layer

**Status:** Complete. 14 new tests passing, 353/353 total (339 from Unit 3.8 + 14 new). Milestone 1 contracts and Milestone 2 business logic remain 100% frozen.

## What this unit delivers
- `app/schemas/query.py` — added `answer: Optional[str] = None` to `QueryResult` domain return model.
- `app/api/v1/schemas.py` — added `answer: Optional[str] = None` to `QueryResponseBody` DTO and mapped `result.answer` in `from_query_result()`.
- `app/services/response_generation/base.py` — `BaseResponseGenerator` abstract interface defining answer generation signature.
- `app/services/response_generation/service.py` — `ResponseGenerator` concrete deterministic service implementation supporting `PROCEED`, `LOW_CONFIDENCE_RESPONSE`, `CLARIFY`, and `HUMAN_REVIEW` actions.
- `app/orchestration/graph_state.py` — added `answer: Optional[str] = None` to carry answer state across graph nodes.
- `app/orchestration/nodes/response_generation.py` — `ResponseGenerationNode` thin LangGraph node wrapper delegating to `BaseResponseGenerator`.
- `app/orchestration/graph_builder.py` — wired `response_generation` node as the terminal node after `decision` node for non-retry paths before `END`.
- `app/services/query/query_service.py` — extracts `final_state.answer` and attaches it to `QueryResult`.
- `app/api/dependencies.py` — composition root factory `get_response_generator()` created and injected into `GraphBuilder`.

## Architecture & Design Decisions
1. **Decoupled Prompting & Answer Formatting**:
   - `ResponseGenerationNode` contains zero prompting or formatting logic. It extracts state variables (`decision`, `verification_result`, `diagnostics`, `effective_query`) and calls `BaseResponseGenerator.generate()`.
2. **Action-Specific Response Formatting**:
   - `PROCEED`: Formats grounded natural-language responses referencing verified evidence chunks.
   - `LOW_CONFIDENCE_RESPONSE`: Formats grounded responses with explicit low-confidence caveats.
   - `CLARIFY`: Formats structured user clarification requests based on decision reasons.
   - `HUMAN_REVIEW`: Formats human review escalation notices indicating contradiction or policy constraints.
3. **Deterministic First (No External LLM Dependency)**:
   - `ResponseGenerator` operates deterministically using evidence data and decision metadata, preserving test speed, reliability, and sandbox independence.
4. **Public API & Dependency Injection Integrity**:
   - `QueryService.handle_query()` signature remains completely unchanged.
   - All components are wired via constructor DI and `app/api/dependencies.py`.

## Test Coverage
1. `tests/test_response_generator.py`: 6 unit tests covering all 4 actions and fallback scenarios.
2. `tests/test_response_generation_node.py`: 2 unit tests covering graph state transformations.
3. `tests/test_unit_3_9_integration.py`: 6 integration tests verifying:
   - Normal answer (`PROCEED`)
   - Low confidence answer (`LOW_CONFIDENCE_RESPONSE`)
   - Clarification answer (`CLARIFY`)
   - Human review answer (`HUMAN_REVIEW`)
   - Retry completion
   - API & DTO compatibility

## Summary of Test Results
- **353 passed in 2.30s**.
