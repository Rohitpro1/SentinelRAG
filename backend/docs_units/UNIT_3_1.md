# Unit 3.1 — LangGraph Foundation

**Status:** Complete. 25/25 new tests passing, 249/249 total. Zero
Milestone 2 files modified (verified by file-modification-time check).

## What this unit delivers
A new, isolated `app/orchestration/` package:
- `graph_state.py` — `GraphState`, a Pydantic model carrying exactly the
  seven fields specified: `original_query`, `rewritten_query`,
  `retrieval_result`, `verification_result`, `decision`, `retry_count`,
  `diagnostics`. Built entirely from existing Milestone 2 schemas
  (`SearchResponse`, `VerifiedEvidence`, `Decision`, `VerificationDiagnostics`)
  — zero new business types introduced.
- `nodes/base.py` — `BaseGraphNode`, the uniform interface every node
  implements (`async def __call__(state) -> state`).
- `nodes/{planner,retrieval,verification,decision}.py` — four concrete
  placeholder classes, each raising `NotImplementedError`, mirroring the
  `TableChunker`/`SemanticChunker` pattern from Unit 1.3: the interface
  shape exists, the logic doesn't yet, and calling it fails loudly rather
  than silently no-op-ing.
- `graph_builder.py` — `GraphBuilder`, compiling the minimal
  `START -> END` graph, with constructor DI already accepting all four
  node instances (unused in this unit's `build()`, but present so the
  constructor signature won't need to change later).

## Key engineering decisions

**1. `GraphState` reuses domain schemas rather than redefining them.**
`retrieval_result` is a real `SearchResponse`, `verification_result` a
real `VerifiedEvidence`, `decision` a real `Decision` — not parallel
graph-specific types. This means nothing about Milestone 2's schemas had
to change, and a future node wrapping (say) `RetrieverAgent` writes its
real `SearchResponse` directly into `state.retrieval_result` with no
translation layer.

**2. A behavioral quirk was discovered and is now tested, not just
noted.** `CompiledStateGraph.ainvoke()` on a Pydantic-schema graph
returns a plain `dict` that **omits unset/`None`-valued fields entirely**
— confirmed by direct inspection, not assumed. `GraphState(**result)`
still reconstructs correctly because every optional field has a default.
`test_reconstructing_from_partial_dict_fills_defaults` locks this in
explicitly, since it's the kind of surprising framework behavior that
would otherwise cause a confusing failure the first time a future unit
tries to read a "missing" field from a graph's output.

**3. `effective_query` is a computed property, not a stored field.**
`rewritten_query or original_query` — this is the exact query a future
`RetrievalNode` should use, decided once, in one place, rather than every
future node re-implementing the same fallback logic.

**4. Node placeholders raise, they don't pass through.** A node that
silently returned `state` unchanged would be indistinguishable from a
correctly-implemented no-op node in a test — raising `NotImplementedError`
makes "this isn't built yet" impossible to mistake for "this is built and
does nothing," consistent with the project's established pattern for
unimplemented interfaces (`TableChunker`, `SemanticChunker`).

**5. `GraphBuilder`'s constructor already takes all four nodes.** Unused
today, but this means Unit 3.2 (or whichever unit wires the Planner node
in first) changes only `build()`'s body — adding
`graph.add_node("planner", self._planner_node)` and an edge — not the
class's public shape. Verified directly:
`test_constructor_accepts_injected_nodes_without_using_them_yet` passes
real node instances through construction today.

## What this unit deliberately does NOT do
- No node contains business logic (instruction 4 is explicit about this).
- No wiring to `RetrieverAgent`/`VerificationAgent`/`DecisionEngine`
  exists yet — that's for the unit(s) that replace `QueryService`'s
  current `while` loop (Unit 2.14) with LangGraph's conditional-edge
  routing.
- No checkpointing/persistence — `GraphBuilder.build()` compiles without
  a checkpointer, so state does not survive between separate `ainvoke()`
  calls. Adding one (e.g. LangGraph's `MemorySaver` or a Postgres-backed
  checkpointer, tying into `StorageSettings.postgres_dsn`, already
  defined) is future work, not silently assumed to already work.

## Next unit (anticipated, not yet scoped)
Likely candidates: wiring `RetrievalNode` to `RetrieverAgent` first (the
node with the most already-built Milestone 2 machinery behind it), or
building out `DecisionNode`'s conditional routing to finally replace
`QueryService`'s retry loop with a native LangGraph edge — whichever the
next instruction specifies.
