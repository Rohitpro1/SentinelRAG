# Unit 3.2 — PlannerNode: First Executable LangGraph Node

**Status:** Complete. 20 new tests passing, 268/268 total (249 from Unit
3.1 + 20 new − 1 test removed/renamed as part of correctly updating Unit
3.1's test suite for PlannerNode's transition from placeholder to real
implementation — see "Test maintenance" below).

## What this unit delivers
- `app/core/settings/planner.py` — `PlannerSettings` (`min_words_threshold`,
  `multi_part_question_mark_threshold`), registered in the composition root.
- `app/schemas/planning.py` — `QueryClassification` (5 values:
  `EMPTY`, `TOO_SHORT`, `QUESTION`, `STATEMENT`, `MULTI_PART`) and
  `PlanningMetadata` (`normalized_query`, `classification`, `word_count`,
  `character_count`).
- `app/orchestration/graph_state.py` — gained `planning_metadata: Optional[PlanningMetadata] = None`,
  additive to Unit 3.1's seven fields.
- `app/orchestration/nodes/planner.py` — `PlannerNode` now genuinely
  implemented: normalizes, classifies, populates metadata, leaves
  `rewritten_query` untouched.
- `app/orchestration/graph_builder.py` — now compiles
  `START → planner → END`, the graph's first real expansion beyond
  Unit 3.1's minimal skeleton.

## Key engineering decisions

**1. `GraphState` was extended, not redesigned.** One new optional field
with a default — every Unit 3.1 test that constructs a `GraphState`
without `planning_metadata` still passes unchanged (verified: reran
`test_graph_state.py`, `test_graph_builder.py`, `test_orchestration_nodes.py`
before writing a single new test, confirming the extension was safe first).

**2. Classification is a fixed, explainable rule cascade** — empty check,
then word-count threshold, then question-mark count, then
question-word/mark check, else statement. Every branch is one `if`
away from the input; there's no scoring, weighting, or hidden state.
This is what "simple deterministic rules" (instruction 3) means in
practice, not an approximation of it.

**3. Normalization preserves case; classification lowercases only
internally.** `_normalize()` collapses whitespace but doesn't touch
case — the stored `normalized_query` keeps the user's original casing
(useful for proper nouns once this feeds retrieval later); only the
first-word question-word check lowercases for comparison.
`test_normalization_preserves_original_casing` asserts this directly.

**4. Whitespace-only input classifies as `EMPTY`, not `TOO_SHORT`.**
`"   \n\t  "` normalizes to `""` before classification runs, so it hits
the empty check first rather than being treated as a one-"word" short
query. `test_whitespace_only_string_classifies_as_empty` covers this
edge case specifically because it's the kind of off-by-one a rule
cascade can silently get wrong.

**5. `PlannerNode` never clobbers a pre-existing `rewritten_query`.**
Not explicitly requested, but a reasonable defensive property given a
future retry pass might call `PlannerNode` again on a state that already
has one set — `test_rewritten_query_untouched_if_already_set` locks this in.

**6. Thresholds are settings, the question-word lexicon is a constant** —
consistent with every prior settings class in this codebase never
externalizing a word list via env var either (documented directly in
`PlannerSettings`'s docstring, not left as an unexplained asymmetry).

## Test maintenance on Unit 3.1's own suite (disclosed, not hidden)
Wiring `PlannerNode` into `GraphBuilder.build()` made two things in Unit
3.1's test files inaccurate, both fixed here:
- `test_orchestration_nodes.py`'s parametrized "raises NotImplementedError"
  test included `PlannerNode`, which is no longer true. Split into
  `_ALL_NODES` (subclass-shape checks) and `_STILL_PLACEHOLDER_NODES`
  (Retrieval/Verification/Decision only, for the NotImplementedError check).
- `test_graph_builder.py`'s `test_constructor_accepts_injected_nodes_without_using_them_yet`
  claimed none of the injected nodes are used — no longer true for
  `planner_node`. Renamed and re-scoped to cover specifically the three
  nodes that are still unused (retrieval/verification/decision).

Both changes are behavior-neutral for everything they still assert; only
the parts of their names/docstrings that became factually wrong were
touched.

## Independence verified, not assumed
`grep -nE "app\.services\.(retrieval|verification|decision_engine)" app/orchestration/nodes/planner.py`
returns only the docstring's own prose mentioning those module paths —
zero actual imports. `PlannerNode` reads only `state.original_query` and
writes only `state.planning_metadata`.

## What this unit deliberately does NOT do
- No query rewriting — `rewritten_query` stays exactly as it was.
- No LLM call anywhere in `PlannerNode`.
- `RetrievalNode`/`VerificationNode`/`DecisionNode` remain
  `NotImplementedError` placeholders, unwired in the graph.

## Next unit (anticipated, not yet scoped)
Likely candidates: `RetrievalNode` wrapping `RetrieverAgent` (most
existing Milestone 2 machinery to reuse), or beginning query-rewrite
logic for `PlannerNode` — whichever the next instruction specifies.
