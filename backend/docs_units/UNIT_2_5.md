# Unit 2.5 — BaseReranker Interface + DeterministicReranker

**Status:** Complete. 7/7 new tests passing, 73/73 total at time of writing
(prior to Unit 2.6's additions).

## What this unit delivers
- `app/services/reranking/base.py` — `BaseReranker` ABC: single method
  `rerank(query, candidates, top_n) -> list[RankedChunk]`.
- `app/services/reranking/deterministic.py` — `DeterministicReranker`, named
  and treated the same way as `DeterministicEmbedder` (Unit 2.4, renamed
  from `FakeEmbedder` per this same review cycle): a valid, first-class
  `BaseReranker` implementation, not a test-only stub.

## Key implementation decision
Scoring combines the original `similarity_score` with a small
(±0.05) SHA-256-derived deterministic perturbation, so reranked order can
plausibly differ from similarity order — proving the rerank stage actually
changes something — without being unbounded or semantically arbitrary.
Not a real cross-encoder; explicitly documented as such.

## Naming convention going forward
Per the review instruction to treat deterministic components as real
implementations: no new component in this domain will be named `Fake*`.
`DeterministicEmbedder` and `DeterministicReranker` are both named and
documented as legitimate choices for environments without real model
access, with real implementations (`CrossEncoderReranker`,
`SentenceTransformerEmbedder`, etc.) documented as future siblings behind
the same interface.
