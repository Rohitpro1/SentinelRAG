"""
Unit 3.2 -- PlannerNode: the first executable LangGraph node.
Unit 3.7 -- extended to be retry-aware: generates a deterministic
rewritten_query when retry_count > 0.

Scope, per explicit instruction: normalize the query, classify it with
simple deterministic rules, populate planning_metadata. On the FIRST
execution (retry_count == 0), rewritten_query stays unset. On a RETRY
(retry_count > 0), a deterministic stopword-stripping rewrite rule
generates rewritten_query, and original_query is never touched. NO
LLM-based rewriting -- explicitly out of scope, reserved for a later unit.

Independence (instruction 4, Unit 3.2 -- still holds): this file imports
nothing from app.services.retrieval, app.services.verification, or
app.services.decision_engine. PlannerNode only ever reads
state.original_query and state.retry_count, and writes
state.planning_metadata and (conditionally) state.rewritten_query.

For this rewrite to ever run, the graph's retry loop must route back
THROUGH this node before re-entering RetrievalNode (Unit 3.7 changes
GraphBuilder's retry_increment target from "retrieval" to "planner" --
see graph_builder.py). RetrievalNode itself is unchanged: it still only
ever reads GraphState.effective_query (Unit 3.1's property), which
already prefers rewritten_query when present -- RetrievalNode has no
idea a rewrite even happened.
"""
from __future__ import annotations

import string
from typing import Optional

from app.core.settings.planner import PlannerSettings
from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.base import BaseGraphNode
from app.schemas.planning import PlanningMetadata, QueryClassification

# Deliberately module-level constants, not settings fields -- see
# PlannerSettings' docstring for why word lists aren't externalized as
# env vars in this project's existing convention.
_QUESTION_WORDS = frozenset(
    {"what", "who", "whom", "whose", "when", "where", "why", "how",
     "is", "are", "was", "were", "can", "could", "does", "do", "did",
     "will", "would", "should", "shall", "may", "might"}
)

# Unit 3.7 -- stopwords stripped when generating a rewritten_query on
# retry. Deliberately broader than _QUESTION_WORDS (includes articles,
# prepositions, pronouns, conjunctions) since the goal here is reducing a
# natural-language question down to its core keyword content -- a
# classic, simple, fully deterministic IR technique, not a heuristic
# trying to guess "better" phrasing.
_STOPWORDS = frozenset(
    {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
     "what", "who", "whom", "when", "where", "why", "how", "do", "does", "did",
     "of", "in", "on", "to", "for", "and", "or", "but", "please",
     "could", "would", "should", "can", "will", "shall",
     "this", "that", "these", "those", "with", "at", "by", "from", "about",
     "as", "into", "like", "through", "after", "before", "between",
     "i", "you", "he", "she", "it", "we", "they",
     "me", "my", "your", "his", "her", "its", "our", "their"}
)


def _normalize(text: str) -> str:
    """Trim and collapse all internal whitespace (spaces, tabs, newlines) to single spaces. Preserves case."""
    return " ".join(text.split())


def _rewrite_query(normalized_query: str) -> str:
    """
    Unit 3.7 -- deterministic query rewrite for retries: strips common
    stopwords, keeping only keyword-bearing words, in their original
    order and casing. If stripping removes every word (e.g. the query
    was entirely stopwords), falls back to the normalized query
    unchanged rather than producing an empty/unusable rewrite.

    Deliberately the SAME rewrite regardless of which retry attempt this
    is (idempotent: rewriting an already-rewritten query produces the
    same result) -- simplicity was chosen over a progressively-broadening
    rewrite across multiple retries, since the latter wasn't requested
    and would add complexity/behavior not asked for. Documented here so
    it reads as a deliberate choice, not an oversight.
    """
    words = normalized_query.split()
    keywords = [w for w in words if w.strip(string.punctuation).lower() not in _STOPWORDS]
    if not keywords:
        return normalized_query
    return " ".join(keywords)


class PlannerNode(BaseGraphNode):
    def __init__(self, settings: Optional[PlannerSettings] = None):
        self._settings = settings or PlannerSettings()

    async def __call__(self, state: GraphState) -> GraphState:
        normalized = _normalize(state.original_query)
        classification = self._classify(normalized)
        metadata = PlanningMetadata(
            normalized_query=normalized,
            classification=classification,
            word_count=len(normalized.split()) if normalized else 0,
            character_count=len(normalized),
        )

        updates: dict = {"planning_metadata": metadata}
        if state.retry_count > 0:
            # Unit 3.7: retry-aware rewrite. Always derived fresh from
            # original_query's normalized form (never from a possibly-
            # already-rewritten state) -- keeps the rewrite predictable:
            # "what would rewriting produce for this original query" has
            # exactly one answer, independent of how many retries have
            # already happened. original_query itself is never modified.
            updates["rewritten_query"] = _rewrite_query(normalized)
        # else (first execution, retry_count == 0): rewritten_query is
        # intentionally left out of `updates` -- it stays whatever it
        # already was (None, on a genuine first pass), per instruction 2.

        return state.model_copy(update=updates)

    def _classify(self, normalized: str) -> QueryClassification:
        s = self._settings

        if not normalized:
            return QueryClassification.EMPTY

        word_count = len(normalized.split())
        if word_count < s.min_words_threshold:
            return QueryClassification.TOO_SHORT

        if normalized.count("?") >= s.multi_part_question_mark_threshold:
            return QueryClassification.MULTI_PART

        first_word = normalized.split()[0].lower().strip(",.:;!?")
        if normalized.endswith("?") or first_word in _QUESTION_WORDS:
            return QueryClassification.QUESTION

        return QueryClassification.STATEMENT
