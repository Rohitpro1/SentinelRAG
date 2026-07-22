"""
Unit 2.9 -- DeterministicNLIVerifier.

A valid, first-class BaseNLIVerifier implementation (same treatment as
DeterministicEmbedder / DeterministicReranker) -- deterministic-first per
the Unit 2.9 review instruction: this ships BEFORE any real NLI model
integration, so ContradictionDetector and everything above it can be
built and tested now.

Detection rule: two texts are flagged CONTRADICTION only if both contain
the configured conflict_marker substring (default "[CONTRADICTION]") --
a deliberate, explicit test/dev marker, not a heuristic that tries to
guess semantic conflict from arbitrary text. This keeps behavior
100% deterministic and easy to reason about in tests: a test wanting to
exercise the contradiction path plants the marker in two chunk texts;
a test wanting to exercise the non-contradiction path does not.

Real semantic NLI (e.g. a cross-encoder trained on MNLI/SNLI) is a future
sibling implementation behind the same BaseNLIVerifier interface.
"""
from __future__ import annotations

import hashlib

from app.schemas.embedding import EmbedderHealth, EmbedderHealthState
from app.schemas.nli import NLIResult
from app.schemas.retrieval import NLIRelation
from app.services.verification.nli_base import BaseNLIVerifier
from app.services.verification.nli_result_builder import build_nli_result


class DeterministicNLIVerifier(BaseNLIVerifier):
    def __init__(self, conflict_marker: str = "[CONTRADICTION]", entailment_threshold: float = 0.7):
        self._conflict_marker = conflict_marker
        self._entailment_threshold = entailment_threshold

    async def verify_pair(self, text_a: str, text_b: str) -> tuple[NLIRelation, float]:
        confidence = self._deterministic_confidence(text_a, text_b)

        if self._conflict_marker in text_a and self._conflict_marker in text_b:
            return NLIRelation.CONTRADICTION, confidence

        if confidence >= self._entailment_threshold:
            return NLIRelation.ENTAILMENT, confidence

        return NLIRelation.NEUTRAL, confidence

    async def verify_pair_with_result(self, text_a: str, text_b: str) -> NLIResult:
        """
        Unit 2.13 -- additive capability, not part of BaseNLIVerifier.
        Deterministic-first: proves NLIResult's shape here before any
        real provider exists, same principle as Units 2.11/2.12.
        """
        return await build_nli_result(
            lambda: self.verify_pair(text_a, text_b),
            provider="deterministic",
            model_name="deterministic-sha256-marker",
            model_version="v1",
        )

    def health(self) -> EmbedderHealth:
        """No external dependency, cannot fail -- always READY, reusing EmbedderHealth (instruction 2)."""
        return EmbedderHealth(state=EmbedderHealthState.READY)

    @staticmethod
    def _deterministic_confidence(text_a: str, text_b: str) -> float:
        # Order-independent so verify_pair(a, b) and verify_pair(b, a) agree.
        key = ":".join(sorted([text_a, text_b]))
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") / (2**32)
