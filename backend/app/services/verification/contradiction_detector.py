"""
Unit 2.9 -- ContradictionDetector.

Owns: running pairwise NLI across validated evidence via an injected
BaseNLIVerifier. Runs on EvidenceValidator's output, never on raw
ranked_chunks -- structurally invalid evidence (Empty text etc.) has
nothing meaningful to compare.

Unit 2.13 addition: per instruction 5 ("if an NLI provider becomes
unavailable, degrade gracefully where possible and expose the condition
through structured logging and telemetry rather than silently masking
it"), a failing verify_pair() call for one pair no longer aborts the
entire detection pass. This closes a real gap -- prior to this unit,
ANY verify_pair() exception (guaranteed with a real, fallible NLI
provider; impossible with the always-succeeding DeterministicNLIVerifier,
which is exactly why this gap wasn't visible until a real provider was
introduced) would propagate uncaught through VerificationAgent.verify()
and fail the whole request. Degradation choice: a failed pair becomes
(NEUTRAL, 0.0) -- "no signal obtained," not a false claim of either
entailment or contradiction -- logged as a WARNING with which chunk pair
and error, never silently dropped from the results list.

Constructor gained an optional `logger` parameter for this -- backward
compatible: every existing call site (`ContradictionDetector(verifier)`)
continues to work unchanged.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.core.logging import get_logger, log_event
from app.schemas.retrieval import NLIRelation, PairwiseNLIResult
from app.schemas.retrieval_domain import RankedChunk
from app.services.verification.nli_base import BaseNLIVerifier


class ContradictionDetector:
    def __init__(self, nli_verifier: BaseNLIVerifier, logger: Optional[logging.Logger] = None):
        self._nli_verifier = nli_verifier
        self._logger = logger or get_logger(__name__)

    async def detect(self, valid_evidence: list[RankedChunk]) -> list[PairwiseNLIResult]:
        results: list[PairwiseNLIResult] = []
        chunks = [rc.retrieved_chunk.chunk for rc in valid_evidence]

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                relation, confidence = await self._verify_pair_with_degradation(chunks[i].text, chunks[j].text, chunks[i].chunk_id, chunks[j].chunk_id)
                results.append(
                    PairwiseNLIResult(
                        chunk_id_a=chunks[i].chunk_id, chunk_id_b=chunks[j].chunk_id,
                        relation=relation, confidence=confidence,
                    )
                )
        return results

    async def _verify_pair_with_degradation(
        self, text_a: str, text_b: str, chunk_id_a: str, chunk_id_b: str
    ) -> tuple[NLIRelation, float]:
        try:
            return await self._nli_verifier.verify_pair(text_a, text_b)
        except Exception as exc:  # noqa: BLE001 -- intentional: this must never propagate
            log_event(
                self._logger, "nli_pair_degraded", level=logging.WARNING,
                chunk_id_a=chunk_id_a, chunk_id_b=chunk_id_b, error=str(exc), error_type=type(exc).__name__,
            )
            return NLIRelation.NEUTRAL, 0.0
