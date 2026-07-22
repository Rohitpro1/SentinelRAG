from __future__ import annotations

from abc import ABC
from app.services.reranking.base import BaseReranker


class BaseRerankerProvider(BaseReranker, ABC):
    """
    Abstract Base Class for reranker providers in SentinelRAG.
    Inherits from BaseReranker contract.
    """
    pass
