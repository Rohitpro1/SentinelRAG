from __future__ import annotations

from abc import ABC
from app.services.embedding.base import BaseEmbedder


class BaseEmbeddingProvider(BaseEmbedder, ABC):
    """
    Abstract Base Class for embedding providers in SentinelRAG.
    Inherits from BaseEmbedder contract.
    """
    pass
