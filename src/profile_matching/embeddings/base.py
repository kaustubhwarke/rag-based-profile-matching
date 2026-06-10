"""Embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base for all embedding backends.

    Implementations must return L2-comparable float vectors. The contract is
    deliberately minimal so providers (local models, managed APIs, fallbacks)
    are fully interchangeable behind the factory.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced embedding vectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider/model identifier (for logging & metadata)."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents -> array of shape (len(texts), dimension)."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string -> vector of shape (dimension,)."""

        return self.embed_documents([text])[0]
