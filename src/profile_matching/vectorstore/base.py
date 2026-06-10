"""Vector store interface and shared data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

Metadata = dict[str, str | int | float | bool]


@dataclass
class VectorRecord:
    """A single upsertable record: id + embedding + document + metadata."""

    id: str
    embedding: np.ndarray
    document: str
    metadata: Metadata = field(default_factory=dict)


@dataclass
class SearchHit:
    """A single search result."""

    id: str
    document: str
    metadata: Metadata
    score: float  # cosine similarity in [-1, 1]; higher is better


class VectorStore(ABC):
    """Abstract vector store supporting upsert, similarity query and filtering."""

    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or update records (idempotent on id)."""

    @abstractmethod
    def query(
        self,
        embedding: np.ndarray,
        top_k: int,
        where: Metadata | None = None,
    ) -> list[SearchHit]:
        """Return the ``top_k`` nearest records, optionally metadata-filtered."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""

    @abstractmethod
    def reset(self) -> None:
        """Delete all records in the collection."""
