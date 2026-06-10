"""In-memory vector store.

A NumPy-backed brute-force store used as a dependency-free fallback and for
tests/CI. Exact (not approximate) nearest neighbour — correctness over speed,
which is ideal for the modest resume corpus sizes in this assignment.
"""

from __future__ import annotations

import numpy as np

from profile_matching.vectorstore.base import Metadata, SearchHit, VectorRecord, VectorStore


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[Metadata] = []
        self._matrix: np.ndarray | None = None
        self._index: dict[str, int] = {}

    def upsert(self, records: list[VectorRecord]) -> None:
        for rec in records:
            vec = np.asarray(rec.embedding, dtype=np.float32).reshape(-1)
            if rec.id in self._index:
                pos = self._index[rec.id]
                self._docs[pos] = rec.document
                self._metas[pos] = rec.metadata
                self._matrix[pos] = vec  # type: ignore[index]
            else:
                self._index[rec.id] = len(self._ids)
                self._ids.append(rec.id)
                self._docs.append(rec.document)
                self._metas.append(rec.metadata)
                self._matrix = (
                    vec[None, :]
                    if self._matrix is None
                    else np.vstack([self._matrix, vec])
                )

    def query(
        self,
        embedding: np.ndarray,
        top_k: int,
        where: Metadata | None = None,
    ) -> list[SearchHit]:
        if self._matrix is None or not self._ids:
            return []

        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        sims = self._matrix @ q  # vectors are pre-normalised -> cosine similarity

        candidate_idx = range(len(self._ids))
        if where:
            candidate_idx = [i for i in candidate_idx if self._matches(self._metas[i], where)]
            if not candidate_idx:
                return []

        candidate_idx = list(candidate_idx)
        scored = sorted(candidate_idx, key=lambda i: sims[i], reverse=True)[:top_k]
        return [
            SearchHit(
                id=self._ids[i],
                document=self._docs[i],
                metadata=self._metas[i],
                score=float(sims[i]),
            )
            for i in scored
        ]

    @staticmethod
    def _matches(meta: Metadata, where: Metadata) -> bool:
        return all(meta.get(k) == v for k, v in where.items())

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self.__init__()
