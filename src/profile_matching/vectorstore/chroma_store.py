"""ChromaDB-backed persistent vector store.

Wraps a persistent Chroma collection. Embeddings are computed by our own
provider abstraction and passed in explicitly (Chroma is used purely as the
ANN index + metadata store), keeping the embedding backend swappable.

Chroma returns squared-L2 distances for normalised vectors; we convert these
back to cosine similarity so scores are consistent across all stores:
    for unit vectors,  ||a-b||^2 = 2 - 2*cos  =>  cos = 1 - dist/2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from profile_matching.logging_config import get_logger
from profile_matching.vectorstore.base import Metadata, SearchHit, VectorRecord, VectorStore

logger = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        import chromadb  # lazy import
        from chromadb.config import Settings as ChromaSettings

        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma collection '%s' ready at %s (%d vectors)",
            collection_name,
            persist_directory,
            self._collection.count(),
        )

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._collection.upsert(
            ids=[r.id for r in records],
            embeddings=[np.asarray(r.embedding, dtype=np.float32).tolist() for r in records],
            documents=[r.document for r in records],
            metadatas=[r.metadata for r in records],
        )

    def query(
        self,
        embedding: np.ndarray,
        top_k: int,
        where: Metadata | None = None,
    ) -> list[SearchHit]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=min(top_k, self._collection.count()),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        return [
            SearchHit(
                id=_id,
                document=doc,
                metadata=meta,
                score=1.0 - (float(dist) / 2.0),  # squared-L2 -> cosine
            )
            for _id, doc, meta, dist in zip(ids, docs, metas, dists, strict=False)
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
