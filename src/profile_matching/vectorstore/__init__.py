"""Vector store backends and factory."""

from __future__ import annotations

from profile_matching.vectorstore.base import SearchHit, VectorRecord, VectorStore
from profile_matching.vectorstore.factory import build_vector_store

__all__ = ["VectorStore", "VectorRecord", "SearchHit", "build_vector_store"]
