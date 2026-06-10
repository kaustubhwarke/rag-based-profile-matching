"""Embedding providers and factory."""

from __future__ import annotations

from profile_matching.embeddings.base import EmbeddingProvider
from profile_matching.embeddings.factory import build_embedding_provider

__all__ = ["EmbeddingProvider", "build_embedding_provider"]
