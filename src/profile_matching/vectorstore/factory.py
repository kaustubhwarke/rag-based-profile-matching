"""Vector store factory with graceful fallback to the in-memory backend."""

from __future__ import annotations

from profile_matching.config import VectorBackend, VectorStoreSettings
from profile_matching.logging_config import get_logger
from profile_matching.vectorstore.base import VectorStore
from profile_matching.vectorstore.memory_store import InMemoryVectorStore

logger = get_logger(__name__)


def build_vector_store(settings: VectorStoreSettings) -> VectorStore:
    """Instantiate the configured vector store; fall back to in-memory."""

    if settings.backend is VectorBackend.MEMORY:
        return InMemoryVectorStore()

    if settings.backend is VectorBackend.CHROMA:
        try:
            from profile_matching.vectorstore.chroma_store import ChromaVectorStore

            return ChromaVectorStore(settings.persist_directory, settings.collection_name)
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning(
                "ChromaDB unavailable (%s). Falling back to in-memory vector store "
                "(non-persistent).",
                exc,
            )
            return InMemoryVectorStore()

    raise ValueError(f"Unknown vector backend: {settings.backend}")
