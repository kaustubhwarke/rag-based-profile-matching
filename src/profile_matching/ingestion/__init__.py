"""Document ingestion: filesystem loaders and section-aware chunking."""

from __future__ import annotations

from profile_matching.ingestion.chunking import SectionAwareChunker
from profile_matching.ingestion.loaders import LoadedDocument, load_document, load_documents

__all__ = [
    "LoadedDocument",
    "load_document",
    "load_documents",
    "SectionAwareChunker",
]
