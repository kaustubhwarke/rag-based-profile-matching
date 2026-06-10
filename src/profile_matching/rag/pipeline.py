"""ResumeRAG — the Part A document-processing & indexing pipeline.

Responsibilities (mapping directly to the assignment's Part A requirements):

  Document Processing Pipeline
    * Load resumes from the filesystem                      -> ingestion.loaders
    * Chunk documents intelligently (preserve sections)     -> SectionAwareChunker
    * Generate embeddings (HuggingFace/OpenAI/Cohere/...)   -> embeddings.factory
    * Store in a vector database (ChromaDB / in-memory)     -> vectorstore.factory

  Metadata Extraction
    * Extract Name, Skills, Experience Years, Education     -> MetadataExtractor
    * Store metadata alongside embeddings for filtering     -> VectorRecord.metadata
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path

from profile_matching.config import Settings, get_settings
from profile_matching.embeddings import build_embedding_provider
from profile_matching.embeddings.base import EmbeddingProvider
from profile_matching.extraction import MetadataExtractor
from profile_matching.ingestion import SectionAwareChunker, load_document, load_documents
from profile_matching.logging_config import get_logger
from profile_matching.models.resume import Resume, ResumeChunk, SectionType
from profile_matching.vectorstore import SearchHit, VectorRecord, build_vector_store
from profile_matching.vectorstore.base import VectorStore

logger = get_logger(__name__)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "resume"


class ResumeRAG:
    """End-to-end resume ingestion + indexing + chunk-level retrieval.

    The constructor wires the configured embedding provider and vector store.
    Pass a custom ``embedder``/``store`` for testing or alternative backends.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or build_embedding_provider(self.settings.embedding)
        self.store = store or build_vector_store(self.settings.vectorstore)
        self.chunker = SectionAwareChunker(self.settings.chunking)
        self.extractor = MetadataExtractor()
        logger.info(
            "ResumeRAG initialised | embedder=%s | dim=%d", self.embedder.name, self.embedder.dimension
        )

    # ----------------------------------------------------------- processing
    def process_document(self, path: str | Path) -> Resume:
        """Load, chunk and extract metadata for a single resume (no indexing)."""

        doc = load_document(path)
        resume_id = _slugify(Path(path).stem)
        chunks = self.chunker.chunk(resume_id, doc.text)

        sections = self._sections_from_chunks(chunks)
        metadata = self.extractor.extract(doc.text, sections)
        # Prefer the parsed display name when filename is uninformative.
        if metadata.name == "Unknown":
            metadata.name = Path(path).stem.replace("_", " ").replace("-", " ").title()

        return Resume(
            resume_id=resume_id,
            source_path=str(path),
            raw_text=doc.text,
            metadata=metadata,
            chunks=chunks,
        )

    @staticmethod
    def _sections_from_chunks(chunks: list[ResumeChunk]) -> dict[SectionType, str]:
        grouped: dict[SectionType, list[str]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.section].append(chunk.text)
        return {section: "\n".join(parts) for section, parts in grouped.items()}

    # -------------------------------------------------------------- indexing
    def index_resume(self, resume: Resume) -> int:
        """Embed and upsert all chunks of a single resume. Returns chunk count."""

        if not resume.chunks:
            logger.warning("Resume %s produced no chunks; skipping", resume.resume_id)
            return 0

        texts = [c.embedding_text() for c in resume.chunks]
        embeddings = self.embedder.embed_documents(texts)
        base_meta = resume.metadata.as_store_metadata()

        records = [
            VectorRecord(
                id=chunk.chunk_id,
                embedding=embeddings[i],
                document=chunk.text,
                metadata={
                    **base_meta,
                    "resume_id": resume.resume_id,
                    "source_path": resume.source_path,
                    "section": chunk.section.value,
                    "section_heading": chunk.section_heading or "",
                    "chunk_order": chunk.order,
                },
            )
            for i, chunk in enumerate(resume.chunks)
        ]
        self.store.upsert(records)
        return len(records)

    def index_directory(self, directory: str | Path | None = None) -> dict:
        """Process and index every resume in a directory. Returns a summary."""

        directory = Path(directory or self.settings.paths.resume_dir)
        start = time.perf_counter()
        docs = load_documents(directory)

        resumes_indexed = 0
        chunks_indexed = 0
        for doc in docs:
            resume = self.process_document(doc.path)
            indexed = self.index_resume(resume)
            if indexed:
                resumes_indexed += 1
                chunks_indexed += indexed
                logger.info(
                    "Indexed %-28s | %2d chunks | %4.1f yrs | %d skills",
                    resume.metadata.name,
                    indexed,
                    resume.metadata.experience_years,
                    len(resume.metadata.skills),
                )

        elapsed = time.perf_counter() - start
        summary = {
            "resumes_indexed": resumes_indexed,
            "chunks_indexed": chunks_indexed,
            "total_vectors": self.store.count(),
            "embedding_provider": self.embedder.name,
            "elapsed_seconds": round(elapsed, 3),
        }
        logger.info("Indexing complete: %s", summary)
        return summary

    # ------------------------------------------------------------- retrieval
    def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[SearchHit]:
        """Semantic search over chunks (low-level retrieval primitive)."""

        query_embedding = self.embedder.embed_query(query)
        return self.store.query(query_embedding, top_k=top_k, where=where)

    def reset(self) -> None:
        """Clear the underlying vector store."""

        self.store.reset()
