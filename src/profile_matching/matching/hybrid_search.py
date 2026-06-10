"""Hybrid retrieval: dense semantic search + sparse keyword (BM25) re-ranking.

Strategy
--------
1. **Dense recall** — embed the job description and over-fetch a large pool of
   candidate chunks from the vector store (``candidate_pool``).
2. **Sparse signal** — run BM25 over that candidate pool, with the query
   expanded by the job's *critical skills* (each weighted), so exact
   skill-term matches are explicitly rewarded ("keyword for critical skills").
3. **Fusion** — min-max normalise both signals and combine with configurable
   weights into a per-chunk hybrid score.
4. **Aggregation** — collapse chunk scores to the resume level (best chunk per
   resume drives the score; per-section bests feed match reasoning).

Over-fetching then re-ranking is the standard, backend-agnostic way to get a
hybrid signal without maintaining a second full-corpus index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from profile_matching.config import MatchingSettings
from profile_matching.embeddings.base import EmbeddingProvider
from profile_matching.models.job import JobDescription
from profile_matching.utils.text import tokenize
from profile_matching.vectorstore.base import Metadata, SearchHit
from profile_matching.vectorstore.base import VectorStore


@dataclass
class ChunkMatch:
    """A scored chunk belonging to a candidate."""

    chunk_id: str
    section: str
    text: str
    semantic: float
    keyword: float
    hybrid: float


@dataclass
class CandidateAggregate:
    """All hybrid-scored chunks for one resume, plus aggregate signals."""

    resume_id: str
    metadata: Metadata
    chunks: list[ChunkMatch] = field(default_factory=list)
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0

    def best_chunks(self, limit: int = 3) -> list[ChunkMatch]:
        return sorted(self.chunks, key=lambda c: c.hybrid, reverse=True)[:limit]

    def matched_sections(self) -> list[str]:
        seen: list[str] = []
        for chunk in sorted(self.chunks, key=lambda c: c.hybrid, reverse=True):
            if chunk.section not in seen:
                seen.append(chunk.section)
        return seen


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class HybridSearcher:
    """Dense + sparse retrieval with resume-level aggregation."""

    def __init__(self, settings: MatchingSettings | None = None) -> None:
        self.settings = settings or MatchingSettings()

    def search(
        self,
        job: JobDescription,
        embedder: EmbeddingProvider,
        store: VectorStore,
    ) -> list[CandidateAggregate]:
        # 1. Dense recall over chunks.
        query_vec = embedder.embed_query(job.search_text)
        hits: list[SearchHit] = store.query(query_vec, top_k=self.settings.candidate_pool)
        if not hits:
            return []

        # 2. Sparse BM25 over the candidate pool, expanded by critical skills.
        corpus_tokens = [tokenize(h.document) for h in hits]
        bm25 = BM25Okapi(corpus_tokens) if any(corpus_tokens) else None
        query_tokens = tokenize(job.search_text)
        for skill in job.critical_skills:  # up-weight critical skill terms
            query_tokens.extend(tokenize(skill) * 2)
        keyword_raw = (
            list(bm25.get_scores(query_tokens)) if bm25 else [0.0] * len(hits)
        )

        # 3. Normalise & fuse per chunk.
        semantic_raw = [max(0.0, h.score) for h in hits]  # clip negatives
        sem_norm = _minmax(semantic_raw)
        kw_norm = _minmax(keyword_raw)
        w_sem, w_kw = self.settings.semantic_weight, self.settings.keyword_weight

        # 4. Aggregate to resume level.
        candidates: dict[str, CandidateAggregate] = {}
        for i, hit in enumerate(hits):
            resume_id = str(hit.metadata.get("resume_id", hit.id))
            hybrid = w_sem * sem_norm[i] + w_kw * kw_norm[i]
            chunk = ChunkMatch(
                chunk_id=hit.id,
                section=str(hit.metadata.get("section", "other")),
                text=hit.document,
                semantic=sem_norm[i],
                keyword=kw_norm[i],
                hybrid=hybrid,
            )
            agg = candidates.get(resume_id)
            if agg is None:
                agg = CandidateAggregate(resume_id=resume_id, metadata=hit.metadata)
                candidates[resume_id] = agg
            agg.chunks.append(chunk)

        for agg in candidates.values():
            agg.semantic_score = max(c.semantic for c in agg.chunks)
            agg.keyword_score = max(c.keyword for c in agg.chunks)
            agg.hybrid_score = max(c.hybrid for c in agg.chunks)

        return sorted(candidates.values(), key=lambda a: a.hybrid_score, reverse=True)
