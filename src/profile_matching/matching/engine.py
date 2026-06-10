"""JobMatcher — the Part B job-matching engine.

Pipeline:
    raw JD ─▶ parse (critical skills + must-haves)
           ─▶ hybrid search (dense + sparse over the indexed corpus)
           ─▶ must-have filter (hard requirements)
           ─▶ composite scoring + reasoning
           ─▶ top-K MatchResponse (assignment output contract)
"""

from __future__ import annotations

import time
from pathlib import Path

from profile_matching.config import Settings, get_settings
from profile_matching.ingestion.loaders import load_document
from profile_matching.logging_config import get_logger
from profile_matching.matching.hybrid_search import HybridSearcher
from profile_matching.matching.requirements import parse_job_description, passes_must_haves
from profile_matching.matching.scoring import MatchScorer
from profile_matching.models.job import JobDescription
from profile_matching.models.match import MatchResponse
from profile_matching.rag.pipeline import ResumeRAG
from profile_matching.utils.text import truncate

logger = get_logger(__name__)


class JobMatcher:
    """Matches a job description against the corpus indexed by :class:`ResumeRAG`."""

    def __init__(
        self,
        rag: ResumeRAG | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Reuse the same embedder/store the corpus was indexed with.
        self.rag = rag or ResumeRAG(self.settings)
        self.searcher = HybridSearcher(self.settings.matching)
        self.scorer = MatchScorer(self.settings.matching)

    # ------------------------------------------------------------------ API
    def match(
        self,
        job_text: str,
        *,
        job_id: str = "job",
        title: str | None = None,
        top_k: int | None = None,
        enforce_must_haves: bool = True,
    ) -> MatchResponse:
        """Match raw JD text and return a ranked :class:`MatchResponse`."""

        top_k = top_k or self.settings.matching.top_k
        start = time.perf_counter()

        job = parse_job_description(job_text, job_id=job_id, title=title)
        logger.info(
            "Matching job '%s' | critical_skills=%s | must_have=%d",
            job.title,
            job.critical_skills,
            len(job.must_have),
        )

        candidates = self.searcher.search(job, self.rag.embedder, self.rag.store)

        # Hard requirement gate.
        filtered = []
        for cand in candidates:
            ok, unmet = passes_must_haves(cand.metadata, job)
            if ok or not enforce_must_haves:
                filtered.append(cand)
            else:
                logger.debug("Filtered %s: %s", cand.metadata.get("name"), "; ".join(unmet))

        results = [self.scorer.score(cand, job) for cand in filtered]
        results.sort(key=lambda r: r.match_score, reverse=True)
        top = results[:top_k]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "Matched job '%s': %d/%d candidates passed, top score=%s, %.1f ms",
            job.title,
            len(filtered),
            len(candidates),
            top[0].match_score if top else "n/a",
            elapsed_ms,
        )

        return MatchResponse(
            job_description=truncate(job.text, 500),
            job_title=job.title,
            top_matches=top,
            total_candidates_considered=len(candidates),
            latency_ms=round(elapsed_ms, 2),
        )

    def match_file(self, path: str | Path, **kwargs) -> MatchResponse:
        """Match a job description loaded from a file."""

        doc = load_document(path)
        return self.match(doc.text, job_id=Path(path).stem, **kwargs)
