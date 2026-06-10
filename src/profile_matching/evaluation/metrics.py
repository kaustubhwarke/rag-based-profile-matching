"""Retrieval-quality and latency evaluation.

Given a set of job descriptions, a ground-truth mapping of relevant resume ids
per JD, and a configured :class:`JobMatcher`, this computes the standard
information-retrieval metrics required by the submission:

  * Precision@K, Recall@K
  * Mean Reciprocal Rank (MRR)
  * Average / p95 query latency (ms)

The metrics are backend-agnostic — they evaluate whatever embedder/vector store
the matcher was built with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from profile_matching.ingestion.loaders import load_document
from profile_matching.matching.engine import JobMatcher


def _resume_id_from_path(path: str) -> str:
    return Path(path).stem


@dataclass
class JobEvaluation:
    """Per-JD evaluation result."""

    job_id: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    latency_ms: float
    retrieved: list[str] = field(default_factory=list)
    relevant: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Aggregate evaluation across all JDs."""

    k: int
    per_job: list[JobEvaluation]

    @property
    def mean_precision(self) -> float:
        return _mean(e.precision_at_k for e in self.per_job)

    @property
    def mean_recall(self) -> float:
        return _mean(e.recall_at_k for e in self.per_job)

    @property
    def mrr(self) -> float:
        return _mean(e.reciprocal_rank for e in self.per_job)

    @property
    def mean_latency_ms(self) -> float:
        return _mean(e.latency_ms for e in self.per_job)

    @property
    def p95_latency_ms(self) -> float:
        values = sorted(e.latency_ms for e in self.per_job)
        if not values:
            return 0.0
        idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
        return values[idx]

    def summary(self) -> dict:
        return {
            "k": self.k,
            "jobs_evaluated": len(self.per_job),
            "precision_at_k": round(self.mean_precision, 4),
            "recall_at_k": round(self.mean_recall, 4),
            "mrr": round(self.mrr, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_matcher(
    matcher: JobMatcher,
    job_dir: str | Path,
    relevant_by_jd: dict[str, list[str]],
    k: int = 10,
    enforce_must_haves: bool = False,
) -> EvaluationReport:
    """Evaluate retrieval quality and latency over a directory of JDs.

    ``relevant_by_jd`` maps a JD id (filename stem) to the list of resume ids
    (filename stems) that are truly relevant. Must-have filtering is disabled by
    default so retrieval quality is measured independently of hard gating.
    """

    job_dir = Path(job_dir)
    per_job: list[JobEvaluation] = []

    for jd_path in sorted(job_dir.glob("*.txt")):
        job_id = jd_path.stem
        relevant = set(relevant_by_jd.get(job_id, []))
        if not relevant:
            continue

        doc = load_document(jd_path)
        response = matcher.match(
            doc.text, job_id=job_id, top_k=k, enforce_must_haves=enforce_must_haves
        )
        retrieved = [_resume_id_from_path(m.resume_path) for m in response.top_matches]

        hits = [r in relevant for r in retrieved]
        n_hits = sum(hits)
        precision = n_hits / len(retrieved) if retrieved else 0.0
        recall = n_hits / len(relevant) if relevant else 0.0
        rr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)

        per_job.append(
            JobEvaluation(
                job_id=job_id,
                precision_at_k=precision,
                recall_at_k=recall,
                reciprocal_rank=rr,
                latency_ms=response.latency_ms,
                retrieved=retrieved,
                relevant=sorted(relevant),
            )
        )

    return EvaluationReport(k=k, per_job=per_job)
