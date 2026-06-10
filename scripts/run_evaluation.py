#!/usr/bin/env python
"""Reproducible end-to-end evaluation harness.

Indexes the resume corpus, runs every job description, and reports retrieval
accuracy (Precision@K, Recall@K, MRR) and latency. Results are printed and
written to ``outputs/evaluation.json``.

    python scripts/run_evaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # so `src.profile_matching` resolves
sys.path.insert(0, str(_ROOT / "src"))  # so the package's internal absolute imports resolve

from src.profile_matching import JobMatcher, ResumeRAG  # noqa: E402
from src.profile_matching.evaluation import evaluate_matcher  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ground_truth = json.loads((ROOT / "data" / "ground_truth.json").read_text())

    rag = ResumeRAG()
    index_summary = rag.index_directory(ROOT / "data" / "resumes")

    matcher = JobMatcher(rag=rag)
    report = evaluate_matcher(
        matcher,
        ROOT / "data" / "job_descriptions",
        ground_truth["relevant_by_jd"],
        k=10,
    )

    out = {
        "index": index_summary,
        "retrieval": report.summary(),
        "per_job": [
            {
                "job_id": e.job_id,
                "precision_at_k": round(e.precision_at_k, 3),
                "recall_at_k": round(e.recall_at_k, 3),
                "reciprocal_rank": round(e.reciprocal_rank, 3),
                "latency_ms": e.latency_ms,
            }
            for e in report.per_job
        ],
    }

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    (outputs / "evaluation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    print(f"\nWrote outputs/evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
