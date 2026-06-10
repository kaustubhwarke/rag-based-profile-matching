#!/usr/bin/env python
"""One-command end-to-end demo (Part A + Part B in a single process).

Indexes the résumé corpus and immediately matches a job description, printing
the result in the assignment's JSON schema. Because indexing and matching share
one process, this works with *any* backend — including the offline fallback —
so you never hit the "in-memory index doesn't persist between commands" caveat.

Usage:
    python scripts/demo.py
    python scripts/demo.py data/job_descriptions/jd_03_data_engineer.txt
    python scripts/demo.py <jd-file> --top-k 5 --full

By default it runs fully offline (deterministic hashing embedder + in-memory
store). To use production backends instead, install them and override the env:
    pip install -e ".[all]"
    set PM_EMBEDDING__PROVIDER=huggingface   &  set PM_VECTORSTORE__BACKEND=chroma
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --- Path setup (works whether run from repo root or the scripts/ dir) -------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))          # so `src.profile_matching` resolves
sys.path.insert(0, str(_ROOT / "src"))  # so the package's internal imports resolve

# --- Offline-by-default (override via real env vars before running) ----------
os.environ.setdefault("PM_EMBEDDING__PROVIDER", "hashing")
os.environ.setdefault("PM_VECTORSTORE__BACKEND", "memory")
os.environ.setdefault("PM_LOG_LEVEL", "WARNING")

from src.profile_matching import JobMatcher, ResumeRAG  # noqa: E402

DEFAULT_JD = _ROOT / "data" / "job_descriptions" / "jd_01_ml_engineer.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end Part A + Part B demo.")
    parser.add_argument(
        "jd_file",
        nargs="?",
        default=str(DEFAULT_JD),
        help="Job-description file to match (default: jd_01_ml_engineer).",
    )
    parser.add_argument("--resume-dir", default=str(_ROOT / "data" / "resumes"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Emit full diagnostics instead of the minimal spec schema.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("PART A — Indexing résumés")
    print("=" * 72)
    rag = ResumeRAG()
    summary = rag.index_directory(args.resume_dir)
    print(json.dumps(summary, indent=2))

    print()
    print("=" * 72)
    print(f"PART B — Matching: {Path(args.jd_file).name}")
    print("=" * 72)
    matcher = JobMatcher(rag=rag)
    response = matcher.match_file(args.jd_file, top_k=args.top_k)

    payload = response.model_dump() if args.full else response.to_spec_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    print()
    print(
        f"Matched {len(response.top_matches)} candidates "
        f"(from {response.total_candidates_considered} considered) "
        f"in {response.latency_ms:.1f} ms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
