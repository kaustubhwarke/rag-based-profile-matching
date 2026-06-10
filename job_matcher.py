#!/usr/bin/env python
"""job_matcher.py — Part B entry point (Job Matching Engine).

Semantic + keyword hybrid search, must-have filtering, 0-100 scoring with
reasoning, and the assignment's exact JSON output contract.

Runnable directly without installation:

    python job_matcher.py --jd-file data/job_descriptions/jd_01_ml_engineer.txt
    python job_matcher.py --jd-text "Senior Python engineer, 5+ years, AWS"
    python job_matcher.py --jd-file <path> --top-k 10 --output outputs/result.json
    python job_matcher.py --jd-file <path> --full   # include diagnostics

This thin shim adds ``src/`` to the path and delegates to the packaged CLI in
``profile_matching.cli.job_matcher_cli``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from profile_matching.cli.job_matcher_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
