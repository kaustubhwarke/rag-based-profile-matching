#!/usr/bin/env python
"""resume_rag.py — Part A entry point (RAG System Setup).

Document processing pipeline + metadata extraction + vector indexing.

Runnable directly without installation:

    python resume_rag.py index --resume-dir data/resumes
    python resume_rag.py inspect data/resumes/0001_priya_sharma.txt
    python resume_rag.py search "senior python engineer with aws"
    python resume_rag.py stats

This thin shim adds ``src/`` to the path and delegates to the packaged CLI in
``profile_matching.cli.resume_rag_cli`` so the same logic backs both the script
and the installed ``resume-rag`` console entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from profile_matching.cli.resume_rag_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
