"""Enterprise RAG-based resume-to-job profile matching system.

Public API:
    from profile_matching import ResumeRAG, JobMatcher, get_settings
"""

from __future__ import annotations

from profile_matching.config import Settings, get_settings
from profile_matching.matching.engine import JobMatcher
from profile_matching.rag.pipeline import ResumeRAG

__version__ = "1.0.0"

__all__ = [
    "ResumeRAG",
    "JobMatcher",
    "Settings",
    "get_settings",
    "__version__",
]
