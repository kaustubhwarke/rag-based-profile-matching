"""Shared pytest fixtures.

Tests run fully offline using the deterministic ``hashing`` embedder and the
``memory`` vector store, so no model downloads, API keys, or ChromaDB install
are required in CI.
"""

from __future__ import annotations

import pytest

from profile_matching.config import (
    EmbeddingProvider,
    EmbeddingSettings,
    MatchingSettings,
    Settings,
    VectorBackend,
    VectorStoreSettings,
)
from profile_matching.rag.pipeline import ResumeRAG

SAMPLE_RESUME = """Jane Doe
jane.doe@example.com | +1 (415) 555-1234 | San Francisco

SUMMARY
Senior Machine Learning Engineer with 7+ years of experience building NLP and
LLM systems, including retrieval-augmented generation pipelines.

EXPERIENCE
Senior Machine Learning Engineer — Acme Corp, San Francisco (2019 - Present)
  - Built a RAG pipeline serving millions of queries per day.
Machine Learning Engineer — DataNova, Austin (2017 - 2019)
  - Trained transformer models with PyTorch on AWS.

SKILLS
Python, PyTorch, TensorFlow, NLP, Machine Learning, AWS, Docker, Kubernetes, SQL

EDUCATION
M.S. in Computer Science — Stanford University (2017)

CERTIFICATIONS
- AWS Certified Solutions Architect
"""


@pytest.fixture
def offline_settings() -> Settings:
    return Settings(
        embedding=EmbeddingSettings(provider=EmbeddingProvider.HASHING, dimension=256),
        vectorstore=VectorStoreSettings(backend=VectorBackend.MEMORY),
        matching=MatchingSettings(top_k=5),
    )


@pytest.fixture
def sample_resume_file(tmp_path) -> str:
    path = tmp_path / "jane_doe.txt"
    path.write_text(SAMPLE_RESUME, encoding="utf-8")
    return str(path)


@pytest.fixture
def indexed_rag(offline_settings, sample_resume_file) -> ResumeRAG:
    rag = ResumeRAG(offline_settings)
    resume = rag.process_document(sample_resume_file)
    rag.index_resume(resume)
    return rag
