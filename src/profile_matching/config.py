"""Centralised, environment-driven configuration.

Uses pydantic-settings so configuration can be supplied via environment
variables (prefixed ``PM_``) or a local ``.env`` file. Nested settings use the
``__`` delimiter, e.g. ``PM_EMBEDDING__PROVIDER=openai``.

Configuration is intentionally immutable at runtime and resolved once via the
cached :func:`get_settings` accessor — the canonical pattern for 12-factor apps.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(str, Enum):
    """Supported embedding backends."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    COHERE = "cohere"
    HASHING = "hashing"  # deterministic offline fallback (no external deps)


class VectorBackend(str, Enum):
    """Supported vector store backends."""

    CHROMA = "chroma"
    MEMORY = "memory"  # in-process fallback for tests / CI / demos


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    dimension: int = 384  # used by the hashing fallback / sanity checks
    openai_api_key: str | None = None
    cohere_api_key: str | None = None

    model_config = SettingsConfigDict(env_prefix="PM_EMBEDDING__", extra="ignore")


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    backend: VectorBackend = VectorBackend.CHROMA
    persist_directory: Path = Path(".vectorstore")
    collection_name: str = "resumes"

    model_config = SettingsConfigDict(env_prefix="PM_VECTORSTORE__", extra="ignore")


class ChunkingSettings(BaseSettings):
    """Document chunking configuration."""

    max_chunk_chars: int = 1200
    chunk_overlap_chars: int = 150
    min_chunk_chars: int = 80

    model_config = SettingsConfigDict(env_prefix="PM_CHUNKING__", extra="ignore")


class MatchingSettings(BaseSettings):
    """Retrieval and scoring configuration."""

    top_k: int = 10
    candidate_pool: int = 50  # over-fetch before re-ranking
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    # Weights for the composite 0-100 match score.
    score_semantic_weight: float = 0.55
    score_skill_weight: float = 0.30
    score_experience_weight: float = 0.15

    model_config = SettingsConfigDict(env_prefix="PM_MATCHING__", extra="ignore")


class PathSettings(BaseSettings):
    """Filesystem paths for inputs."""

    resume_dir: Path = Path("data/resumes")
    job_dir: Path = Path("data/job_descriptions")

    model_config = SettingsConfigDict(env_prefix="PM_PATHS__", extra="ignore")


class Settings(BaseSettings):
    """Top-level application settings."""

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    matching: MatchingSettings = Field(default_factory=MatchingSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="PM_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""

    return Settings()
