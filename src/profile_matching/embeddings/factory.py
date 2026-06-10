"""Embedding provider factory.

Resolves an :class:`EmbeddingProvider` from settings. If the requested managed
or local backend is unavailable (missing dependency or API key), the factory
falls back to the deterministic :class:`HashingEmbeddings` provider and logs a
clear warning — the system degrades gracefully instead of failing hard.
"""

from __future__ import annotations

from profile_matching.config import EmbeddingProvider as ProviderEnum
from profile_matching.config import EmbeddingSettings
from profile_matching.embeddings.base import EmbeddingProvider
from profile_matching.embeddings.providers import (
    CohereEmbeddings,
    HashingEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)
from profile_matching.logging_config import get_logger

logger = get_logger(__name__)


def build_embedding_provider(settings: EmbeddingSettings) -> EmbeddingProvider:
    """Instantiate the configured embedding provider, with safe fallback."""

    provider = settings.provider
    try:
        if provider is ProviderEnum.HUGGINGFACE:
            return HuggingFaceEmbeddings(settings.model, settings.batch_size)
        if provider is ProviderEnum.OPENAI:
            if not settings.openai_api_key:
                raise RuntimeError("PM_EMBEDDING__OPENAI_API_KEY is not set")
            return OpenAIEmbeddings(settings.model, settings.openai_api_key, settings.batch_size)
        if provider is ProviderEnum.COHERE:
            if not settings.cohere_api_key:
                raise RuntimeError("PM_EMBEDDING__COHERE_API_KEY is not set")
            return CohereEmbeddings(settings.model, settings.cohere_api_key, settings.batch_size)
        if provider is ProviderEnum.HASHING:
            return HashingEmbeddings(settings.dimension)
    except Exception as exc:  # noqa: BLE001 - graceful degradation is intentional
        logger.warning(
            "Embedding provider '%s' unavailable (%s). "
            "Falling back to offline 'hashing' embeddings.",
            provider.value,
            exc,
        )
        return HashingEmbeddings(settings.dimension)

    raise ValueError(f"Unknown embedding provider: {provider}")
