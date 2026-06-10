"""Concrete embedding provider implementations.

  * HuggingFaceEmbeddings — local sentence-transformers (production default)
  * OpenAIEmbeddings       — OpenAI text-embedding-3-* (managed)
  * CohereEmbeddings       — Cohere embed-* (managed)
  * HashingEmbeddings      — deterministic, dependency-free offline fallback

All providers L2-normalise their output so cosine similarity reduces to a dot
product downstream.
"""

from __future__ import annotations

import hashlib

import numpy as np

from profile_matching.embeddings.base import EmbeddingProvider
from profile_matching.logging_config import get_logger
from profile_matching.utils.text import tokenize

logger = get_logger(__name__)


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HuggingFaceEmbeddings(EmbeddingProvider):
    """Local embeddings via sentence-transformers."""

    def __init__(self, model: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model_name = model
        self._batch_size = batch_size
        logger.info("Loading sentence-transformers model: %s", model)
        self._model = SentenceTransformer(model)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"huggingface:{self._model_name}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbeddings(EmbeddingProvider):
    """Managed embeddings via the OpenAI API."""

    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str, api_key: str, batch_size: int = 128) -> None:
        from openai import OpenAI  # lazy import

        self._model = model
        self._batch_size = batch_size
        self._client = OpenAI(api_key=api_key)
        self._dim = self._DIMS.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return _l2_normalise(np.asarray(out, dtype=np.float32))


class CohereEmbeddings(EmbeddingProvider):
    """Managed embeddings via the Cohere API."""

    def __init__(self, model: str, api_key: str, batch_size: int = 96) -> None:
        import cohere  # lazy import

        self._model = model
        self._batch_size = batch_size
        self._client = cohere.Client(api_key)
        self._dim = 1024  # embed-english-v3.0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"cohere:{self._model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embed(
            texts=texts, model=self._model, input_type="search_document"
        )
        return _l2_normalise(np.asarray(resp.embeddings, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        resp = self._client.embed(
            texts=[text], model=self._model, input_type="search_query"
        )
        return _l2_normalise(np.asarray(resp.embeddings, dtype=np.float32))[0]


class HashingEmbeddings(EmbeddingProvider):
    """Deterministic hashed bag-of-words embeddings (no external dependencies).

    Not state-of-the-art, but fully offline and reproducible — it lets the
    entire pipeline (and the test suite / demo notebook) run anywhere without
    model downloads or API keys. Uses the hashing trick with sublinear term
    weighting, then L2-normalises so cosine similarity is well-defined.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing:{self._dim}d"

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        counts: dict[int, int] = {}
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - non-crypto
            idx = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            counts[idx] = counts.get(idx, 0)
            vec[idx] += sign
            counts[idx] += 1
        # Sublinear scaling dampens repeated-token dominance.
        for idx, c in counts.items():
            if c > 1:
                vec[idx] *= (1.0 + np.log(c)) / c
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros(
            (0, self._dim), dtype=np.float32
        )
        return _l2_normalise(matrix)
