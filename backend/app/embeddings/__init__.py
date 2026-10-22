"""Embedding provider factory (cached singleton per process)."""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider

_cached: EmbeddingProvider | None = None


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "sentence_transformers":
        from app.embeddings.sentence_transformers_provider import SentenceTransformersProvider

        return SentenceTransformersProvider(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
        )
    if settings.embedding_provider == "hashing":
        from app.embeddings.hashing_provider import HashingEmbeddingProvider

        return HashingEmbeddingProvider(dimension=settings.embedding_dim)
    raise ValueError(f"Unknown embedding provider '{settings.embedding_provider}'")


def get_embedding_provider() -> EmbeddingProvider:
    global _cached
    if _cached is None:
        _cached = build_embedding_provider(get_settings())
    return _cached


def reset_embedding_provider() -> None:
    global _cached
    _cached = None
