"""Sentence Transformers embedding provider (local, offline after first download)."""

from __future__ import annotations

import threading

from app.core.logging import get_logger
from app.embeddings.base import EmbeddingProvider


class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(self, model_name: str, dimension: int, batch_size: int = 32):
        self.name = f"sentence_transformers:{model_name}"
        self.dimension = dimension
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()
        self._log = get_logger("embeddings.st")

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._log.info("loading_embedding_model", extra={"context": {"model": self.name}})
                    self._model = SentenceTransformer(self.name.split(":", 1)[1])
                    real_dim = self._model.get_sentence_embedding_dimension()
                    if real_dim != self.dimension:
                        raise ValueError(
                            f"EKIP_EMBEDDING_DIM={self.dimension} does not match model "
                            f"'{self.name}' dimension {real_dim}. Fix the configuration."
                        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # cosine similarity space
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]
