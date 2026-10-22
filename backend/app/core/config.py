"""Application configuration.

Every tunable knob of the platform lives here (database, providers, retrieval
parameters, upload limits).  Configuration is read from environment variables
prefixed with ``EKIP_`` (see ``.env.example``); nothing is hardcoded in
business logic.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ProjectRoot = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EKIP_",
        env_file=(ProjectRoot / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Enterprise Knowledge Intelligence Platform"
    environment: Literal["local", "docker", "test"] = "local"
    log_level: str = "INFO"
    # NoDecode: env values are comma-separated strings, not JSON — the
    # field_validator below splits them (pydantic-settings would otherwise
    # try json.loads on list fields and fail, e.g. in Docker Compose).
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:8080"]
    )

    # -------------------------------------------------------------- database
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/ekip"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    # ``pgvector`` is the production backend (cosine ANN via pgvector).
    # ``array`` is a compatibility backend used where the pgvector extension
    # cannot be installed (e.g. plain PostgreSQL on Windows): embeddings are
    # stored as float8[] and similarity is computed in Python.  The retrieval
    # interface is identical; see docs/design_decisions.md.
    vector_backend: Literal["pgvector", "array"] = "pgvector"

    # ------------------------------------------------------------- providers
    # LLM. ``openai`` is the default; ``extractive`` is a deterministic,
    # offline provider (no external calls) used for demos/tests without an
    # API key.  Additional providers can be registered in app/llm/factory.py.
    llm_provider: Literal["openai", "extractive"] = "openai"
    openai_api_key: str = ""
    openai_base_url: str = ""  # optional; empty = official OpenAI endpoint
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 1024
    llm_timeout_seconds: float = 60.0

    # Embeddings. sentence-transformers runs locally; ``hashing`` is a
    # deterministic feature-hashing embedder used for pipeline tests.
    embedding_provider: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384  # must match embedding_model
    embedding_batch_size: int = 32

    # Reranker. ``crossencoder`` downloads a local cross-encoder;
    # ``lexical`` is a deterministic lexical-overlap reranker for tests.
    reranker_provider: Literal["crossencoder", "lexical"] = "crossencoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # -------------------------------------------------------------- chunking
    chunking_strategy: Literal["structure", "fixed"] = "structure"
    chunk_size: int = 1200        # characters (~300 tokens of English prose)
    chunk_overlap: int = 200      # characters

    # ------------------------------------------------------------- retrieval
    dense_top_k: int = 20
    lexical_top_k: int = 20
    fusion_top_k: int = 10   # candidates passed to the reranker
    rerank_top_k: int = 5    # final context chunks after reranking
    rrf_k: int = 60          # Reciprocal Rank Fusion constant
    # Heuristic relevance gate for context validation (cross-encoder score or
    # fused-normalised score, both in [0,1]).  Deliberately conservative and
    # documented as a heuristic, not a calibrated probability.
    min_relevance_score: float = 0.35
    max_query_rewrites: int = 1  # one retry with the rewritten query on weak retrieval

    # ---------------------------------------------------------------- upload
    max_upload_mb: int = 25
    allowed_extensions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm"]
    )

    # --------------------------------------------------------------- pricing
    # JSON mapping model -> {input_per_1m, output_per_1m} in USD.  Kept in
    # configuration so outdated pricing never lands in core logic.
    model_pricing_json: str = (
        '{"gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},'
        ' "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.00}}'
    )

    # ----------------------------------------------------------------- misc
    seed_demo_data: bool = True  # create demo tenants/users on startup
    evaluation_max_questions: int = 50

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _split_extensions(cls, v: object) -> object:
        if isinstance(v, str):
            return [e.strip().lower() for e in v.split(",") if e.strip()]
        return v

    @property
    def model_pricing(self) -> dict[str, dict[str, float]]:
        try:
            parsed = json.loads(self.model_pricing_json)
        except json.JSONDecodeError as exc:  # pragma: no cover - config error
            raise ValueError("EKIP_MODEL_PRICING_JSON is not valid JSON") from exc
        return parsed

    def estimate_cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        """Return an estimated USD cost, or None when pricing is unavailable."""
        entry = self.model_pricing.get(model)
        if not entry:
            return None
        return round(
            prompt_tokens / 1_000_000 * entry.get("input_per_1m", 0.0)
            + completion_tokens / 1_000_000 * entry.get("output_per_1m", 0.0),
            6,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
