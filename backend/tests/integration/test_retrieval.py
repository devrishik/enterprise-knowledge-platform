"""Integration tests: hybrid retrieval engine against real PostgreSQL."""

from __future__ import annotations

import pytest

from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]


@pytest.fixture()
def engine(db_session):
    from app.retrieval.engine import HybridRetrievalEngine

    return HybridRetrievalEngine(db_session)


@requires_db
class TestDenseRetrieval:
    def test_returns_relevant_chunks_with_scores(self, engine, ingested_corpus, acme_user):
        chunks, latency = engine.dense.search(
            "how many days of paid annual leave", acme_user.tenant_id, top_k=5
        )
        assert len(chunks) > 0
        assert latency >= 0
        assert all(c.source == "dense" for c in chunks)
        assert chunks[0].score >= chunks[-1].score  # ranked descending
        filenames = {c.document_filename for c in chunks}
        assert filenames & {"leave_policy.md", "employee_handbook.md"}

    def test_no_results_for_gibberish(self, engine, ingested_corpus, acme_user):
        chunks, _ = engine.dense.search("zzqxj vvbnmk", acme_user.tenant_id, top_k=5)
        assert all(c.score < 0.5 for c in chunks)  # weak scores at most


@requires_db
class TestLexicalRetrieval:
    def test_fts_finds_exact_terms(self, engine, ingested_corpus, acme_user):
        chunks, latency = engine.lexical.search("per diem domestic travel", acme_user.tenant_id, top_k=5)
        assert len(chunks) > 0
        assert latency >= 0
        assert any(c.document_filename == "expense_policy.md" for c in chunks)
        assert all(c.source == "lexical" for c in chunks)

    def test_no_match_yields_empty(self, engine, ingested_corpus, acme_user):
        chunks, _ = engine.lexical.search("zzqxjvvbnmk", acme_user.tenant_id, top_k=5)
        assert chunks == []


@requires_db
class TestHybridPipeline:
    def test_fusion_combines_and_dedupes(self, engine, ingested_corpus, acme_user):
        result = engine.retrieve("annual leave entitlement", acme_user.tenant_id, strategy="hybrid")
        assert result.dense and result.lexical
        assert len(result.fused) > 0
        chunk_ids = [c.chunk_id for c in result.fused]
        assert len(chunk_ids) == len(set(chunk_ids))  # deduplicated
        assert "dense" in result.latency_ms and "lexical" in result.latency_ms

    def test_rerank_selects_top_k(self, engine, ingested_corpus, acme_user):
        from app.core.config import get_settings

        result = engine.retrieve("password rotation requirements", acme_user.tenant_id,
                                 strategy="hybrid_rerank")
        settings = get_settings()
        assert len(result.reranked) <= settings.rerank_top_k
        assert "rerank" in result.latency_ms
        assert result.reranked[0].document_filename == "security_policy.md"
        scores = [c.score for c in result.reranked]
        assert scores == sorted(scores, reverse=True)

    def test_strategy_variants_return_different_final_sets(self, engine, ingested_corpus, acme_user):
        question = "what is the home office stipend for remote work"
        dense = engine.retrieve(question, acme_user.tenant_id, strategy="dense").final_chunks()
        hybrid = engine.retrieve(question, acme_user.tenant_id, strategy="hybrid").final_chunks()
        reranked = engine.retrieve(question, acme_user.tenant_id, strategy="hybrid_rerank").final_chunks()
        assert all(c.source == "dense" for c in dense)
        assert all(c.source == "fused" for c in hybrid)
        assert all(c.source == "reranked" for c in reranked)


@requires_db
class TestFullTextIndexGeneratedColumn:
    def test_tsv_maintained_by_postgres(self, db_session, ingested_corpus):
        from sqlalchemy import text

        row = db_session.execute(text(
            "SELECT count(*) FROM document_chunks "
            "WHERE tsv @@ websearch_to_tsquery('english', 'per diem')"
        )).scalar_one()
        assert row > 0
