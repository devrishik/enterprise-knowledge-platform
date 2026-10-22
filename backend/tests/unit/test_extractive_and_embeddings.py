"""Unit tests: extractive provider (offline grounded answering) + hashing embeddings."""

from __future__ import annotations

import json
import math

from app.embeddings.hashing_provider import HashingEmbeddingProvider
from app.llm.base import LLMMessage
from app.llm.extractive_provider import ExtractiveProvider
from app.llm.prompts import build_answer_prompt
from app.retrieval import ScoredChunk


def make_prompt(question: str, texts: list[tuple[str, str, int | None, str | None]]) -> str:
    chunks = [
        ScoredChunk(
            chunk_id=f"c{i}", document_id=f"d{i}", document_filename=filename,
            text=text, score=0.9, source="reranked", page_number=page, section=section,
        )
        for i, (filename, text, page, section) in enumerate(texts, start=1)
    ]
    return build_answer_prompt(question, chunks)


class TestExtractiveProvider:
    def test_answers_with_citations_from_context(self):
        prompt = make_prompt(
            "How many days of paid annual leave do employees get?",
            [("leave_policy.md", "Employees are entitled to 22 days of paid annual leave per leave year. "
                                 "The entitlement accrues monthly. Unused days expire unless carried over.", None, "Annual Leave")],
        )
        result = ExtractiveProvider().answer_grounded(prompt)
        assert result["insufficient_evidence"] is False
        assert "22 days" in result["answer"]
        assert result["citations"] == [
            {
                "ref": 1,
                "quote": "Employees are entitled to 22 days of paid annual leave per leave year.",
            }
        ]
        assert 0 < result["confidence"] <= 1

    def test_refusal_when_no_overlap(self):
        prompt = make_prompt(
            "What is the CEO's favorite color?",
            [("leave_policy.md", "Employees are entitled to 22 days of paid annual leave per year.", None, None)],
        )
        result = ExtractiveProvider().answer_grounded(prompt)
        assert result["insufficient_evidence"] is True
        assert "couldn't find sufficient evidence" in result["answer"]
        assert result["citations"] == []

    def test_refusal_when_context_missing(self):
        result = ExtractiveProvider().answer_grounded("[QUESTION]\nanything?")
        assert result["insufficient_evidence"] is True

    def test_complete_roundtrip_via_prompt_builder(self):
        # the provider must parse exactly what the prompt builder emits
        provider = ExtractiveProvider()
        prompt = make_prompt(
            "What is the meal per diem for domestic travel?",
            [("expense_policy.md", "The daily meal per diem is 60 USD for domestic travel "
                                   "and 75 USD for international travel.", None, "Meal Per Diem")],
        )
        response = provider.complete([LLMMessage("user", prompt)])
        payload = json.loads(response.content)
        assert "60 USD" in payload["answer"]


class TestHashingEmbeddings:
    def test_deterministic(self):
        provider = HashingEmbeddingProvider(dimension=64)
        a = provider.embed_texts(["remote work policy"])[0]
        b = provider.embed_texts(["remote work policy"])[0]
        assert a == b

    def test_dimension_matches_config(self):
        assert len(HashingEmbeddingProvider(dimension=64).embed_texts(["x"])[0]) == 64

    def test_l2_normalised(self):
        vector = HashingEmbeddingProvider(dimension=64).embed_texts(["hello world policy"])[0]
        norm = math.sqrt(sum(v * v for v in vector))
        assert abs(norm - 1.0) < 1e-9

    def test_identical_texts_similarity_one(self):
        provider = HashingEmbeddingProvider(dimension=64)
        a, b = provider.embed_texts(["annual leave entitlement policy", "annual leave entitlement policy"])
        dot = sum(x * y for x, y in zip(a, b))
        assert abs(dot - 1.0) < 1e-9

    def test_different_texts_less_similar(self):
        provider = HashingEmbeddingProvider(dimension=256)
        a, b = provider.embed_texts(["annual leave entitlement", "kubernetes cluster networking"])
        dot = sum(x * y for x, y in zip(a, b))
        assert dot < 0.99
