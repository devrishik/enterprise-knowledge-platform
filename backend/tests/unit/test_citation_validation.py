"""Unit tests: citation validation node (fabrication defence)."""

from __future__ import annotations

from app.agents.nodes.generation_nodes import (
    CitationValidatorNode,
    FinalResponseNode,
    GenerateAnswerNode,
    parse_generated_answer,
    quote_supported,
)
from app.core.config import get_settings
from app.llm.base import LLMResponse


def context(n: int) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk-{i}",
            "document_id": f"doc-{i}",
            "document_filename": f"policy_{i}.md",
            "text": f"Chunk number {i} about policy details. " * 5,
            "score": 0.9 - i * 0.1,
            "source": "reranked",
            "page_number": i,
            "section": f"Section {i}",
        }
        for i in range(1, n + 1)
    ]


def node() -> CitationValidatorNode:
    return CitationValidatorNode(get_settings())


def verbatim(i: int) -> str:
    return f"Chunk number {i} about policy details."


def test_quote_supported_requires_a_verbatim_span():
    chunk = "Chunk number 1 about policy details. " * 3
    assert quote_supported(verbatim(1), chunk)
    assert quote_supported("Chunk   number\n1   about   policy details.", chunk)
    assert not quote_supported("Chunk number 1", chunk)  # shorter than 20 characters
    assert not quote_supported("This quote was invented by the model today.", chunk)
    assert not quote_supported("", chunk)


def test_valid_citations_rebuilt_from_chunks():
    state = node()(
        {
            "reranked_results": context(3),
            "generated": {
                "answer": "Answer text.",
                "citations": [
                    {"ref": 2, "quote": verbatim(2)},
                    {"ref": 3, "quote": verbatim(3)},
                ],
            },
            "generation_attempts": 1,
        }
    )
    validation = state["citation_validation"]
    assert validation["valid"] is True
    assert validation["regenerate"] is False
    assert len(validation["citations"]) == 2
    citation = validation["citations"][0]
    # every displayed field comes from the actual retrieved chunk
    assert citation["chunk_id"] == "chunk-2"
    assert citation["document_name"] == "policy_2.md"
    assert citation["page"] == 2
    assert citation["excerpt"].startswith("Chunk number 2")
    assert citation["quote"] == verbatim(2)


def test_fabricated_ref_is_rejected_and_triggers_regen():
    state = node()(
        {
            "reranked_results": context(3),
            "generated": {"answer": "Answer.", "citations": [{"ref": 9}]},
            "generation_attempts": 1,
        }
    )
    validation = state["citation_validation"]
    assert validation["valid"] is False
    assert validation["invalid_refs"] == [9]
    assert validation["regenerate"] is True
    assert state["citation_feedback"]  # regeneration prompt includes the fix


def test_regeneration_is_bounded():
    state = node()(
        {
            "reranked_results": context(2),
            "generated": {"answer": "Answer.", "citations": [{"ref": 42}]},
            "generation_attempts": 2,  # already regenerated once
        }
    )
    assert state["citation_validation"]["regenerate"] is False


def test_mixed_valid_and_invalid():
    state = node()(
        {
            "reranked_results": context(2),
            "generated": {
                "answer": "Answer.",
                "citations": [{"ref": 1, "quote": verbatim(1)}, {"ref": 7, "quote": verbatim(7)}],
            },
            "generation_attempts": 1,
        }
    )
    validation = state["citation_validation"]
    assert validation["invalid_refs"] == [7]
    assert [c["ref"] for c in validation["citations"]] == [1]


def test_verbatim_quote_is_accepted():
    state = node()(
        {
            "reranked_results": context(1),
            "generated": {"answer": "Answer.", "citations": [{"ref": 1, "quote": verbatim(1)}]},
            "generation_attempts": 1,
        }
    )
    validation = state["citation_validation"]
    assert validation["valid"] is True
    assert validation["regenerate"] is False
    assert validation["citations"][0]["quote"] == verbatim(1)
    assert validation["citations"][0]["document_name"] == "policy_1.md"
    assert validation["citations"][0]["chunk_id"] == "chunk-1"


def test_fabricated_quote_is_rejected_and_regenerated():
    invented = "The board approved a secret bonus that is not in the policy."
    state = node()(
        {
            "reranked_results": context(1),
            "generated": {"answer": "Answer.", "citations": [{"ref": 1, "quote": invented}]},
            "generation_attempts": 1,
        }
    )
    validation = state["citation_validation"]
    assert validation["valid"] is False
    assert validation["citations"] == []
    assert validation["invalid_quotes"] == [{"ref": 1, "quote": invented}]
    assert validation["regenerate"] is True
    assert invented in state["citation_feedback"]


def test_missing_quote_is_not_regenerated_twice():
    state = node()(
        {
            "reranked_results": context(1),
            "generated": {"answer": "Answer.", "citations": [{"ref": 1, "quote": ""}]},
            "generation_attempts": 2,
        }
    )
    validation = state["citation_validation"]
    assert validation["valid"] is False
    assert validation["citations"] == []
    assert validation["regenerate"] is False


def test_generation_answers_the_original_question(monkeypatch):
    captured: dict[str, str] = {}

    class _FakeProvider:
        name = "fake"
        is_generative_llm = True

        def complete(self, messages, json_mode=False):
            captured["user"] = messages[-1].content
            return LLMResponse(
                content='{"answer": "22 days.", "citations": [], "confidence": 0.4}',
                model="fake",
            )

    monkeypatch.setattr(
        "app.agents.nodes.generation_nodes.get_llm_provider", lambda: _FakeProvider()
    )
    GenerateAnswerNode(get_settings())(
        {
            "question": "How many days of leave?",
            "effective_query": "annual leave entitlement days policy",
            "reranked_results": context(1),
            "conversation_history": [{"role": "user", "content": "Tell me about leave"}],
            "generation_attempts": 0,
        }
    )
    question_section = captured["user"].split("[QUESTION]", 1)[1]
    assert "How many days of leave?" in question_section
    assert "annual leave entitlement days policy" not in question_section
    assert "CONVERSATION HISTORY" in captured["user"]
    assert "not a source of facts" in captured["user"]


class TestFinalResponseNode:
    def test_no_answer_when_context_insufficient(self):
        state = FinalResponseNode(get_settings())(
            {
                "context_validation": {"status": "insufficient", "retrieval_confidence": 0.1, "chunk_count": 0},
                "citation_validation": {},
                "generated": {"answer": "Something hallucinated.", "confidence": 0.9},
            }
        )
        final = state["final"]
        assert final["insufficient_evidence"] is True
        assert final["answer"] == (
            "I couldn't find sufficient evidence in the available documents to answer this question."
        )
        assert final["citations"] == []
        assert final["confidence"] == 0.0

    def test_answer_kept_when_sufficient(self):
        state = FinalResponseNode(get_settings())(
            {
                "context_validation": {"status": "sufficient", "retrieval_confidence": 0.8, "chunk_count": 3},
                "citation_validation": {"citations": [{"ref": 1, "chunk_id": "c1"}]},
                "generated": {"answer": "22 days of leave.", "confidence": 0.85},
                "model": "extractive",
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )
        final = state["final"]
        assert final["insufficient_evidence"] is False
        assert final["answer"] == "22 days of leave."
        assert final["citations"][0]["chunk_id"] == "c1"


class TestParseGeneratedAnswer:
    def test_plain_json(self):
        parsed = parse_generated_answer(
            '{"answer": "42", "citations": [{"ref": 1}], "confidence": 0.9,'
            ' "insufficient_evidence": false, "conflicting_evidence": false}'
        )
        assert parsed.answer == "42"
        assert parsed.citations[0].ref == 1
        assert parsed.citations[0].quote == ""

    def test_json_in_code_fence(self):
        parsed = parse_generated_answer('```json\n{"answer": "ok", "citations": [2]}\n```')
        assert parsed.answer == "ok"
        assert parsed.citations[0].ref == 2
        assert parsed.citations[0].quote == ""

    def test_bare_int_citations(self):
        parsed = parse_generated_answer('{"answer": "ok", "citations": [1, 2], "confidence": 0.5}')
        assert [c.ref for c in parsed.citations] == [1, 2]
        assert [c.quote for c in parsed.citations] == ["", ""]

    def test_quote_is_preserved(self):
        parsed = parse_generated_answer(
            '{"answer": "ok", "citations": [{"ref": 1, "quote": "exact words copied from block 1"}]}'
        )
        assert parsed.citations[0].ref == 1
        assert parsed.citations[0].quote == "exact words copied from block 1"

    def test_invalid_json_raises(self):
        import pytest

        with pytest.raises(ValueError):
            parse_generated_answer("I cannot answer in JSON, sorry.")

    def test_confidence_clamped(self):
        import pytest

        with pytest.raises(Exception):
            parse_generated_answer('{"answer": "x", "confidence": 7.5}')
