"""Unit tests: heuristic query analysis (offline provider path)."""

from __future__ import annotations

from app.agents.nodes.query_understanding import heuristic_analysis


def test_clear_query_is_not_rewritten():
    analysis = heuristic_analysis("How many days of paid annual leave do employees get?", history=None)
    assert analysis["needs_rewrite"] is False
    assert analysis["rewritten_query"] == ""
    assert analysis["answerable_from_documents"] is True


def test_pronoun_followup_with_history_is_rewritten():
    history = [
        {"role": "user", "content": "What is the parental leave policy for primary caregivers?"},
        {"role": "assistant", "content": "Primary caregivers get 16 weeks of paid parental leave."},
    ]
    analysis = heuristic_analysis("What about the previous policy for secondary caregivers?", history=history)
    assert analysis["needs_rewrite"] is True
    assert analysis["rewritten_query"]
    assert "referring to" in analysis["rewritten_query"]


def test_short_pronoun_query_is_rewritten():
    analysis = heuristic_analysis("what about it?", history=None)
    # no history to anchor the pronoun -> still flagged, cannot produce context
    assert analysis["needs_rewrite"] is True


def test_chitchat_is_not_answerable():
    assert heuristic_analysis("hi there!", history=None)["answerable_from_documents"] is False
    assert heuristic_analysis("thanks", history=None)["answerable_from_documents"] is False


def test_compound_question_split_into_subqueries():
    question = "What is the domestic per diem; and also what is the international per diem?"
    analysis = heuristic_analysis(question, history=None)
    assert len(analysis["sub_queries"]) == 2


def test_no_false_split_for_simple_and():
    analysis = heuristic_analysis("What are the health and dental benefits?", history=None)
    assert analysis["sub_queries"] == []
