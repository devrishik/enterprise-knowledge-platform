"""Faithfulness scoring: the judge sees the question, and an expected refusal is a pass."""

from __future__ import annotations

import pytest

from app.evaluation.generation_eval import faithfulness, is_canonical_refusal

REFUSAL = (
    "I couldn't find sufficient evidence in the available documents to answer this question."
)


def test_canonical_refusal_matches_pipeline_sentence_only():
    assert is_canonical_refusal(REFUSAL)
    assert is_canonical_refusal(f"  {REFUSAL}  ")
    assert not is_canonical_refusal(REFUSAL + " Employees receive 22 days.")
    assert not is_canonical_refusal("I am not sure.")


def test_expected_refusal_is_faithful_and_skips_the_judge(monkeypatch: pytest.MonkeyPatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("judge should not run for a canonical expected refusal")

    monkeypatch.setattr("app.evaluation.generation_eval._judge", boom)
    result = faithfulness(REFUSAL, "", question="What is the CEO's favorite color?", expected_answerable=False)
    assert result["faithful"] is True
    assert result["mode"] == "rule"
    assert result["reason"] == "expected_refusal"


def test_refusal_on_an_answerable_question_is_still_judged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.evaluation.generation_eval._judge", lambda *_a, **_k: None)
    result = faithfulness(
        REFUSAL,
        "Full-time employees are entitled to 22 days of paid annual leave.",
        question="How many days of paid annual leave are employees entitled to?",
        expected_answerable=True,
    )
    assert result["mode"] == "heuristic"
    assert result["faithful"] is False


def test_judge_receives_the_question(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, str] = {}

    def fake_judge(_system: str, user_prompt: str) -> dict:
        captured["user"] = user_prompt
        return {
            "faithful": True,
            "supported_claims": 1,
            "unsupported_claims": 0,
            "explanation": "The 22 days are in the context.",
        }

    monkeypatch.setattr("app.evaluation.generation_eval._judge", fake_judge)
    result = faithfulness(
        "Employees are entitled to 22 days of paid annual leave.",
        "Full-time employees are entitled to 22 days of paid annual leave.",
        question="How many days of paid annual leave are employees entitled to?",
        expected_answerable=True,
    )
    assert "How many days of paid annual leave" in captured["user"]
    assert "[QUESTION]" in captured["user"]
    assert result["mode"] == "llm_judge"
    assert result["faithful"] is True


def test_extra_claims_after_a_refusal_are_not_an_automatic_pass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "app.evaluation.generation_eval._judge",
        lambda *_a, **_k: {"faithful": False, "explanation": "extra claim"},
    )
    result = faithfulness(
        REFUSAL + " The CEO's favorite color is blue.",
        "",
        question="What is the CEO's favorite color?",
        expected_answerable=False,
    )
    assert result["mode"] == "llm_judge"
    assert result["faithful"] is False
