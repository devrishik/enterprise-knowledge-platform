"""Unit tests: retrieval metrics against hand-computed values."""

from __future__ import annotations

import math

from app.evaluation.metrics import (
    average,
    mean_reciprocal_rank,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
)


class TestRecallPrecision:
    def test_perfect(self):
        retrieved = ["leave_policy.md", "security_policy.md"]
        relevant = {"leave_policy.md", "security_policy.md"}
        assert recall_at_k(retrieved, relevant, 5) == 1.0
        assert precision_at_k(retrieved, relevant, 5) == 1.0

    def test_partial(self):
        retrieved = ["handbook.md", "leave_policy.md", "it_policy.md"]
        relevant = {"leave_policy.md", "expense_policy.md"}
        assert recall_at_k(retrieved, relevant, 5) == 0.5
        assert precision_at_k(retrieved, relevant, 5) == 1 / 3

    def test_k_truncates(self):
        retrieved = ["a", "b", "leave_policy.md"]
        relevant = {"leave_policy.md"}
        assert recall_at_k(retrieved, relevant, 2) == 0.0
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set(), 5) == 0.0


class TestMRR:
    def test_first_hit(self):
        assert mean_reciprocal_rank(["leave_policy.md", "x"], {"leave_policy.md"}) == 1.0

    def test_second_hit(self):
        assert mean_reciprocal_rank(["x", "leave_policy.md"], {"leave_policy.md"}) == 0.5

    def test_no_hit(self):
        assert mean_reciprocal_rank(["x", "y"], {"z"}) == 0.0


class TestNDCG:
    def test_ideal_ordering(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, 5) == 1.0

    def test_reversed_order_penalised(self):
        ideal = ndcg_at_k(["a", "b", "c"], {"a", "b"}, 5)
        reversed_ = ndcg_at_k(["c", "b", "a"], {"a", "b"}, 5)
        assert reversed_ < ideal

    def test_no_relevant(self):
        assert ndcg_at_k(["a"], set(), 5) == 0.0

    def test_single_relevant_at_rank_2(self):
        # dcg = 1/log2(3), idcg = 1/log2(2)
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert abs(ndcg_at_k(["x", "a"], {"a"}, 5) - expected) < 1e-12


class TestStatsHelpers:
    def test_average(self):
        assert average([1.0, 2.0, 3.0]) == 2.0
        assert average([]) == 0.0

    def test_percentile(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert percentile(values, 50) == 30.0
        assert percentile(values, 95) == 50.0
        assert percentile([], 50) == 0.0
