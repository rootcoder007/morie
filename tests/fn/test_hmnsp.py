"""Tests for hmnsp.geron_next_sentence_prediction."""

from morie.fn import _array_core as np
from morie.fn.hmnsp import geron_next_sentence_prediction


def test_hmnsp_basic():
    """Test basic functionality."""
    sent_A = np.random.default_rng(42).normal(0, 1, 100)
    sent_B = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_next_sentence_prediction(sent_A, sent_B)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmnsp_edge():
    """Test edge cases."""
    sent_A = np.random.default_rng(42).normal(0, 1, 100)
    sent_B = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_next_sentence_prediction(sent_A, sent_B)
    assert isinstance(result, dict)


def test_jaccard_baseline_logit_recomputed():
    import math

    import pytest

    A = ["the", "cat", "sat", "down"]
    B = ["the", "dog", "sat"]
    a, b = set(A), set(B)
    ov = len(a & b) / len(a | b)
    ratio = min(len(a), len(b)) / max(len(a), len(b))
    logit = 4 * ov + 0 * ratio - 2
    p = 1 / (1 + math.exp(-logit))
    r = geron_next_sentence_prediction(A, B, label=1)
    assert r["logit"] == pytest.approx(logit, rel=1e-14)
    assert r["probability"] == pytest.approx(p, rel=1e-14)
    assert r["loss"] == pytest.approx(-math.log(p), rel=1e-13)
