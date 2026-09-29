"""Tests for randres.randomized_response."""

from morie.fn import _array_core as np
from morie.fn.randres import randomized_response


def test_randres_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    truth = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = randomized_response(y, truth, p)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_randres_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    truth = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = randomized_response(y, truth, p)
    assert isinstance(result, dict)


def test_warner_estimator_recomputed():
    import math

    import pytest

    y = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    lam = 6 / 10
    p = 0.75
    pi = (lam - (1 - p)) / (2 * p - 1)
    r = randomized_response(y, p=p)
    assert r["estimate"] == pytest.approx(pi, rel=1e-14)
    assert r["se"] == pytest.approx(math.sqrt(lam * (1 - lam) / (10 * 0.25)), rel=1e-13)
