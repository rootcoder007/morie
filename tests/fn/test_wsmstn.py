"""Tests for wsmstn.wasserman_sufficient."""

from morie.fn import _array_core as np
from morie.fn.wsmstn import wasserman_sufficient


def test_wsmstn_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = wasserman_sufficient(x)
    assert isinstance(result, dict)
    assert "T1" in result or "T1" in result


def test_wsmstn_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = wasserman_sufficient(x)
    assert isinstance(result, dict)


def test_sufficient_statistics_recomputed():
    import math

    import pytest

    x = [2.0, 4.5, 3.0, 7.5, 1.0]
    m = sum(x) / 5
    r = wasserman_sufficient(x)
    assert r["T1"] == pytest.approx(m, rel=1e-14)
    assert r["T2"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in x) / 4), rel=1e-13)
    assert r["mle_sigma2"] == pytest.approx(sum((v - m) ** 2 for v in x) / 5, rel=1e-13)
    b = wasserman_sufficient([1, 0, 1, 1, 0], family="bernoulli")
    assert b["T1"] == 3.0 and b["mle_mu"] == 0.6
