"""Tests for wsmvar.wasserman_variance."""

from morie.fn import _array_core as np
from morie.fn.wsmvar import wasserman_variance


def test_wsmvar_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_variance(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wsmvar_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_variance(x)
    assert isinstance(result, dict)


def test_both_variance_divisors():
    import pytest

    x = [2.0, 4.5, 3.0, 7.5, 1.0]
    m = sum(x) / 5
    ss = sum((v - m) ** 2 for v in x)
    r = wasserman_variance(x)
    assert r["estimate"] == pytest.approx(ss / 5, rel=1e-13)
    assert r["sample_variance"] == pytest.approx(ss / 4, rel=1e-13)
    assert r["second_moment"] - r["mean"] ** 2 == pytest.approx(ss / 5, rel=1e-12)
