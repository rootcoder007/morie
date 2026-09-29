"""Tests for dpmed.dp_median."""

from morie.fn import _array_core as np
from morie.fn.dpmed import dp_median


def test_dpmed_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_median(x, epsilon)
    assert isinstance(result, dict)
    assert "release" in result


def test_dpmed_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = dp_median(x, epsilon)
    assert isinstance(result, dict)


def test_dp_median_is_dp_quantile_at_one_half():
    import pytest

    from morie.fn.dpqua import dp_quantile

    x = [2.0, 5.0, 3.0, 9.0, 4.0, 7.0, 6.0]
    r = dp_median(x, epsilon=0.8, a=0.0, b=10.0, seed=5)
    q = dp_quantile(x, q=0.5, epsilon=0.8, a=0.0, b=10.0, seed=5)
    assert r["release"] == q["release"]
    assert r["true_median"] == pytest.approx(5.0, rel=1e-15)
