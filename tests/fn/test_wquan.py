"""Tests for wquan.weighted_quantile."""

from morie.fn import _array_core as np
from morie.fn.wquan import weighted_quantile


def test_wquan_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    p = 5
    result = weighted_quantile(y, weights, p)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_wquan_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    p = 5
    result = weighted_quantile(y, weights, p)
    assert isinstance(result, dict)


def test_harrell_davis_weights_for_three_points():
    """n = 3, p = 1/2: a = b = 2, so I_x(2, 2) = 3x^2 - 2x^3 and the weights
    are 7/27, 13/27, 7/27."""
    import pytest

    x = [4.0, 1.0, 2.5]
    r = weighted_quantile(x, None, 0.5)

    def ib(t):
        return 3 * t * t - 2 * t**3

    w = [ib(1 / 3) - ib(0), ib(2 / 3) - ib(1 / 3), ib(1) - ib(2 / 3)]
    assert [float(v) for v in r["w"]] == pytest.approx(w, rel=1e-12)
    assert r["estimate"] == pytest.approx(w[0] * 1.0 + w[1] * 2.5 + w[2] * 4.0, rel=1e-12)
    assert r["ecdf"] == 2.5
