"""Tests for gb241.gibbons_order_cdf."""

import math

from morie.fn import _array_core as np

from morie.fn.gb241 import gibbons_order_cdf


def _ostatcdf_independent(t, r, n, fx):
    """Independent re-implementation of the documented formula."""
    return sum(
        math.comb(n, i) * fx**i * (1.0 - fx) ** (n - i) for i in range(r, n + 1)
    )


def test_gb241_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)
    t_values = rng.uniform(0.0, 10.0, 100)
    r = 10
    n = 100

    def F(x):
        return 0.5 + 0.05 * (x - 5.0)

    results = [gibbons_order_cdf(float(t), r, n, F) for t in t_values]
    for t, res in zip(t_values, results):
        fx = F(float(t))
        expected_cdf = _ostatcdf_independent(float(t), r, n, fx)
        assert isinstance(res, dict)
        assert "cdf" in res
        assert "fx" in res
        assert "r" in res
        assert "n" in res
        assert "sf" in res
        assert "method" in res
        assert res["r"] == r
        assert res["n"] == n
        assert abs(res["fx"] - fx) < 1e-12
        assert abs(res["cdf"] - expected_cdf) < 1e-10
        assert abs(res["sf"] - (1.0 - expected_cdf)) < 1e-10


def test_gb241_edge():
    """Test edge cases: F_X(t) given directly as a float."""
    t = 5.0
    r = 1
    n = 1
    p = 0.7
    result = gibbons_order_cdf(t, r, n, p)
    expected = _ostatcdf_independent(t, r, n, p)
    assert isinstance(result, dict)
    assert "cdf" in result
    assert abs(result["cdf"] - expected) < 1e-12
    assert abs(result["sf"] - (1.0 - expected)) < 1e-12
    assert abs(result["fx"] - p) < 1e-12

    # also the trivial r=n=1, p=1.0
    result2 = gibbons_order_cdf(t, 1, 1, 1.0)
    assert abs(result2["cdf"] - 1.0) < 1e-12
