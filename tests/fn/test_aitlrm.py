"""Tests for aitlrm.compositional_lrmean."""

import math

from morie.fn import _array_core as np

from morie.fn.aitlrm import compositional_lrmean


def test_aitlrm_basic():
    """Test basic functionality with a strictly positive composition."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.1, 5.0, (100, 5))
    result = compositional_lrmean(X)

    # The function returns a RichResult with a payload dict.
    assert isinstance(result, dict)

    # Documented keys.
    assert "clr_mean" in result
    assert "center" in result
    assert "sum_clr_mean" in result
    assert "n" in result
    assert "D" in result

    # Shapes.
    n, D = X.shape
    assert result["n"] == n
    assert result["D"] == D
    assert len(result["clr_mean"]) == D
    assert len(result["center"]) == D

    # The centre must be strictly positive and sum to ``total`` (default 1).
    assert all(v > 0 for v in result["center"])
    total = 1.0
    s = sum(result["center"])
    assert math.isclose(s, total, rel_tol=1e-10, abs_tol=1e-12)

    # Verify clr_mean = (1/n) sum_k clr(x_k) using plain arithmetic on inputs.
    gm = [math.exp(sum(math.log(v) for v in X[:, j]) / n) for j in range(D)]
    g_total = sum(gm)
    expected_center = [total * g / g_total for g in gm]
    for got, exp in zip(result["center"], expected_center):
        assert math.isclose(got, exp, rel_tol=1e-10, abs_tol=1e-12)

    # sum_clr_mean must equal sum(clr_mean) and be ~0 (clr is zero-sum by def).
    assert math.isclose(result["sum_clr_mean"], sum(result["clr_mean"]),
                        rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(result["sum_clr_mean"], 0.0, abs_tol=1e-10)


def test_aitlrm_edge():
    """Test edge cases with a strictly positive composition."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.1, 5.0, (100, 5))
    result = compositional_lrmean(X)
    assert isinstance(result, dict)
    assert result["n"] == 100
    assert result["D"] == 5
