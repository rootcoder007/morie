"""Tests for gh_ap_e1.ghosal_bernstein_poly."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_e1 import ghosal_bernstein_poly


def test_gh_ap_e1_basic():
    """Test basic functionality."""
    K_list = (5, 20, 80)
    result = ghosal_bernstein_poly(K_list=K_list)

    # Documented keys per the function's docstring/implementation
    assert "estimate" in result
    assert "err_by_K" in result
    assert "improving" in result
    assert "method" in result

    # estimate must be finite
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))

    # err_by_K has one entry per K value
    assert len(result["err_by_K"]) == len(K_list)

    # estimate equals the last entry of err_by_K
    assert result["estimate"] == result["err_by_K"][-1]

    # For f(x) = |x - 1/2| the Bernstein error decays like K^{-1/2},
    # so errs[-1] < errs[0]
    assert result["improving"] is True

    # Independent numeric sanity check: compute Bernstein polynomial
    # value at x = 0.5 for the largest K, and compare against f(0.5) = 0.
    # Since |x - 1/2| is symmetric about 0.5, B_K[f](0.5) should be tiny.
    K = K_list[-1]
    f = lambda x: abs(x - 0.5)
    x = 0.5
    import math as _math
    bk = sum(
        f(k / K) * _math.comb(K, k) * x ** k * (1.0 - x) ** (K - k)
        for k in range(K + 1)
    )
    expected_bk_minus_f = abs(bk - f(x))
    # The function's err_by_K[-1] is the max over x in {0/20, ..., 20/20},
    # which must be >= the error at x = 0.5.
    assert result["err_by_K"][-1] >= expected_bk_minus_f - 1e-12


def test_gh_ap_e1_edge():
    """Test edge cases."""
    # Single K value
    result = ghosal_bernstein_poly(K_list=(42,))
    assert "estimate" in result
    assert len(result["err_by_K"]) == 1
    assert result["estimate"] == result["err_by_K"][0]

    # Default K_list should yield three entries
    default_result = ghosal_bernstein_poly()
    assert len(default_result["err_by_K"]) == 3
    assert default_result["improving"] is True
