"""Tests for gh_ap_d1.ghosal_exp_test."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_d1 import ghosal_exp_test


def test_gh_ap_d1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    err_null = float(rng.uniform(0.05, 0.5))
    err_alt = float(rng.uniform(0.05, 0.5))
    n = 100
    result = ghosal_exp_test(err_null, err_alt, n)
    assert isinstance(result, dict)
    # The RichResult payload exposes the documented keys.
    for key in ("rate", "rate_null", "rate_alt", "bound", "exponential", "n"):
        assert key in result
    # Independent recomputation of the documented formula.
    import math
    c0_expected = -math.log(err_null) / n
    c1_expected = -math.log(err_alt) / n
    c_expected = min(c0_expected, c1_expected)
    bound_expected = math.exp(-c_expected * n)
    assert math.isclose(result["rate_null"], c0_expected, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["rate_alt"], c1_expected, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["rate"], c_expected, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["bound"], bound_expected, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(result["bound"], math.exp(-c_expected * n), rel_tol=1e-12, abs_tol=1e-12)
    # Both errors are < 1 here, so both rates are positive and the
    # sequence is exponentially consistent at a positive rate.
    assert result["rate"] > 0.0
    assert result["exponential"] == 1.0
    assert result["n"] == float(n)


def test_gh_ap_d1_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    # err_null == 1.0 gives rate_null == 0; err_alt < 1 makes rate_alt > 0,
    # so the min rate is 0 and `exponential` should be 0.
    err_null = 1.0
    err_alt = float(rng.uniform(0.05, 0.5))
    n = 50
    result = ghosal_exp_test(err_null, err_alt, n)
    assert isinstance(result, dict)
    assert result["rate"] == 0.0
    assert result["rate_null"] == 0.0
    assert result["rate"] == result["rate_null"]
    assert result["exponential"] == 0.0
    assert result["bound"] == 1.0
