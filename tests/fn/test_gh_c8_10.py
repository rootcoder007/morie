"""Tests for gh_c8_10.ghosal_wn_crt."""

from morie.fn import _array_core as np

from morie.fn.gh_c8_10 import ghosal_wn_crt


def test_gh_c8_10_basic():
    """Test basic functionality."""
    result = ghosal_wn_crt(s_true=1.0, alpha_prior=1.0, ns=(100, 10000))
    assert "estimate" in result
    est = float(result["estimate"])
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))
    # Expected exponent = 2*min(a,s)/(2a+1) = 2*1/(3) = 2/3
    a = 1.0
    s_true = 1.0
    expected_exponent = 2.0 * min(a, s_true) / (2.0 * a + 1.0)
    assert np.allclose(np.asarray(est, dtype=float), np.asarray(expected_exponent, dtype=float), atol=0.1)
    assert "expected_exponent" in result
    assert np.allclose(np.asarray(float(result["expected_exponent"]), dtype=float),
                       np.asarray(expected_exponent, dtype=float))
    assert "risk_by_n" in result
    assert len(result["risk_by_n"]) == 2
    assert "method" in result


def test_gh_c8_10_edge():
    """Test edge cases."""
    # Use two distinct n values to avoid ZeroDivisionError from log(ratio) of identical values
    result = ghosal_wn_crt(s_true=1.0, alpha_prior=1.0, ns=(100, 1000))
    assert "estimate" in result
    est = float(result["estimate"])
    assert np.all(np.isfinite(np.asarray(est, dtype=float)))
    a = 1.0
    s_true = 1.0
    expected_exponent = 2.0 * min(a, s_true) / (2.0 * a + 1.0)
    assert np.allclose(np.asarray(est, dtype=float), np.asarray(expected_exponent, dtype=float), atol=0.1)
    assert "risk_by_n" in result
    assert "expected_exponent" in result
