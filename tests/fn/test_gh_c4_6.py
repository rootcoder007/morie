"""Tests for gh_c4_6.ghosal_dp_post."""

from morie.fn import _array_core as np

from morie.fn.gh_c4_6 import ghosal_dp_post


def test_gh_c4_6_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 2.0
    n_in_A = 3.0
    n = float(x.size)
    result = ghosal_dp_post(x, alpha, n_in_A, n)
    # Posterior mean per eq. (4.11): M/(M+n) G0(A) + n/(M+n) P_n(A)
    G0_A = float(np.asarray(x).reshape(-1)[0])
    pn = n_in_A / n
    expected_mean = alpha / (alpha + n) * G0_A + n / (alpha + n) * pn
    expected_var = expected_mean * (1.0 - expected_mean) / (1.0 + alpha + n)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert float(result["estimate"]) == expected_mean
    assert float(result["posterior_var"]) == expected_var


def test_gh_c4_6_edge():
    """Test edge cases."""
    alpha = 0.0
    n_in_A = 1.0
    n = 1.0
    result = ghosal_dp_post(np.array([42.0]), alpha, n_in_A, n)
    # With M=0, posterior mean reduces to P_n(A) = n_in_A / n
    pn = n_in_A / n
    expected_mean = 0.0 / (0.0 + n) * 42.0 + n / (0.0 + n) * pn
    assert float(result["estimate"]) == expected_mean
    assert float(result["posterior_precision"]) == alpha + n
