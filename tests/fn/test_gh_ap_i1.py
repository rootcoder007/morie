"""Tests for gh_ap_i1.ghosal_gp_sample_cont."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_i1 import ghosal_gp_sample_cont


def test_gh_ap_i1_basic():
    """Test basic functionality."""
    p = 2.0
    alpha_exc = 1.0
    result = ghosal_gp_sample_cont(p=p, alpha_exc=alpha_exc)
    assert "estimate" in result
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))
    # estimate is the Hölder exponent alpha_exc / p
    expected_estimate = alpha_exc / p
    assert np.allclose(estimate, expected_estimate)
    assert "bm_limit_exponent" in result
    assert np.allclose(result["bm_limit_exponent"], 0.5)


def test_gh_ap_i1_edge():
    """Test edge cases."""
    p = 42.0
    alpha_exc = 1.0
    result = ghosal_gp_sample_cont(p=p, alpha_exc=alpha_exc)
    assert "estimate" in result
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.allclose(estimate, alpha_exc / p)
