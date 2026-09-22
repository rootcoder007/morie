"""Tests for gh_ap_j2.ghosal_crm_laplace."""

from morie.fn import _array_core as np

from morie.fn.gh_ap_j2 import ghosal_crm_laplace


def test_gh_ap_j2_basic():
    """Test basic functionality."""
    f_val = 1.0
    gamma_shape = 2.0
    u_max = 15.0
    n_grid = 6000
    result = ghosal_crm_laplace(f_val, gamma_shape, u_max, n_grid)
    assert "estimate" in result
    assert "closed_form" in result
    assert "gap" in result
    assert "method" in result
    closed_form = gamma_shape * np.log(1.0 + f_val)
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))
    assert np.abs(float(estimate) - float(closed_form)) < 0.01


def test_gh_ap_j2_edge():
    """Test edge case with non-default parameters."""
    f_val = 0.5
    gamma_shape = 3.0
    result = ghosal_crm_laplace(f_val, gamma_shape)
    assert "estimate" in result
    closed_form = gamma_shape * np.log(1.0 + f_val)
    estimate = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(estimate))
    assert np.abs(float(estimate) - float(closed_form)) < 0.05
