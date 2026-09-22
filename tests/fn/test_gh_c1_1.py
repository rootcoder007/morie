"""Tests for gh_c1_1.ghosal_bayes_rule_infinite."""

from morie.fn import _array_core as np

from morie.fn.gh_c1_1 import ghosal_bayes_rule_infinite


def test_gh_c1_1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_bayes_rule_infinite(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))


def test_gh_c1_1_edge():
    """Test edge cases."""
    grid = np.array([42.0])
    result = ghosal_bayes_rule_infinite(grid)
    # Documented payload keys: estimate, posterior, grid, method.
    assert "estimate" in result
    assert "posterior" in result
    assert "grid" in result
    # With a single grid point, the posterior mass lives entirely there,
    # so the posterior estimate must equal that grid value.
    th = np.asarray(grid, dtype=float).ravel()
    expected_estimate = float(th[0])
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert abs(float(result["estimate"]) - expected_estimate) < 1e-12
