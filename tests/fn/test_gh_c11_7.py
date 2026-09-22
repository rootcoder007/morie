"""Tests for gh_c11_7.ghosal_rl_process."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c11_7 import ghosal_rl_process


def test_gh_c11_7_basic():
    """Test basic functionality."""
    alpha = 0.75
    n_grid = 200
    n_sim = 300
    result = ghosal_rl_process(alpha=alpha, n_grid=n_grid, n_sim=n_sim, seed=42)
    assert "estimate" in result
    estimate = float(np.asarray(result["estimate"], dtype=float))
    assert math.isfinite(estimate)
    # Independent expectation derived from the documented formula:
    # the variance-growth exponent should equal 2*alpha.
    expected = 2.0 * alpha
    assert math.isclose(estimate, expected, abs_tol=0.5)
    assert "expected" in result
    assert math.isclose(float(result["expected"]), expected)
    assert "gap" in result
    assert math.isclose(float(result["gap"]), abs(estimate - expected))
    assert "method" in result


def test_gh_c11_7_edge():
    """Test edge cases."""
    alpha = 0.5
    n_grid = 100
    n_sim = 100
    result = ghosal_rl_process(alpha=alpha, n_grid=n_grid, n_sim=n_sim, seed=7)
    estimate = float(np.asarray(result["estimate"], dtype=float))
    expected = 2.0 * alpha
    assert math.isfinite(estimate)
    assert math.isclose(estimate, expected, abs_tol=0.5)
