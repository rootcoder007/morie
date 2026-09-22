"""Tests for cvxsbp.boyd_subgradient."""

from morie.fn import _array_core as np

from morie.fn.cvxsbp import boyd_subgradient


def test_cvxsbp_basic():
    """Test basic functionality."""
    # f must be a callable; use a convex function and supply g explicitly.
    f = np.abs
    x = np.array([0.0])
    # At the kink of |.| any g in [-1, 1] is a subgradient.
    g = np.array([0.5])
    result = boyd_subgradient(f, x, g)
    assert isinstance(result, dict)
    # The function returns a RichResult with a documented payload.
    assert "is_subgradient" in result
    assert "violations" in result
    assert "worst_gap" in result
    assert "n_tested" in result
    # 0.5 is inside the subdifferential at the kink => no violations.
    assert bool(result["is_subgradient"]) is True
    assert int(result["violations"]) == 0
    # Sanity check on the number of probes: random + 2*dim axis probes.
    expected_n = 64 + 2 * x.size
    assert int(result["n_tested"]) == expected_n


def test_cvxsbp_edge():
    """Test edge cases."""
    # Same setup: f callable, x and g as 1-D points.
    f = np.abs
    x = np.array([2.0])
    g = np.array([1.0])  # derivative of |.| at x=2 is +1.
    result = boyd_subgradient(f, x, g)
    assert isinstance(result, dict)
    assert bool(result["is_subgradient"]) is True
    assert int(result["violations"]) == 0
    # Outside the subdifferential => a counterexample is found.
    bad = boyd_subgradient(f, x, np.array([0.5]))
    assert bool(bad["is_subgradient"]) is False
    assert int(bad["violations"]) > 0
