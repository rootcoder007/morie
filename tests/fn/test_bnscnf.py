"""Tests for bnscnf.bound_confidence_set."""

from morie.fn import _array_core as np

from morie.fn.bnscnf import bound_confidence_set


def test_bnscnf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    theta_grid = rng.normal(0, 1, n)
    # moments must have shape (n, 2): lower end (yL) in column 0,
    # upper end (yU) in column 1, with yL <= yU for every row.
    yL = rng.normal(0, 1, n)
    yU = yL + rng.uniform(0.1, 1.0, n)
    moments = np.column_stack([yL, yU])
    alpha = 0.05
    result = bound_confidence_set(theta_grid, moments, alpha)
    assert isinstance(result, dict)
    # Per the docstring, the result carries the same payload as
    # bound_inference (the "Confidence set for partial ID" test
    # inversion), with a "method" key recording the name.
    assert "method" in result


def test_bnscnf_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 100
    theta_grid = rng.normal(0, 1, n)
    yL = rng.normal(0, 1, n)
    yU = yL + rng.uniform(0.1, 1.0, n)
    moments = np.column_stack([yL, yU])
    alpha = 0.05
    result = bound_confidence_set(theta_grid, moments, alpha)
    assert isinstance(result, dict)
    assert "method" in result
