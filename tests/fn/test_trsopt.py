"""Tests for trsopt.trust_region_subproblem."""

from morie.fn import _array_core as np

from morie.fn.trsopt import trust_region_subproblem


def test_trsopt_basic():
    """Test basic functionality."""
    g = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    H = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = trust_region_subproblem(g, H)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_trsopt_edge():
    """Test edge cases."""
    g = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    H = np.random.default_rng(43).normal(0.0, 1.0, (8, 8))
    result = trust_region_subproblem(g, H)
    assert isinstance(result, dict)
