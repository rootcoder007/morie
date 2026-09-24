"""Tests for rrblp.rrblup_marker_effects."""

from morie.fn import _array_core as np

from morie.fn.rrblp import rrblup_marker_effects


def test_rrblp_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    M = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    sigma2_m = 0.1
    result = rrblup_marker_effects(X, y, M, sigma2_m)
    assert isinstance(result, dict)
    assert "beta" in result


def test_rrblp_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    M = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    sigma2_m = 0.1
    result = rrblup_marker_effects(X, y, M, sigma2_m)
    assert isinstance(result, dict)
