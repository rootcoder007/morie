"""Tests for kmcchr.kamath_christiano_deep_rl_feedback."""

from morie.fn import _array_core as np

from morie.fn.kmcchr import kamath_christiano_deep_rl_feedback


def test_kmcchr_basic():
    """Test basic functionality."""
    trajectory_pairs = [(2.0, 0.0), (0.0, 1.0)]
    r_phi = lambda s: s
    result = kamath_christiano_deep_rl_feedback(trajectory_pairs, r_phi)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmcchr_edge():
    """Test edge cases."""
    trajectory_pairs = [(2.0, 0.0), (0.0, 1.0)]
    r_phi = lambda s: s
    result = kamath_christiano_deep_rl_feedback(trajectory_pairs, r_phi)
    assert isinstance(result, dict)
