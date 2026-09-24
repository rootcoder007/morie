"""Tests for hmlrs.geron_learning_rate_schedule."""

from morie.fn import _array_core as np

from morie.fn.hmlrs import geron_learning_rate_schedule


def test_hmlrs_basic():
    """Test basic functionality."""
    t = np.array([float(i + 1) for i in range(40)])
    eta0 = 0.1
    t0 = 0.1
    result = geron_learning_rate_schedule(t, eta0, t0)
    assert isinstance(result, dict)
    assert "estimate" in result or "eta" in result


def test_hmlrs_edge():
    """Test edge cases."""
    t = np.array([float(i + 1) for i in range(40)])
    eta0 = 0.1
    t0 = 0.1
    result = geron_learning_rate_schedule(t, eta0, t0)
    assert isinstance(result, dict)
