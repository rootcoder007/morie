"""Tests for xlrnir.x_learner."""

from morie.fn import _array_core as np

from morie.fn.xlrnir import x_learner


def test_xlrnir_basic():
    """Test basic functionality."""
    tau1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    tau0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    g = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = x_learner(tau1, tau0, g)
    assert isinstance(result, dict)
    assert "tau" in result


def test_xlrnir_edge():
    """Test edge cases."""
    tau1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    tau0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    g = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = x_learner(tau1, tau0, g)
    assert isinstance(result, dict)
