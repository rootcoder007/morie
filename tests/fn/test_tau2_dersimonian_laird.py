"""Tests for tau2_dersimonian_laird.tau2_dersimonian_laird."""

from morie.fn import _array_core as np

from morie.fn.tau2_dersimonian_laird import tau2_dersimonian_laird


def test_ca11e44_basic():
    """Test basic functionality."""
    ys = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    ws_fixed = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = tau2_dersimonian_laird(ys, ws_fixed)
    assert isinstance(result, dict)
    assert "value" in result or "value" in result


def test_ca11e44_edge():
    """Test edge cases."""
    ys = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    ws_fixed = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = tau2_dersimonian_laird(ys, ws_fixed)
    assert isinstance(result, dict)
