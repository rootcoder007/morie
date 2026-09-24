"""Tests for wsmchi.wasserman_chi_sq_gof."""

from morie.fn import _array_core as np

from morie.fn.wsmchi import wasserman_chi_sq_gof


def test_wsmchi_basic():
    """Test basic functionality."""
    observed = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    expected = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = wasserman_chi_sq_gof(observed, expected)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_wsmchi_edge():
    """Test edge cases."""
    observed = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    expected = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = wasserman_chi_sq_gof(observed, expected)
    assert isinstance(result, dict)
