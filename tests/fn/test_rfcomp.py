"""Tests for rfcomp.robust_factor_analysis."""

from morie.fn import _array_core as np

from morie.fn.rfcomp import robust_factor_analysis


def test_rfcomp_basic():
    """Test basic functionality."""
    X = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = robust_factor_analysis(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_rfcomp_edge():
    """Test edge cases."""
    X = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = robust_factor_analysis(X)
    assert isinstance(result, dict)
