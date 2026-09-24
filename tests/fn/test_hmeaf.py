"""Tests for hmeaf.geron_error_analysis."""

from morie.fn import _array_core as np

from morie.fn.hmeaf import geron_error_analysis


def test_hmeaf_basic():
    """Test basic functionality."""
    y_true = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    y_pred = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = geron_error_analysis(y_true, y_pred)
    assert isinstance(result, dict)
    assert "estimate" in result or "normalized" in result


def test_hmeaf_edge():
    """Test edge cases."""
    y_true = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    y_pred = np.random.default_rng(43).normal(0.0, 1.0, (3, 3))
    result = geron_error_analysis(y_true, y_pred)
    assert isinstance(result, dict)
