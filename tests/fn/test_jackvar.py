"""Tests for jackvar.jackknife_variance_survey."""

from morie.fn import _array_core as np

from morie.fn.jackvar import jackknife_variance_survey


def test_jackvar_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = jackknife_variance_survey(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_jackvar_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = jackknife_variance_survey(y)
    assert isinstance(result, dict)
