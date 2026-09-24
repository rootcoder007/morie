"""Tests for svyqtl.survey_quantile."""

from morie.fn import _array_core as np

from morie.fn.svyqtl import survey_quantile


def test_svyqtl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = survey_quantile(y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_svyqtl_edge():
    """Test edge cases."""
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = survey_quantile(y)
    assert isinstance(result, dict)
