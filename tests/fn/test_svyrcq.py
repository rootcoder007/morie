"""Tests for svyrcq.survey_quantile_reg."""

from morie.fn import _array_core as np

from morie.fn.svyrcq import survey_quantile_reg


def test_svyrcq_basic():
    """Test basic functionality."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = survey_quantile_reg(X, y)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_svyrcq_edge():
    """Test edge cases."""
    X = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = survey_quantile_reg(X, y)
    assert isinstance(result, dict)
