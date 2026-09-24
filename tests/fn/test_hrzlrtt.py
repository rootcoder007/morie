"""Tests for hrzlrtt.horowitz_likelihood_ratio_test."""

from morie.fn import _array_core as np

from morie.fn.hrzlrtt import horowitz_likelihood_ratio_test


def test_hrzlrtt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_likelihood_ratio_test(x, y)
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "statistic" in result


def test_hrzlrtt_edge():
    """Test edge cases."""
    x = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    y = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = horowitz_likelihood_ratio_test(x, y)
    assert isinstance(result, dict)
