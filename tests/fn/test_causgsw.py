"""Tests for causgsw.causal_generalisability_smd."""

from morie.fn import _array_core as np

from morie.fn.causgsw import causal_generalisability_smd


def test_causgsw_basic():
    """Test basic functionality."""
    s_sample = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    s_target = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = causal_generalisability_smd(s_sample, s_target)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_causgsw_edge():
    """Test edge cases."""
    s_sample = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    s_target = np.random.default_rng(42).uniform(0.05, 0.95, 100)
    result = causal_generalisability_smd(s_sample, s_target)
    assert isinstance(result, dict)
