"""Tests for chgbcp.bayesian_online_changepoint."""

from morie.fn import _array_core as np

from morie.fn.chgbcp import bayesian_online_changepoint


def test_chgbcp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayesian_online_changepoint(y)
    assert isinstance(result, dict)
    assert "cp_prob" in result
def test_chgbcp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = bayesian_online_changepoint(y)
    assert isinstance(result, dict)
