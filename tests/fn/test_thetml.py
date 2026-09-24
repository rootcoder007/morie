"""Tests for thetml.theta_mle."""

from morie.fn import _array_core as np

from morie.fn.thetml import theta_mle


def test_thetml_basic():
    """Test basic functionality."""
    x = 1
    result = theta_mle(x)
    assert isinstance(result, dict)
    assert "theta" in result


def test_thetml_edge():
    """Test edge cases."""
    x = 1
    result = theta_mle(x)
    assert isinstance(result, dict)
