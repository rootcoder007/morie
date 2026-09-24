"""Tests for hmmnl.geron_multinomial_logistic."""

from morie.fn import _array_core as np

from morie.fn.hmmnl import geron_multinomial_logistic


def test_hmmnl_basic():
    """Test basic functionality."""
    X = 0.5
    Y = 5
    result = geron_multinomial_logistic(X, Y)
    assert isinstance(result, dict)
    assert "estimate" in result or "Theta" in result


def test_hmmnl_edge():
    """Test edge cases."""
    X = 0.5
    Y = 5
    result = geron_multinomial_logistic(X, Y)
    assert isinstance(result, dict)
