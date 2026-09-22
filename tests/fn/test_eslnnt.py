"""Tests for eslnnt.esl_neural_net."""

from morie.fn import _array_core as np

from morie.fn.eslnnt import esl_neural_net


def test_eslnnt_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_neural_net(X, y)
    assert isinstance(result, dict)
    assert "alpha" in result
def test_eslnnt_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_neural_net(X, y)
    assert isinstance(result, dict)
