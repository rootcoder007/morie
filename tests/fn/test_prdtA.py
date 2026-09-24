"""Tests for prdtA.prefix_evaluation."""

from morie.fn import _array_core as np
from morie.fn.prdtA import prefix_evaluation


def test_prdtA_basic():
    """Test basic functionality."""
    expr = ['+', 2, 3]
    result = prefix_evaluation(expr)
    assert result == 5


def test_prdtA_edge():
    """Test edge cases."""
    expr = [42]
    result = prefix_evaluation(expr)
    assert result == 42
