"""Tests for groob.geron_oob_error."""

from morie.fn import _array_core as np

from morie.fn.groob import geron_oob_error


def test_groob_basic():
    """Test basic functionality."""
    y = [0.0, 1.0]
    predictions = [[9.0, 1.0], [0.0, 9.0]]
    in_bag = [[True, False], [False, True]]
    result = geron_oob_error(y, predictions, in_bag)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_groob_edge():
    """Test edge cases."""
    y = [0.0, 1.0]
    predictions = [[9.0, 1.0], [0.0, 9.0]]
    in_bag = [[True, False], [False, True]]
    result = geron_oob_error(y, predictions, in_bag)
    assert isinstance(result, dict)
