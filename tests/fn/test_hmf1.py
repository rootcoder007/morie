"""Tests for hmf1.geron_f1_score."""

from morie.fn import _array_core as np

from morie.fn.hmf1 import geron_f1_score


def test_hmf1_basic():
    """Test basic functionality."""
    y_true = [1, 1, 1, 0]
    y_pred = [1, 0, 0, 0]
    result = geron_f1_score(y_true, y_pred)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_hmf1_edge():
    """Test edge cases."""
    y_true = [1, 1, 1, 0]
    y_pred = [1, 0, 0, 0]
    result = geron_f1_score(y_true, y_pred)
    assert isinstance(result, dict)
