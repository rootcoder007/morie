"""Tests for middle.middle_out."""

from morie.fn import _array_core as np
from morie.fn.middle import middle_out


def test_middle_basic():
    """Test basic functionality with mixed aggregation/disaggregation rows."""
    middle = np.array([1.0, 2.0, 3.0])
    S = [
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.3, 0.7],
        [0.4, 0.0, 0.6],
    ]
    result = middle_out(middle, S)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "aggregated" in result
    assert "disaggregated" in result
    assert "middle" in result
    assert "n" in result
    assert "method" in result
    assert len(result["estimate"]) == 5
    assert result["n"] == 5
    assert len(result["aggregated"]) == 2
    assert len(result["disaggregated"]) == 3
    assert len(result["middle"]) == 3


def test_middle_edge():
    """Test edge case: single middle series, single aggregation row."""
    middle = np.array([2.5])
    S = [[1.0]]
    result = middle_out(middle, S)
    assert isinstance(result, dict)
    assert result["n"] == 1
    assert len(result["estimate"]) == 1
    assert len(result["aggregated"]) == 1
    assert len(result["disaggregated"]) == 0
    assert len(result["middle"]) == 1
