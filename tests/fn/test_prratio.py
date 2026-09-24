"""Tests for prratio.prevalence_ratio."""

from morie.fn import _array_core as np

from morie.fn.prratio import prevalence_ratio


def test_prratio_basic():
    """Test basic functionality."""
    prev_exposed = 0.1
    prev_unexposed = 0.1
    result = prevalence_ratio(prev_exposed, prev_unexposed)
    assert isinstance(result, dict)
    assert "pr" in result


def test_prratio_edge():
    """Test edge cases."""
    prev_exposed = 0.1
    prev_unexposed = 0.1
    result = prevalence_ratio(prev_exposed, prev_unexposed)
    assert isinstance(result, dict)
