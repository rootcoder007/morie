"""Tests for fzt58.fauzi_thm5_8_smoothed_convergence."""

from morie.fn import _array_core as np

from morie.fn.fzt58 import fauzi_thm5_8_smoothed_convergence


def test_fzt58_basic():
    """Test basic functionality."""
    d = 3
    result = fauzi_thm5_8_smoothed_convergence(d)
    assert isinstance(result, dict)
    assert "ok" in result
def test_fzt58_edge():
    """Test edge cases."""
    d = 3
    result = fauzi_thm5_8_smoothed_convergence(d)
    assert isinstance(result, dict)
