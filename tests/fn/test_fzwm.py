"""Tests for fzwm.fauzi_wilcoxon_moments."""

from morie.fn import _array_core as np

from morie.fn.fzwm import fauzi_wilcoxon_moments


def test_fzwm_basic():
    """Test basic functionality."""
    n = 100
    result = fauzi_wilcoxon_moments(n)
    assert isinstance(result, dict)
    assert "mean" in result
def test_fzwm_edge():
    """Test edge cases."""
    n = 100
    result = fauzi_wilcoxon_moments(n)
    assert isinstance(result, dict)
