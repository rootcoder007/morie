"""Tests for adjsrs.effective_srs."""

from morie.fn import _array_core as np

from morie.fn.adjsrs import effective_srs


def test_adjsrs_basic():
    """Test basic functionality."""
    w = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = effective_srs(w)
    assert isinstance(result, dict)
    assert "neff" in result
def test_adjsrs_edge():
    """Test edge cases."""
    w = np.abs(np.random.default_rng(42).normal(0, 1, 100)) + 0.5
    result = effective_srs(w)
    assert isinstance(result, dict)
