"""Tests for fzbrd.fauzi_bias_reduced_kdfe."""

from morie.fn import _array_core as np

from morie.fn.fzbrd import fauzi_bias_reduced_kdfe


def test_fzbrd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_bias_reduced_kdfe(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_fzbrd_edge():
    """Test edge cases."""
    result = fauzi_bias_reduced_kdfe(np.array([42.0]))
    assert result["n"] == 1
