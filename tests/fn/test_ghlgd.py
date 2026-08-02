"""Tests for ghlgd.ghosal_log_density."""

from morie.fn import _array_core as np

from morie.fn.ghlgd import ghosal_log_density


def test_ghlgd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_log_density(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_ghlgd_edge():
    """Test edge cases."""
    result = ghosal_log_density(np.array([42.0]))
    assert result["n"] == 1
