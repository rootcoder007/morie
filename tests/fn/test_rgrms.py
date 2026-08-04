"""Tests for rgrms.rangayyan_rms."""

from morie.fn import _array_core as np

from morie.fn.bsastat import rangayyan_rms


def test_rgrms_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_rms(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgrms_edge():
    """Test edge cases."""
    result = rangayyan_rms(np.array([42.0]))
    assert result["n"] == 1
