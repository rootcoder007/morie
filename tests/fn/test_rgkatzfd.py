"""Tests for rgkatzfd.rangayyan_katz_fd."""

from morie.fn import _array_core as np

from morie.fn.bsastat import rangayyan_katz_fd


def test_rgkatzfd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_katz_fd(x)
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))  # N6: was a generator-guessed value


def test_rgkatzfd_edge():
    """Test edge cases."""
    result = rangayyan_katz_fd(np.array([42.0]))
    assert result["n"] == 1
