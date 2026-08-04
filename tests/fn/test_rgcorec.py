"""Tests for rgcorec.rangayyan_correlation_coeff."""

from morie.fn import _array_core as np

from morie.fn.bsastat import rangayyan_correlation_coeff


def test_rgcorec_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = rangayyan_correlation_coeff(x, y)
    assert np.all(np.isfinite(np.asarray(result["statistic"], dtype=float)))  # N6: was a generator-guessed value


def test_rgcorec_edge():
    """Test edge cases."""
    result = rangayyan_correlation_coeff(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result["n"] == 2
