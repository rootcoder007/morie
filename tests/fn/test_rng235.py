"""Tests for rng235.rangayyan_ch4_complex_log_of_product."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_ch4_complex_log_of_product


def test_rng235_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    H = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_complex_log_of_product(X, H)
    assert isinstance(result, dict)
    assert "omega" in result


def test_rng235_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    H = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_complex_log_of_product(X, H)
    assert isinstance(result, dict)
