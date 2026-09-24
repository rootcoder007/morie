"""Tests for rng240.rangayyan_ch4_complex_log_x_z."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_ch4_complex_log_x_z


def test_rng240_basic():
    """Test basic functionality."""
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_complex_log_x_z(z)
    assert isinstance(result, dict)
    assert "z_real" in result


def test_rng240_edge():
    """Test edge cases."""
    z = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = rangayyan_ch4_complex_log_x_z(z)
    assert isinstance(result, dict)
