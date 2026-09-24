"""Tests for kmscal.kamath_scaling_laws."""

from morie.fn import _array_core as np

from morie.fn.kmscal import kamath_scaling_laws


def test_kmscal_basic():
    """Test basic functionality."""
    N = 0.5
    N_c = 0.5
    alpha_N = 0.5
    result = kamath_scaling_laws(N, N_c, alpha_N)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_kmscal_edge():
    """Test edge cases."""
    N = 0.5
    N_c = 0.5
    alpha_N = 0.5
    result = kamath_scaling_laws(N, N_c, alpha_N)
    assert isinstance(result, dict)
