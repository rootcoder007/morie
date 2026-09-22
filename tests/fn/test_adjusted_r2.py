"""Tests for adjusted_r2.adjusted_r2."""

from morie.fn import _array_core as np

from morie.fn.adjusted_r2 import adjusted_r2


def test_ca2e15_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    r2 = 0.30
    n = 100
    k = 3
    result = adjusted_r2(r2, n, k)
    assert isinstance(result, dict)
    assert "value" in result
    expected = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    assert abs(result["value"] - expected) < 1e-12


def test_ca2e15_edge():
    """Test edge cases."""
    r2 = 0.0
    n = 50
    k = 1
    result = adjusted_r2(r2, n, k)
    assert isinstance(result, dict)
    assert "value" in result
    expected = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    assert abs(result["value"] - expected) < 1e-12
