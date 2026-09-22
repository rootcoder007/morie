"""Tests for f_overall_r2.f_overall_r2."""

from morie.fn import _array_core as np

from morie.fn.f_overall_r2 import f_overall_r2


def test_ca2e17_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r2 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) ** 2
    n = x.size
    k = 3
    result = f_overall_r2(r2, n, k)
    assert isinstance(result, dict)
    assert "value" in result
    expected = r2 * (n - k - 1) / ((1 - r2) * k)
    assert result["value"] == expected


def test_ca2e17_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    r2 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) ** 2
    n = x.size
    k = 5
    result = f_overall_r2(r2, n, k)
    assert isinstance(result, dict)
    assert "value" in result
    assert "method" in result
