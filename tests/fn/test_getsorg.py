"""Tests for getsorg.getis_ord_g."""

from morie.fn import _array_core as np
import math

from morie.fn.getsorg import getis_ord_g


def test_getsorg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40
    x = np.abs(rng.normal(0, 1, n))
    W = rng.normal(0, 1, (n, n))
    result = getis_ord_g(x, W)
    assert isinstance(result, dict)
    for key in ("estimate", "statistic", "p_value", "expected", "var", "S0", "S1", "S2"):
        assert math.isfinite(result[key])
    assert result["n"] == n


def test_getsorg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n = 40
    x = np.abs(rng.normal(0, 1, n))
    W = rng.normal(0, 1, (n, n))
    result = getis_ord_g(x, W)
    assert "method" in result
    assert 0.0 <= result["p_value"] <= 1.0
    assert math.isfinite(result["estimate"])
