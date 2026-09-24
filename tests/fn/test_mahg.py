"""Tests for mahg.ma_hedges_g."""

import math

from morie.fn import _array_core as np

from morie.fn.mahg import ma_hedges_g


def test_mahg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m1, m2 = 1.5, 0.7
    s1, s2 = 1.1, 0.9
    n1, n2 = 30, 35
    result = ma_hedges_g(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
    for key in (
        "estimate",
        "d",
        "J",
        "J_approx",
        "se",
        "variance",
        "df",
        "s_pooled",
        "n",
        "method",
    ):
        assert key in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["se"])
    assert result["se"] >= 0.0
    assert result["n"] == int(n1 + n2)
    assert result["df"] == float(n1 + n2 - 2)


def test_mahg_edge():
    """Test edge cases with balanced small samples."""
    rng = np.random.default_rng(42)
    s1, s2 = 0.5, 0.5
    m1, m2 = 0.0, 0.0
    n1, n2 = 5, 5
    result = ma_hedges_g(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
    for key in ("estimate", "J", "J_approx", "df", "s_pooled", "variance", "se"):
        assert key in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["J"])
    assert result["estimate"] == 0.0
    assert 0.0 < result["J"] <= 1.0
    assert result["df"] == 8.0
