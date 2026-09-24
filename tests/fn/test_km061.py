"""Tests for km061.kamath_ch4_krona_output."""

from morie.fn import _array_core as np

from morie.fn.km061 import kamath_ch4_krona_output


def test_km061_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, p = 40, 2
    X = rng.normal(0, 1, (n, p))
    # W is a 2D matrix (p, q); A_k (x) B_k must have shape (p, q)
    W = rng.normal(0, 1, (p, 1))
    A_k = rng.normal(0, 1, (p, 1))
    B_k = rng.normal(0, 1, (1, 1))
    s = 0.5
    result = kamath_ch4_krona_output(X, W, A_k, B_k, s)
    assert isinstance(result, dict)
    assert "Y" in result
    assert "base" in result
    assert "adapter_term" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == n
    assert len(result["Y"]) == n
    assert len(result["Y"][0]) == 1


def test_km061_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, p = 40, 4
    X = rng.normal(0, 1, (n, p))
    # A_k (2, 2) (x) B_k (2, 1) = (4, 2); W must be (4, 2)
    A_k = rng.normal(0, 1, (2, 2))
    B_k = rng.normal(0, 1, (2, 1))
    W = rng.normal(0, 1, (p, 2))
    s = 1.0
    result = kamath_ch4_krona_output(X, W, A_k, B_k, s)
    assert isinstance(result, dict)
    assert "Y" in result
    assert "s" in result
    assert result["n"] == n
    assert len(result["Y"]) == n
    assert len(result["Y"][0]) == 2
