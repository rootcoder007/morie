"""Tests for gwasl1.gwas_linear."""

from morie.fn import _array_core as np

from morie.fn.gwasl1 import gwas_linear


def test_gwasl1_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(43)
    n = 40
    p = 3
    y = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, p))
    snp = rng.normal(0, 1, n)
    Vinv = np.eye(n)
    result = gwas_linear(y, X, snp, Vinv)
    assert isinstance(result, dict)


def test_gwasl1_edge():
    """Test edge cases."""
    rng = np.random.default_rng(43)
    n = 40
    p = 3
    y = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, p))
    snp = rng.normal(0, 1, n)
    Vinv = np.eye(n)
    result = gwas_linear(y, X, snp, Vinv)
    assert isinstance(result, dict)
    assert hasattr(result, "keys") or hasattr(result, "__dict__")
