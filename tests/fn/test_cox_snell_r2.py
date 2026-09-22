"""Tests for cox_snell_r2.cox_snell_r2."""

from morie.fn import _array_core as np

from morie.fn.cox_snell_r2 import cox_snell_r2


def test_ca4e13_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    neg2ll_null = rng.normal(200, 1, 100).sum()
    neg2ll_full = rng.normal(150, 1, 100).sum()
    n = 100
    result = cox_snell_r2(neg2ll_null, neg2ll_full, n)
    assert isinstance(result, dict)
    assert "value" in result
    expected = 1 - np.exp(-(neg2ll_null - neg2ll_full) / n)
    assert np.isclose(result["value"], expected)


def test_ca4e13_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    neg2ll_null = rng.normal(200, 1, 50).sum()
    neg2ll_full = rng.normal(180, 1, 50).sum()
    n = 50
    result = cox_snell_r2(neg2ll_null, neg2ll_full, n)
    assert isinstance(result, dict)
    assert "value" in result
    expected = 1 - np.exp(-(neg2ll_null - neg2ll_full) / n)
    assert np.isclose(result["value"], expected)
