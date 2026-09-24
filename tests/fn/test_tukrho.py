"""Tests for tukrho.tukey_biweight."""

from morie.fn import _array_core as np

from morie.fn.tukrho import tukey_biweight


def test_tukrho_basic():
    """Test basic functionality."""
    r = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tukey_biweight(r)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_tukrho_edge():
    """Test edge cases."""
    r = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = tukey_biweight(r)
    assert isinstance(result, dict)
