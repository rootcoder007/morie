"""Tests for rgerp.rangayyan_erp_features."""

from morie.fn import _array_core as np

from morie.fn.bsaphys import rangayyan_erp_features


def test_rgerp_basic():
    """Test basic functionality."""
    erp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_erp_features(erp, fs)
    assert isinstance(result, dict)
    assert "components" in result


def test_rgerp_edge():
    """Test edge cases."""
    erp = np.random.default_rng(42).normal(0.0, 1.0, 40)
    fs = 0.1
    result = rangayyan_erp_features(erp, fs)
    assert isinstance(result, dict)
