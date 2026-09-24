"""Tests for gtruncwt.truncate_weights."""

from morie.fn import _array_core as np

from morie.fn.gtruncwt import truncate_weights


def test_gtruncwt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(45)
    weights = np.abs(rng.normal(1.0, 0.5, 100))
    quantile = 0.9
    result = truncate_weights(weights, quantile)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_gtruncwt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(45)
    weights = np.abs(rng.normal(1.0, 0.5, 100))
    quantile = 0.5
    result = truncate_weights(weights, quantile)
    assert isinstance(result, dict)
    assert len(result) > 0
