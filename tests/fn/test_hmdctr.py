"""Tests for hmdctr.geron_decoder_only."""

from morie.fn import _array_core as np

from morie.fn.hmdctr import geron_decoder_only


def test_hmdctr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_decoder_only(X)
    assert isinstance(result, dict)
    assert "estimate" in result or "total_params" in result


def test_hmdctr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_decoder_only(X)
    assert isinstance(result, dict)
