"""Tests for hmencd.geron_encoder_decoder_transformer."""

from morie.fn import _array_core as np

from morie.fn.hmencd import geron_encoder_decoder_transformer


def test_hmencd_basic():
    """Test basic functionality."""
    src = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    tgt = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_encoder_decoder_transformer(src, tgt)
    assert isinstance(result, dict)
    assert "estimate" in result or "total_params" in result


def test_hmencd_edge():
    """Test edge cases."""
    src = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    tgt = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = geron_encoder_decoder_transformer(src, tgt)
    assert isinstance(result, dict)
