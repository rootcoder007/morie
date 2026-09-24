"""Tests for llavx.llava_visual_chat."""

from morie.fn import _array_core as np

from morie.fn.llavx import llava_visual_chat


def test_llavx_basic():
    """Test basic functionality."""
    visual_tokens = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    text_embeddings = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = llava_visual_chat(visual_tokens, text_embeddings)
    assert isinstance(result, dict)
    assert "estimate" in result or "estimate" in result


def test_llavx_edge():
    """Test edge cases."""
    visual_tokens = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    text_embeddings = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    result = llava_visual_chat(visual_tokens, text_embeddings)
    assert isinstance(result, dict)
