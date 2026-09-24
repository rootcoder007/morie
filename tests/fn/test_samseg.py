"""Tests for samseg.sam_segment."""

import pytest

from morie.fn import _array_core as np
from morie.fn.samseg import sam_segment


class _MaskDecoder:
    """Minimal mask-decoder stand-in: callable with (image_embedding, prompt_tokens, multimask)."""

    def __call__(self, image_embedding, prompt_tokens, multimask):
        if multimask:
            return np.zeros((3, 8, 8))
        return np.zeros((8, 8))


def test_samseg_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    image = rng.normal(0, 1, (16, 16))
    prompts = rng.normal(0, 1, (3, 2))
    decoder = _MaskDecoder()
    result = sam_segment(image, prompts, decoder)
    assert isinstance(result, dict)
    assert "mask" in result or "masks" in result


def test_samseg_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    image = rng.normal(0, 1, (8, 8))
    prompts = rng.normal(0, 1, (1, 2))
    decoder = _MaskDecoder()
    result = sam_segment(image, prompts, decoder)
    assert isinstance(result, dict)
    assert len(result) > 0
