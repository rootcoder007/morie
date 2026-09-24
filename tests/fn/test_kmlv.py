"""Tests for kmlv.kamath_llava_visual_instruction."""

import math

from morie.fn import _array_core as np

from morie.fn.kmlv import kamath_llava_visual_instruction


def test_kmlv_basic():
    """Test basic assembly of visual tokens and text tokens."""
    rng = np.random.default_rng(42)
    d, d_v = 3, 4
    n_patches, n_text = 2, 3

    image = "img"  # placeholder; the encoder does not have to consume it
    W = rng.normal(0, 1, (d, d_v))
    feats = rng.normal(0, 1, (n_patches, d_v))
    visual_encoder = lambda im: feats
    text_tokens = rng.normal(0, 1, (n_text, d))

    result = kamath_llava_visual_instruction(
        image, W, visual_encoder, text_tokens)

    # The docstring / return payload advertises these keys
    assert "visual_tokens" in result
    assert "inputs" in result
    assert "n_visual" in result
    assert "n_text" in result
    assert "d_model" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    # Counts and dim are fully determined by the inputs
    assert result["n_visual"] == n_patches
    assert result["n_text"] == n_text
    assert result["d_model"] == d
    assert result["n"] == n_patches + n_text

    # visual_tokens must carry every patch in the LLM's token space
    assert len(result["visual_tokens"]) == n_patches
    assert len(result["visual_tokens"][0]) == d

    # inputs stack visual tokens in front of the text tokens
    assert len(result["inputs"]) == n_patches + n_text
    assert len(result["inputs"][0]) == d

    assert isinstance(result["method"], str)


def test_kmlv_edge():
    """Test with a causal-LM head: loss, perplexity, n_response_tokens."""
    rng = np.random.default_rng(7)
    d, d_v = 3, 4
    n_patches, n_text = 2, 3
    vocab_size = 5
    ignore_index = -100

    image = "img"
    W = rng.normal(0, 1, (d, d_v))
    feats = rng.normal(0, 1, (n_patches, d_v))
    visual_encoder = lambda im: feats
    text_tokens = rng.normal(0, 1, (n_text, d))
    lm_head = lambda x: rng.normal(0, 1, (x.shape[0], vocab_size))
    targets = np.array([ignore_index, ignore_index, 0, 1, 2])

    result = kamath_llava_visual_instruction(
        image, W, visual_encoder, text_tokens,
        lm_head=lm_head, targets=targets, ignore_index=ignore_index)

    assert "loss" in result
    assert "perplexity" in result
    assert "n_response_tokens" in result

    assert math.isfinite(result["loss"])
    assert math.isfinite(result["perplexity"])
    assert result["loss"] >= 0.0
    assert result["perplexity"] >= 1.0

    assert isinstance(result["n_response_tokens"], int)
    assert result["n_response_tokens"] >= 0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmlv as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
