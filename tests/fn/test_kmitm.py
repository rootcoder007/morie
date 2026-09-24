"""Tests for kmitm.kamath_image_text_matching."""

import math

import pytest
from morie.fn import _array_core as np

from morie.fn.kmitm import kamath_image_text_matching


def test_kmitm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    d = 4
    image_emb = rng.normal(0, 1, d)
    text_emb = rng.normal(0, 1, d)
    W = rng.normal(0, 1, 2 * d)
    b = 0.5
    result = kamath_image_text_matching(image_emb, text_emb, W, b)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert 0.0 <= result["estimate"] <= 1.0
    assert math.isfinite(result["logit"])
    assert result["n"] == 2 * d
    assert result["fusion"] == "concatenation [I; T]"
    assert isinstance(result["match"], bool)
    assert len(result["fused"]) == 2 * d


def test_kmitm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    d = 4
    image_emb = rng.normal(0, 1, d)
    text_emb = rng.normal(0, 1, d)
    W = rng.normal(0, 1, 2 * d)
    # When all weights are zero and bias is zero, sigmoid(0) = 0.5.
    zero_W = [0.0] * (2 * d)
    zero_result = kamath_image_text_matching(image_emb, text_emb, zero_W, 0.0)
    assert isinstance(zero_result, dict)
    assert math.isfinite(zero_result["estimate"])
    assert abs(zero_result["estimate"] - 0.5) < 1e-12
    assert zero_result["match"] is True
    # Caller-supplied fusion produces a fused width that W must match.
    def fuse(I, T):
        return [I[0] * T[0], I[0] + T[0]]
    custom_result = kamath_image_text_matching(image_emb, text_emb,
                                               [1.0, 1.0], 0.0,
                                               fuse=fuse)
    assert isinstance(custom_result, dict)
    assert custom_result["fusion"] == "caller-supplied fusion"
    assert custom_result["n"] == 2
    assert math.isfinite(custom_result["estimate"])
    # Empty embeddings raise (docstring: "both embeddings must be non-empty").
    with pytest.raises(ValueError):
        kamath_image_text_matching([], text_emb, W, 0.0)
    with pytest.raises(ValueError):
        kamath_image_text_matching(image_emb, [], W, 0.0)
    # W length mismatch raises (docstring: "W must match the fused width").
    with pytest.raises(ValueError):
        kamath_image_text_matching(image_emb, text_emb, [0.0], 0.0)
    # Non-callable fuse raises (docstring: "fuse must be callable").
    with pytest.raises(ValueError):
        kamath_image_text_matching(image_emb, text_emb, [0.0, 0.0], 0.0,
                                   fuse=0.0)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmitm as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
