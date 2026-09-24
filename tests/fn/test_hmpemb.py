"""Tests for hmpemb.geron_pretrained_embeddings."""

import math

from morie.fn.hmpemb import geron_pretrained_embeddings


def test_hmpemb_basic():
    """Test basic functionality."""
    vocab = ["cat", "dog", "bird", "zzz", "xxx"]
    pretrained = {
        "cat": [1.0, 0.0, 0.0],
        "dog": [0.0, 1.0, 0.0],
        "bird": [0.0, 0.0, 1.0],
    }
    result = geron_pretrained_embeddings(vocab, pretrained, freeze=True, seed=42)
    assert isinstance(result, dict)
    for key in ("embeddings", "coverage", "oov", "oov_indices", "dim",
                "trainable", "n_parameters", "estimate", "n", "method"):
        assert key in result
    assert math.isclose(float(result["coverage"]), 0.6)
    assert int(result["n"]) == len(vocab)
    assert int(result["dim"]) == 3
    assert "zzz" in result["oov"]
    assert "xxx" in result["oov"]
    assert int(result["trainable"]) == 0
    assert int(result["n_parameters"]) == 15
    embeddings = result["embeddings"]
    assert len(embeddings) == len(vocab)
    assert len(embeddings[0]) == 3


def test_hmpemb_edge():
    """Test edge cases."""
    vocab = ["cat", "dog"]
    pretrained = {"cat": [1.0, 0.0]}
    result = geron_pretrained_embeddings(vocab, pretrained, freeze=False, seed=42)
    assert isinstance(result, dict)
    assert int(result["trainable"]) == 4
    assert math.isclose(float(result["coverage"]), 0.5)
    assert int(result["dim"]) == 2
    assert "dog" in result["oov"]
    embeddings = result["embeddings"]
    assert len(embeddings) == len(vocab)
    assert len(embeddings[0]) == 2


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmpemb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
