"""Tests for hmseq2.geron_seq2seq."""

import math

from morie.fn import _array_core as np

from morie.fn.hmseq2 import geron_seq2seq


def test_hmseq2_basic():
    """Test basic functionality."""
    src = [1.0, 2.0, 3.0, 4.0]
    tgt = [0, 1, 2]

    def encoder(s):
        return [float(sum(s))]

    def decoder(z, prefix):
        return [0.0, 0.0, 0.0]

    result = geron_seq2seq(src, tgt, encoder, decoder)
    assert isinstance(result, dict)
    expected_keys = {"z", "loss", "perplexity", "greedy", "token_logprobs",
                     "exposure_bias", "estimate", "n", "method"}
    assert expected_keys.issubset(set(result.keys()))
    assert math.isfinite(float(result["loss"]))
    assert math.isfinite(float(result["perplexity"]))
    assert math.isfinite(float(result["exposure_bias"]))
    assert 0.0 <= float(result["perplexity"])
    assert len(list(result["greedy"])) == len(tgt)
    assert math.isfinite(float(result["estimate"]))
    assert int(result["n"]) == len(tgt)


def test_hmseq2_edge():
    """Test edge cases."""
    src = [1.0, 2.0]
    tgt = [1, 1]

    def encoder(s):
        return [float(sum(s))]

    def decoder(z, prefix):
        return [0.0, 20.0, 0.0]

    result = geron_seq2seq(src, tgt, encoder, decoder)
    assert isinstance(result, dict)
    expected_keys = {"z", "loss", "perplexity", "greedy", "token_logprobs",
                     "exposure_bias", "estimate", "n", "method"}
    assert expected_keys.issubset(set(result.keys()))
    assert math.isfinite(float(result["loss"]))
    assert float(result["loss"]) < 1.0
    assert math.isfinite(float(result["perplexity"]))
    assert 0.0 <= float(result["perplexity"])
    assert len(list(result["greedy"])) == len(tgt)
    assert int(result["n"]) == len(tgt)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmseq2 as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
