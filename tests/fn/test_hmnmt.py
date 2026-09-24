"""Tests for hmnmt.geron_encoder_decoder_nmt."""

import math

from morie.fn import _array_core as np
from morie.fn.hmnmt import geron_encoder_decoder_nmt


def test_hmnmt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    vocab_size = 8

    # Build a simple model mapping with fixed probabilities over the vocabulary
    probs = [0.1, 0.2, 0.3, 0.05, 0.15, 0.1, 0.05, 0.05]  # sums to 1
    model = {
        "encode": lambda s: len(s),
        "decode": lambda z, prefix: list(probs),
    }

    # Generate source and target token ids
    src = [int(x) for x in rng.integers(0, vocab_size, 10)]
    tgt = [int(x) for x in rng.integers(0, vocab_size, 10)]

    result = geron_encoder_decoder_nmt(src, tgt, model)

    assert isinstance(result, dict)
    # Keys named in the return statement
    for key in ["loss", "token_losses", "perplexity", "greedy",
                "exact_match", "z", "estimate", "n", "method"]:
        assert key in result

    # Structural properties
    assert math.isfinite(float(result["loss"]))
    assert math.isfinite(float(result["perplexity"]))
    assert float(result["perplexity"]) >= 1.0
    assert len(list(result["token_losses"])) == len(tgt)
    assert len(list(result["greedy"])) == len(tgt)
    assert int(result["n"]) == len(tgt)


def test_hmnmt_edge():
    """Test edge cases using the docstring's worked-example model."""
    model = {
        "encode": lambda s: len(s),
        "decode": lambda z, prefix: [0.2, 0.5, 0.3],
    }

    # tgt=[1,1] with probs [0.2,0.5,0.3]: each -log(0.5)=0.693147, sum=1.386294
    result = geron_encoder_decoder_nmt([9, 9], [1, 1], model)

    assert isinstance(result, dict)
    assert [round(float(v), 6) for v in result["token_losses"]] == [0.693147, 0.693147]
    assert round(float(result["loss"]), 6) == 1.386294
    assert round(float(result["perplexity"]), 6) == 2.0
    assert [int(t) for t in result["greedy"]] == [1, 1]
    assert bool(result["exact_match"]) is True


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmnmt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
