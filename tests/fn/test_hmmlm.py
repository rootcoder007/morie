"""Tests for hmmlm.geron_masked_lm."""

import math

from morie.fn import _array_core as np

from morie.fn.hmmlm import geron_masked_lm


def test_hmmlm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    vocab_size = 10
    n = 100
    X = rng.integers(0, vocab_size, n)
    result = geron_masked_lm(X, mask_frac=0.15, vocab_size=vocab_size, seed=42)
    assert isinstance(result, dict)
    # Keys returned by geron_masked_lm (RichResult)
    for key in ("loss", "baseline_loss", "perplexity", "masked_positions",
                "targets", "probabilities", "n_masked", "estimate", "n",
                "method"):
        assert key in result
    # Numerical values must be finite
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["baseline_loss"])
    assert math.isfinite(result["perplexity"])
    # Shape sanity
    assert result["n"] == n
    assert result["n_masked"] == int(n * 0.15)
    assert len(result["masked_positions"]) == result["n_masked"]
    assert len(result["targets"]) == result["n_masked"]


def test_hmmlm_edge():
    """Test edge cases."""
    # Reproduce the docstring's deterministic example: short repeated
    # sequence with seed=0 yields exactly 3 masked positions out of 20.
    X = [0, 1, 2, 3] * 5
    result = geron_masked_lm(X, mask_frac=0.15, seed=0)
    assert isinstance(result, dict)
    assert result["n"] == 20
    assert result["n_masked"] == 3
    # All masked positions must lie inside the sequence and be distinct
    positions = list(result["masked_positions"])
    assert len(positions) == 3
    assert len(set(positions)) == 3
    assert all(0 <= p < 20 for p in positions)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmmlm as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
