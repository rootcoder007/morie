"""Tests for kminst.kamath_instruction_tuning_loss."""

import doctest as _doctest
import math

from morie.fn import _array_core as np

from morie.fn.kminst import kamath_instruction_tuning_loss
import morie.fn.kminst as _doctest_module


def test_kminst_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    T, V = 50, 10
    logits = rng.normal(0, 1, (T, V))
    response_mask = list(rng.integers(0, 2, T))
    if not any(response_mask):
        response_mask[0] = 1
    targets = list(rng.integers(0, V, T))
    result = kamath_instruction_tuning_loss(logits, response_mask, targets)
    assert "estimate" in result
    assert "loss" in result
    assert "perplexity" in result
    assert "n_response_tokens" in result
    assert "token_losses" in result
    assert "vocab_size" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["perplexity"])
    assert result["vocab_size"] == V
    assert result["n"] == T
    assert result["n_response_tokens"] == sum(response_mask)


def test_kminst_edge():
    """Test edge cases with uniform logits (matches docstring example)."""
    logits = np.zeros((3, 2))
    response_mask = [0, 1, 1]
    targets = [0, 1, 0]
    result = kamath_instruction_tuning_loss(logits, response_mask, targets)
    assert "estimate" in result
    assert abs(result["estimate"] - math.log(2)) < 1e-12
    assert abs(result["perplexity"] - 2.0) < 1e-12
    assert result["n_response_tokens"] == 2
    assert result["vocab_size"] == 2
    assert result["n"] == 3


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
