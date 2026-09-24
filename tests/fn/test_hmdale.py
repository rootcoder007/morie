"""Tests for hmdale.geron_dalle."""

import math

from morie.fn import _array_core as np

from morie.fn.hmdale import geron_dalle


def test_hmdale_basic():
    """Test basic functionality."""
    text = [0, 1]
    model = lambda ctx: np.zeros(2)
    result = geron_dalle(text, model, n_image_tokens=4)
    assert isinstance(result, dict)
    for key in ("image_tokens", "token_grid", "log_likelihood",
                "token_logprobs", "perplexity", "context",
                "n_steps", "estimate", "n", "method"):
        assert key in result
    assert len(result["image_tokens"]) == 4
    assert result["n_steps"] == 4
    assert math.isfinite(result["log_likelihood"])
    assert math.isfinite(result["perplexity"])
    assert result["perplexity"] > 0


def test_hmdale_edge():
    """Test edge cases with a custom grid shape."""
    text = [0]
    model = lambda ctx: np.array([0.0, 5.0])
    result = geron_dalle(text, model, n_image_tokens=6, grid=(2, 3))
    assert isinstance(result, dict)
    assert len(result["image_tokens"]) == 6
    assert result["n_steps"] == 6
    assert len(result["token_grid"]) == 2
    assert len(result["token_grid"][0]) == 3
    assert math.isfinite(result["log_likelihood"])
    assert math.isfinite(result["perplexity"])


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.hmdale as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
