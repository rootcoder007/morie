"""Tests for kmperp.kamath_perplexity."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.kmperp import kamath_perplexity


def test_kmperp_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    p = list(rng.uniform(0.001, 1.0, 100))
    log_probs = [math.log(x) for x in p]
    result = kamath_perplexity(log_probs)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "perplexity" in result
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 1.0
    assert result["n"] == 100


def test_kmperp_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    # empty sequence should raise
    with pytest.raises(ValueError):
        kamath_perplexity([])

    # positive log-probabilities should raise
    bad = list(rng.uniform(0.1, 1.0, 5))
    with pytest.raises(ValueError):
        kamath_perplexity(bad)

    # base='2' should work and report base='2'
    p = list(rng.uniform(0.001, 1.0, 20))
    log_probs = [math.log(x) for x in p]
    result = kamath_perplexity(log_probs, base="2")
    assert result["base"] == "2"
    assert math.isfinite(result["estimate"])
    assert result["estimate"] >= 1.0


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmperp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
