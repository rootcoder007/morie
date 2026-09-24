"""Tests for kmstgn.kamath_summarize_from_feedback."""

from morie.fn import _array_core as np

from morie.fn.kmstgn import kamath_summarize_from_feedback


def test_kmstgn_basic():
    """Test basic functionality."""
    preferences = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    pi_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    ref_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_summarize_from_feedback(preferences, rewards, pi_logprobs, ref_logprobs, beta)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_kmstgn_edge():
    """Test edge cases."""
    preferences = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    pi_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    ref_logprobs = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_summarize_from_feedback(preferences, rewards, pi_logprobs, ref_logprobs, beta)
    assert isinstance(result, dict)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmstgn as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
