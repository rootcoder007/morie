"""Tests for kmstgn.kamath_summarize_from_feedback."""

import math

from morie.fn import _array_core as np

from morie.fn.kmstgn import kamath_summarize_from_feedback


def test_kmstgn_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 40

    # Build preference pairs (score_chosen, score_rejected).
    chosen_scores = rng.normal(0.0, 1.0, n)
    rejected_scores = rng.normal(0.0, 1.0, n)
    preferences = [(chosen_scores[i] + 1.0, rejected_scores[i])
                   for i in range(n)]

    rewards = rng.normal(0.0, 1.0, n)
    pi_logprobs = rng.normal(-2.0, 0.5, n)
    ref_logprobs = rng.normal(-2.0, 0.5, n)
    beta = 0.8

    result = kamath_summarize_from_feedback(
        preferences, rewards, pi_logprobs, ref_logprobs, beta)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "loss_rm" in result
    assert "rm_accuracy" in result
    assert "objective" in result
    assert "mean_reward" in result
    assert "kl_estimate" in result
    assert "beta" in result
    assert "n_preferences" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["loss_rm"])
    assert math.isfinite(result["objective"])
    assert math.isfinite(result["mean_reward"])
    assert math.isfinite(result["kl_estimate"])
    assert 0.0 <= result["rm_accuracy"] <= 1.0
    assert result["n_preferences"] == n
    assert result["n"] == n
    assert result["beta"] == 0.8


def test_kmstgn_edge():
    """Test with a small valid input drawn from the docstring example."""
    preferences = [(2.0, 1.0), (3.0, 0.0)]
    rewards = [1.0, 3.0]
    pi_logprobs = [math.log(0.5), math.log(0.5)]
    ref_logprobs = [math.log(0.25), math.log(0.5)]
    beta = 2.0

    result = kamath_summarize_from_feedback(
        preferences, rewards, pi_logprobs, ref_logprobs, beta)

    assert isinstance(result, dict)
    assert "loss_rm" in result
    assert "objective" in result
    assert "rm_accuracy" in result
    assert "mean_reward" in result
    assert math.isfinite(result["loss_rm"])
    assert math.isfinite(result["objective"])
    assert math.isfinite(result["mean_reward"])
    assert math.isfinite(result["rm_accuracy"])
    assert 0.0 <= result["rm_accuracy"] <= 1.0
    assert result["n_preferences"] == 2
    assert result["n"] == 2
    assert result["beta"] == 2.0


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
