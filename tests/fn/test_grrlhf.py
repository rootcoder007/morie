"""Tests for grrlhf.geron_rlhf_reward_kl_objective."""

import math

from morie.fn import _array_core as np

from morie.fn.grrlhf import geron_rlhf_reward_kl_objective


def test_grrlhf_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 100
    rewards = rng.normal(0, 1, n)
    # Log-probabilities must be non-positive.
    policy_logprobs = -np.abs(rng.normal(0, 1, n))
    ref_logprobs = -np.abs(rng.normal(0, 1, n))
    beta = 0.8
    result = geron_rlhf_reward_kl_objective(rewards, policy_logprobs, ref_logprobs, beta)
    assert isinstance(result, dict)
    assert "objective" in result
    assert "mean_reward" in result
    assert "kl" in result
    assert "per_sample" in result
    assert "kl_terms" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["objective"])
    assert math.isfinite(result["mean_reward"])
    assert math.isfinite(result["kl"])
    assert math.isfinite(result["estimate"])
    # objective = mean_reward - beta * kl
    assert abs(result["objective"] - (result["mean_reward"] - beta * result["kl"])) < 1e-10
    # per_sample and kl_terms should have one entry per sample.
    assert len(result["per_sample"]) == n
    assert len(result["kl_terms"]) == n
    # n reports the sample size.
    assert result["n"] == n


def test_grrlhf_edge():
    """Test edge cases."""
    # n = 1 with policy identical to reference: kl must be exactly 0.
    result = geron_rlhf_reward_kl_objective([1.0], [-1.0], [-1.0], beta=0.5)
    assert isinstance(result, dict)
    assert "objective" in result
    assert "mean_reward" in result
    assert "kl" in result
    assert result["kl"] == 0.0
    # With zero KL the KL penalty vanishes: objective == mean_reward.
    assert result["objective"] == result["mean_reward"]


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.grrlhf as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
