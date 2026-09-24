"""Tests for kmppok.kamath_ppo_rlhf_objective."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.kmppok import kamath_ppo_rlhf_objective


def test_kmppok_basic():
    """Test basic functionality with valid log-probabilities."""
    rng = np.random.default_rng(42)
    n = 100
    rewards = rng.normal(0, 1, n)
    # log-probabilities must be <= 0, so use negative values
    logp_theta = -rng.uniform(0, 5, n)
    logp_ref = -rng.uniform(0, 5, n)
    beta = 0.8
    result = kamath_ppo_rlhf_objective(rewards, logp_theta, logp_ref, beta)
    assert isinstance(result, dict)
    for key in ("estimate", "objective", "mean_reward", "kl_estimate",
                "penalty", "per_sample", "beta", "n", "method"):
        assert key in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["objective"])
    assert math.isfinite(result["mean_reward"])
    assert math.isfinite(result["kl_estimate"])
    assert math.isfinite(result["penalty"])
    assert result["n"] == n
    assert result["beta"] == beta
    assert result["method"] == "PPO-RLHF objective E[r] - beta * E[log pi/pi_ref]"
    # estimate and objective are the same scalar per the return statement
    assert result["estimate"] == result["objective"]
    # penalty = beta * kl_estimate
    assert abs(result["penalty"] - beta * result["kl_estimate"]) < 1e-12
    # per_sample length matches n
    assert len(result["per_sample"]) == n


def test_kmppok_edge():
    """Test edge case: beta=0 collapses the objective to the mean reward."""
    rng = np.random.default_rng(42)
    n = 50
    rewards = rng.normal(0, 1, n)
    logp_theta = -rng.uniform(0, 5, n)
    logp_ref = -rng.uniform(0, 5, n)
    beta = 0.0
    result = kamath_ppo_rlhf_objective(rewards, logp_theta, logp_ref, beta)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    # With beta=0, the penalty vanishes and J equals the mean reward.
    assert abs(result["estimate"] - result["mean_reward"]) < 1e-12
    assert abs(result["penalty"]) < 1e-12
    assert result["beta"] == 0.0
    assert result["n"] == n


# --- appended: the module's own worked example as a gate -----------
import doctest as _doctest

import morie.fn.kmppok as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
