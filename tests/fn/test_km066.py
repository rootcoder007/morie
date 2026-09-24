"""Tests for km066.kamath_ch5_reward_kl_penalty."""

import math

from morie.fn import _array_core as np

from morie.fn.km066 import kamath_ch5_reward_kl_penalty


def test_km066_basic():
    """Test basic functionality with array probabilities and a scalar reward."""
    x = np.random.default_rng(44).normal(0, 1, 50)
    y = np.random.default_rng(45).normal(0, 1, 50)
    pi_RL = np.random.default_rng(42).uniform(0.1, 0.9, 50)
    pi_SFT = np.random.default_rng(43).uniform(0.1, 0.9, 50)
    beta = 0.5
    result = kamath_ch5_reward_kl_penalty(x, y, pi_RL, pi_SFT, beta,
                                          r_theta=1.0)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "penalised_reward" in result
    assert "raw_reward" in result
    assert "penalty" in result
    assert "beta" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == 50
    assert result["raw_reward"] == 1.0
    assert result["beta"] == 0.5
    assert len(result["penalised_reward"]) == 50
    assert len(result["penalty"]) == 50


def test_km066_edge():
    """Test edge case: beta=0 yields the raw reward with zero penalty."""
    x = np.random.default_rng(44).normal(0, 1, 50)
    y = np.random.default_rng(45).normal(0, 1, 50)
    pi_RL = np.random.default_rng(42).uniform(0.1, 0.9, 50)
    pi_SFT = np.random.default_rng(43).uniform(0.1, 0.9, 50)
    result = kamath_ch5_reward_kl_penalty(x, y, pi_RL, pi_SFT, 0.0,
                                          r_theta=2.5)
    assert isinstance(result, dict)
    assert result["raw_reward"] == 2.5
    assert result["beta"] == 0.0
    assert math.isfinite(result["estimate"])
    assert result["estimate"] == 2.5
    assert all(p == 0.0 for p in result["penalty"])
    assert all(pr == 2.5 for pr in result["penalised_reward"])
