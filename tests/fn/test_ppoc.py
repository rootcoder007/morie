"""Tests for ppoc.ppo."""

from morie.fn import _array_core as np
from morie.fn.ppoc import ppo

import math


def test_ppoc_basic():
    """Test basic functionality with random advantages and ratios."""
    rng = np.random.default_rng(42)
    env = rng.normal(0, 1, 100)
    policy = rng.uniform(0.5, 1.5, 100)
    result = ppo(env, policy, 0.2)
    assert isinstance(result, dict)
    assert "estimate" in result


def test_ppoc_edge():
    """Test edge case with small input and log probabilities."""
    rng = np.random.default_rng(123)
    env = rng.normal(0, 1, 5)
    logp_new = rng.normal(0, 1, 5)
    logp_old = rng.normal(0, 1, 5)
    result = ppo(env, logp_new=logp_new, logp_old=logp_old)
    assert isinstance(result, dict)
    assert "estimate" in result
