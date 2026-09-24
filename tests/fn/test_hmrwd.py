"""Tests for hmrwd.geron_reward_function."""

import math

from morie.fn import _array_core as np

from morie.fn.hmrwd import geron_reward_function


def test_hmrwd_basic():
    """Test basic functionality with a callable R and scalar transition."""
    s = 0
    a = 1
    s_next = 1
    R = lambda s, a, sp: 1.0 if sp == 1 else 0.0
    result = geron_reward_function(s, a, s_next, R=R, gamma=0.9)
    assert isinstance(result, dict)
    for key in ("rewards", "total_reward", "returns", "discounted_return", "estimate", "n", "method"):
        assert key in result
    # Scalar transition -> single reward
    assert len(result["rewards"]) == 1
    assert result["n"] == 1
    assert math.isfinite(result["total_reward"])
    assert math.isfinite(result["discounted_return"])


def test_hmrwd_edge():
    """Test edge cases with a small trajectory and a callable R using a table."""
    s = [0, 1, 1]
    a = [0, 0, 1]
    s_next = [1, 0, 1]

    table = [[[0.0, 1.0], [2.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]]

    def R(s, a, sp):
        return table[s][a][sp]

    result = geron_reward_function(s, a, s_next, R=R, gamma=0.5)
    assert isinstance(result, dict)
    for key in ("rewards", "total_reward", "returns", "discounted_return", "estimate", "n", "method"):
        assert key in result
    assert len(result["rewards"]) == 3
    assert result["n"] == 3
    assert math.isfinite(result["discounted_return"])
    assert math.isfinite(result["total_reward"])
