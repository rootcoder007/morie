"""Tests for km065.kamath_ch5_reward_loss_pairwise."""

import math

from morie.fn import _array_core as np

from morie.fn.km065 import kamath_ch5_reward_loss_pairwise


def test_km065_basic():
    """Test basic functionality with a callable reward model."""
    def r_theta(x, y):
        return 1.0 if y == "good" else 0.0

    x = ["p1", "p2", "p3", "p4", "p5", "p6"]
    y_0 = ["good", "bad", "good", "bad", "good", "bad"]
    y_1 = ["bad", "good", "bad", "good", "bad", "good"]
    i = [0, 1, 0, 1, 0, 1]

    result = kamath_ch5_reward_loss_pairwise(r_theta, x, y_0, y_1, i)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "margins" in result
    assert "per_pair" in result
    assert "n" in result
    assert "method" in result
    assert result["n"] == 6
    assert math.isfinite(result["estimate"])
    assert len(result["margins"]) == 6
    assert len(result["per_pair"]) == 6


def test_km065_edge():
    """Test scale invariance: adding a constant to every reward leaves the loss unchanged."""
    def r_theta(x, y):
        return 2.0 if y == "good" else -1.0

    def r_shifted(x, y):
        return r_theta(x, y) + 7.5

    x = ["a", "b", "c", "d", "f"]
    y_0 = ["good", "bad", "good", "bad", "good"]
    y_1 = ["bad", "good", "bad", "good", "bad"]
    i = [0, 1, 0, 1, 0]

    result1 = kamath_ch5_reward_loss_pairwise(r_theta, x, y_0, y_1, i)
    result2 = kamath_ch5_reward_loss_pairwise(r_shifted, x, y_0, y_1, i)

    assert math.isfinite(result1["estimate"])
    assert math.isfinite(result2["estimate"])
    assert abs(result1["estimate"] - result2["estimate"]) < 1e-12
    assert result1["margins"] == result2["margins"]
