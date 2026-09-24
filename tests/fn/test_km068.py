"""Tests for km068.kamath_ch5_ppo_loss."""

import math
import pytest
from morie.fn import _array_core as np

from morie.fn.km068 import kamath_ch5_ppo_loss


def test_km068_basic():
    """Test basic functionality."""
    phi = [[0.5, 0.5], [0.3, 0.7]]
    x = ["p1", "p2"]
    y = [["a", "b"], ["c", "d"]]

    def r_theta(xi, yi):
        return 1.0 if yi == "a" else 0.5

    beta = 0.8
    pi_ref = [[0.5, 0.5], [0.5, 0.5]]

    result = kamath_ch5_ppo_loss(phi, x, y, r_theta, beta, pi_ref=pi_ref)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "per_prompt_objective" in result
    assert "kl" in result
    assert "expected_reward" in result
    assert "beta" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert result["n"] == 2
    assert result["beta"] == beta
    assert len(result["per_prompt_objective"]) == 2
    assert len(result["kl"]) == 2
    assert len(result["expected_reward"]) == 2


def test_km068_edge():
    """Test edge cases: missing pi_ref raises ValueError."""
    phi = [[0.5, 0.5]]
    x = ["p1"]
    y = [["a", "b"]]

    def r_theta(xi, yi):
        return 1.0

    beta = 0.8

    with pytest.raises(ValueError):
        kamath_ch5_ppo_loss(phi, x, y, r_theta, beta)
