"""Tests for grppo.geron_ppo_clipped_objective."""

from morie.fn import _array_core as np

from morie.fn.grppo import geron_ppo_clipped_objective


def test_grppo_basic():
    """Test basic functionality."""
    ratios = [1.5, 0.5]
    advantages = [1.0, -1.0]
    result = geron_ppo_clipped_objective(ratios, advantages)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grppo_edge():
    """Test edge cases."""
    ratios = [1.5, 0.5]
    advantages = [1.0, -1.0]
    result = geron_ppo_clipped_objective(ratios, advantages)
    assert isinstance(result, dict)
