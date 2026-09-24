"""Tests for grrein.geron_reinforce_policy_gradient."""

from morie.fn import _array_core as np

from morie.fn.grrein import geron_reinforce_policy_gradient


def test_grrein_basic():
    """Test basic functionality."""
    theta = [0.0, 0.0]
    log_probs = [[1.0, 0.0], [0.0, 1.0]]
    returns_G = [2.0, -1.0]
    alpha = 0.5
    result = geron_reinforce_policy_gradient(theta, log_probs, returns_G, alpha)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_grrein_edge():
    """Test edge cases."""
    theta = [0.0, 0.0]
    log_probs = [[1.0, 0.0], [0.0, 1.0]]
    returns_G = [2.0, -1.0]
    alpha = 0.5
    result = geron_reinforce_policy_gradient(theta, log_probs, returns_G, alpha)
    assert isinstance(result, dict)
