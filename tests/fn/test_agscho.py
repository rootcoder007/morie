"""Tests for agscho.alphazero_search_horizon."""

from morie.fn import _array_core as np

from morie.fn.agscho import alphazero_search_horizon


def test_agscho_basic():
    """Test basic functionality."""
    depth_limit = 4
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    state = "root_state"
    gamma = 0.5
    k_start = 0
    result = alphazero_search_horizon(
        depth_limit, state, rewards=rewards, values=values, gamma=gamma, k_start=k_start
    )
    # Independent recomputation of the documented formula:
    l = depth_limit
    g = gamma
    kk = k_start
    part = 0.0
    tau = 0
    while kk + tau < l:
        part += (g ** tau) * rewards[kk + tau]
        tau += 1
    idx = l if l < len(values) else len(values) - 1
    boot = (g ** (l - kk)) * values[idx]
    expected_estimate = part + boot
    expected_reward_part = part
    expected_bootstrap = boot
    expected_depth = l

    assert "estimate" in result
    assert "bootstrap" in result
    assert "reward_part" in result
    assert "depth" in result
    assert result["estimate"] == expected_estimate
    assert result["reward_part"] == expected_reward_part
    assert result["bootstrap"] == expected_bootstrap
    assert result["depth"] == expected_depth
    assert result["state"] == state


def test_agscho_edge():
    """Test edge cases: bootstrap fallback when values is None."""
    depth_limit = 3
    rewards = [1.0, 1.0, 1.0]
    state = "root_state"
    gamma = 1.0
    k_start = 1
    result = alphazero_search_horizon(
        depth_limit, state, rewards=rewards, values=None, gamma=gamma, k_start=k_start
    )
    # Independent recomputation: no bootstrap since values is None.
    l = depth_limit
    g = gamma
    kk = k_start
    part = 0.0
    tau = 0
    while kk + tau < l:
        part += (g ** tau) * rewards[kk + tau]
        tau += 1
    expected_reward_part = part
    expected_bootstrap = 0.0
    expected_estimate = part
    expected_depth = l

    assert "estimate" in result
    assert result["estimate"] == expected_estimate
    assert result["bootstrap"] == expected_bootstrap
    assert result["reward_part"] == expected_reward_part
    assert result["depth"] == expected_depth
