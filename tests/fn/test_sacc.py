"""Tests for sacc (soft policy iteration on a finite MDP).

Replaces the generated stub, which imported ``sac``.
"""

import math

from morie.fn.sacc import soft_policy_iteration


def _two_state():
    # state 0: action 0 stays, action 1 moves to state 1 (the good one)
    # state 1 is absorbing and pays 1 per step
    P = [[[1.0, 0.0], [0.0, 1.0]],
         [[0.0, 1.0], [0.0, 1.0]]]
    R = [[0.0, 0.0], [1.0, 1.0]]
    return P, R


def test_a_low_temperature_policy_is_almost_greedy():
    P, R = _two_state()
    res = soft_policy_iteration(P, R, gamma=0.9, temp=0.01)
    assert res["policy"][0][1] > 0.99          # move to the good state
    assert res["converged"]


def test_a_high_temperature_policy_is_almost_uniform():
    P, R = _two_state()
    res = soft_policy_iteration(P, R, gamma=0.9, temp=100.0)
    assert abs(res["policy"][0][0] - 0.5) < 0.05


def test_the_policy_rows_are_distributions():
    P, R = _two_state()
    res = soft_policy_iteration(P, R, gamma=0.9, temp=1.0)
    for row in res["policy"]:
        assert abs(sum(row) - 1.0) < 1e-9
        assert all(v >= 0 for v in row)


def test_the_q_values_order_the_actions_correctly():
    P, R = _two_state()
    res = soft_policy_iteration(P, R, gamma=0.9, temp=0.1)
    assert res["q"][0][1] > res["q"][0][0]


def test_entropy_rises_with_temperature():
    P, R = _two_state()
    cold = soft_policy_iteration(P, R, gamma=0.9, temp=0.05)["entropy"]
    hot = soft_policy_iteration(P, R, gamma=0.9, temp=50.0)["entropy"]
    assert sum(hot) > sum(cold)
    assert max(hot) <= math.log(2.0) + 1e-9    # two actions


def test_a_zero_discount_looks_only_at_the_immediate_reward():
    P, R = _two_state()
    res = soft_policy_iteration(P, R, gamma=0.0, temp=0.01)
    # from state 0 both actions pay 0 now, so the policy is indifferent
    assert abs(res["policy"][0][0] - 0.5) < 0.05


def test_validation():
    P, R = _two_state()
    try:
        soft_policy_iteration(P, R, gamma=0.9, temp=0.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
