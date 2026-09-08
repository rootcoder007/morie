"""Tests for qlearn.q_learning.

Anchor: exactly hand-computed Q tables after small numbers of episodes
on deterministic chains (Sutton-Barto 2018 eq. 6.8 applied by hand),
plus convergence to the exact dynamic-programming Q* on a small MDP.
"""

from morie.fn.mdpval import mdpval
from morie.fn.qlearn import qlearn


def test_qlearn_three_episode_hand_value():
    # single action, 0 -> terminal 1, r = 1, alpha = 0.5, gamma = 0.9:
    # Q <- Q + 0.5 (1 - Q): 0 -> 0.5 -> 0.75 -> 0.875 after 3 episodes.
    P = [[[0.0, 1.0], [0.0, 1.0]]]
    R = [[1.0], [0.0]]
    r = qlearn(P, R, 0.9, alpha=0.5, epsilon=0.0, n_episodes=3,
               start=0, terminal=[1], seed=0)
    assert abs(r["estimate"][0, 0] - 0.875) < 1e-15
    assert r["n_steps"] == 3


def test_qlearn_two_step_chain_hand_table():
    # deterministic chain 0 -> 1 -> 2(terminal); rewards r(0)=0, r(1)=1;
    # alpha = 0.5, gamma = 0.5, greedy (single action). Hand updates:
    # ep1: Q0 = 0 + .5(0 + .5*0 - 0) = 0;      Q1 = .5(1) = .5
    # ep2: Q0 = .5(0 + .5*.5) = .125;          Q1 = .5 + .5(1 - .5) = .75
    P = [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
    R = [[0.0], [1.0], [0.0]]
    r = qlearn(P, R, 0.5, alpha=0.5, epsilon=0.0, n_episodes=2,
               start=0, terminal=[2], seed=0)
    assert abs(r["estimate"][0, 0] - 0.125) < 1e-15
    assert abs(r["estimate"][1, 0] - 0.75) < 1e-15


def test_qlearn_converges_to_dp_qstar():
    # 2 states, 2 actions, deterministic transitions, terminal state 1;
    # enough greedy-plus-exploration episodes with decaying-free small
    # alpha drive Q near the exact DP solution.
    P = [
        [[0.0, 1.0], [0.0, 1.0]],   # action 0: leave to terminal
        [[1.0, 0.0], [0.0, 1.0]],   # action 1: stay in 0
    ]
    R = [[1.0, 0.2], [0.0, 0.0]]
    exact = mdpval(P, R, 0.9, tol=1e-14)
    r = qlearn(P, R, 0.9, alpha=0.02, epsilon=0.5, n_episodes=4000,
               start=0, terminal=[1], max_steps=50, seed=11)
    for a in range(2):
        assert abs(r["estimate"][0, a] - exact["q"][0, a]) < 0.05
    assert r["policy"][0] == exact["policy"][0]


def test_qlearn_terminal_rows_stay_zero():
    P = [[[0.0, 1.0], [0.0, 1.0]]]
    R = [[1.0], [0.0]]
    r = qlearn(P, R, 0.9, alpha=0.5, epsilon=0.3, n_episodes=10,
               start=0, terminal=[1], seed=3)
    assert r["estimate"][1, 0] == 0.0
