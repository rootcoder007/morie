"""Tests for sarsa.sarsa_control.

Anchor: exactly hand-computed updates (Sutton-Barto 2018, Sec 6.4 box,
applied by hand) and the expected-SARSA fixed point on a deterministic
chain, where greedy SARSA and Q-learning coincide.
"""

from morie.fn.qlearn import qlearn
from morie.fn.sarsa import sarsa


def test_sarsa_three_episode_hand_value():
    # single action, 0 -> terminal, r = 1, alpha = 0.5: 0.875 after 3.
    P = [[[0.0, 1.0], [0.0, 1.0]]]
    R = [[1.0], [0.0]]
    r = sarsa(P, R, 0.9, alpha=0.5, epsilon=0.0, n_episodes=3,
              start=0, terminal=[1], seed=0)
    assert abs(r["estimate"][0, 0] - 0.875) < 1e-15
    assert r["n_steps"] == 3


def test_sarsa_two_step_chain_hand_table():
    # chain 0 -> 1 -> 2(terminal), rewards 0 then 1, alpha=.5, gamma=.5.
    # SARSA with a single action produces the same numbers as Q-learning:
    # ep1: Q0 = .5(0 + .5 Q1) = 0, then Q1 = .5(1) = .5
    # ep2: Q0 = .5(.5*.5) = .125,  then Q1 = .5 + .5(1 - .5) = .75
    P = [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]
    R = [[0.0], [1.0], [0.0]]
    r = sarsa(P, R, 0.5, alpha=0.5, epsilon=0.0, n_episodes=2,
              start=0, terminal=[2], seed=0)
    assert abs(r["estimate"][0, 0] - 0.125) < 1e-15
    assert abs(r["estimate"][1, 0] - 0.75) < 1e-15


def test_sarsa_greedy_matches_qlearning_on_single_action_mdp():
    # with one action the on-policy and off-policy targets coincide, so
    # the two algorithms must produce identical tables for equal seeds.
    P = [[[0.1, 0.6, 0.3], [0.0, 0.4, 0.6], [0.0, 0.0, 1.0]]]
    R = [[0.3], [-0.1], [0.0]]
    a = sarsa(P, R, 0.8, alpha=0.2, epsilon=0.0, n_episodes=30,
              start=0, terminal=[2], seed=5)
    b = qlearn(P, R, 0.8, alpha=0.2, epsilon=0.0, n_episodes=30,
               start=0, terminal=[2], seed=5)
    for s in range(3):
        assert abs(a["estimate"][s, 0] - b["estimate"][s, 0]) < 1e-15


def test_sarsa_terminal_rows_stay_zero():
    P = [[[0.0, 1.0], [0.0, 1.0]]]
    R = [[1.0], [0.0]]
    r = sarsa(P, R, 0.9, alpha=0.5, epsilon=0.3, n_episodes=10,
              start=0, terminal=[1], seed=3)
    assert r["estimate"][1, 0] == 0.0
