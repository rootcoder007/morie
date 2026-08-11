"""Tests for ddqn.double_q_learning.

Anchors: closed-form per-table updates on a one-step chain (each table
obeys Q <- 1 - (1 - Q)/2 when its coin comes up, so Q = 1 - 2^-k after
k updates), with the coin sequence replayed from the SplitMix64
stream; and convergence of both tables to the exact DP Q*.
"""

from morie.fn import _array_core as np
from morie.fn.ddqn import ddqn
from morie.fn.mdpval import mdpval

P1 = [[[0.0, 1.0], [0.0, 1.0]]]
R1 = [[1.0], [0.0]]


def _coins(seed, n):
    """Replay the stream: per greedy step, draws are (eps, next, coin)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rng.uniform()          # epsilon test (epsilon = 0, still consumed)
        rng.uniform()          # next-state draw
        out.append(float(rng.uniform()) < 0.5)
    return out


def test_ddqn_closed_form_table_updates():
    n = 6
    coins = _coins(2, n)
    h = sum(1 for c in coins if c)
    t = n - h
    r = ddqn(P1, R1, 0.9, alpha=0.5, epsilon=0.0, n_episodes=n,
             start=0, terminal=[1], seed=2)
    # gamma * next is always 0 (terminal), so each update of a table is
    # Q <- Q + 0.5 (1 - Q), i.e. Q = 1 - 2^-k after k updates.
    assert abs(r["q1"][0, 0] - (1.0 - 0.5 ** h)) < 1e-15
    assert abs(r["q2"][0, 0] - (1.0 - 0.5 ** t)) < 1e-15
    assert abs(r["estimate"][0, 0]
               - 0.5 * (r["q1"][0, 0] + r["q2"][0, 0])) < 1e-16
    assert r["n_steps"] == n


def test_ddqn_converges_to_dp_qstar():
    P = [
        [[0.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ]
    R = [[1.0, 0.2], [0.0, 0.0]]
    exact = mdpval(P, R, 0.9, tol=1e-14)
    r = ddqn(P, R, 0.9, alpha=0.02, epsilon=0.5, n_episodes=8000,
             start=0, terminal=[1], max_steps=50, seed=7)
    for a in range(2):
        assert abs(r["estimate"][0, a] - exact["q"][0, a]) < 0.1
    assert r["policy"][0] == exact["policy"][0]


def test_ddqn_terminal_rows_stay_zero():
    r = ddqn(P1, R1, 0.9, alpha=0.5, epsilon=0.3, n_episodes=10,
             start=0, terminal=[1], seed=3)
    assert r["q1"][1, 0] == 0.0 and r["q2"][1, 0] == 0.0
