# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tabular Double Q-learning."""

from . import _array_core as np

from ._richresult import RichResult
from .mdpval import _mdp_args
from .qlearn import _greedy, _sample_row

__all__ = ["ddqn", "double_q_learning"]

_METHOD = "Tabular Double Q-learning (decoupled selection and evaluation)"


def double_q_learning(P, R, gamma, alpha=0.1, epsilon=0.1, n_episodes=100,
                      start=0, terminal=(), max_steps=1000, seed=0):
    r"""
    Tabular Double Q-learning on an explicit finite MDP.

    Maintains two tables and, with probability 0.5 per step, applies

        Q1(S,A) <- Q1(S,A) + alpha [ R + gamma Q2(S2, argmax_a Q1(S2,a))
                                     - Q1(S,A) ]

    (else the same update with Q1 and Q2 swapped), behaviour actions
    being epsilon-greedy in Q1 + Q2.  This is the boxed algorithm
    "Double Q-learning" of Sutton and Barto (2018), Section 6.7,
    p. 136, eq. (6.10); the estimator and its no-overestimation
    rationale are van Hasselt (2010).  Decoupling action selection
    (argmax under one estimate) from evaluation (the other estimate) is
    the same decomposition that van Hasselt, Guez and Silver (2016)
    apply to deep Q-networks as Double DQN, their eq. (4).

    Determinism conventions (mirrored bit-exactly in the R arm): each
    step consumes, in order, one uniform for the epsilon test, one for
    the uniform-random action when exploring (floor of u times A), one
    for the next-state draw (inverse CDF on the transition row), and
    one for the coin (heads = update Q1 when u < 0.5).  Greedy ties
    break to the lowest action index.

    Parameters
    ----------
    P : sequence of A matrices, each (S, S)
        Transition probabilities ``P[a][s, s2] = P(s2 | s, a)``.
    R : array-like (S, A), or sequence of A matrices (S, S)
        Expected one-step rewards r(s, a).
    gamma : float
        Discount factor in [0, 1].
    alpha : float
        Step size in (0, 1].
    epsilon : float
        Exploration probability.
    n_episodes : int
        Number of episodes.
    start : int
        0-based start state.
    terminal : sequence of int
        0-based absorbing terminal states.
    max_steps : int
        Step cap per episode.
    seed : int
        SplitMix64 seed.

    Returns
    -------
    result : dict
        Keys: ``estimate`` ((S, A) average of the two tables), ``q1``,
        ``q2``, ``policy`` (greedy on the average, 0-based), ``v``,
        ``n_steps``, ``n_episodes``, ``method``.

    References
    ----------
    Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An
    Introduction, 2nd ed., MIT Press, Section 6.7, boxed algorithm
    p. 136, eq. (6.10).  Local source:
    fetched-wave3/sutton-barto-2018-reinforcement-learning-2nd-ed.pdf.
    van Hasselt, H. (2010). Double Q-learning. Advances in Neural
    Information Processing Systems 23, 2613-2621.  Local source:
    fetched-wave3/hasselt-2010-double-qlearning-neurips.pdf.
    van Hasselt, H., Guez, A. and Silver, D. (2016). Deep reinforcement
    learning with Double Q-learning. AAAI 2016 (arXiv:1509.06461),
    eq. (4).  Local source:
    fetched-wave3/hasselt-guez-silver-2016-ddqn-arxiv1509.06461.pdf.
    """
    Pm, R, S, A = _mdp_args(P, R)
    gamma = float(gamma)
    alpha = float(alpha)
    epsilon = float(epsilon)
    start = int(start)
    term = set(int(t) for t in terminal)
    if start < 0 or start >= S:
        raise ValueError("start out of range")
    rng = np.random.default_rng(seed)
    Q1 = np.zeros((S, A))
    Q2 = np.zeros((S, A))
    n_steps = 0
    for _ in range(int(n_episodes)):
        s = start
        for _ in range(int(max_steps)):
            if s in term:
                break
            u1 = float(rng.uniform())
            if u1 < epsilon:
                a = int(float(rng.uniform()) * A)
                if a >= A:
                    a = A - 1
            else:
                qs = [float(Q1[s, j]) + float(Q2[s, j]) for j in range(A)]
                a = 0
                for j in range(1, A):
                    if qs[j] > qs[a]:
                        a = j
            u3 = float(rng.uniform())
            s2 = _sample_row(Pm[a][s], S, u3)
            r = float(R[s, a])
            coin = float(rng.uniform())
            if coin < 0.5:
                nxt = 0.0
                if s2 not in term:
                    nxt = float(Q2[s2, _greedy(Q1[s2], A)])
                Q1[s, a] = float(Q1[s, a]) + alpha * (
                    r + gamma * nxt - float(Q1[s, a]))
            else:
                nxt = 0.0
                if s2 not in term:
                    nxt = float(Q1[s2, _greedy(Q2[s2], A)])
                Q2[s, a] = float(Q2[s, a]) + alpha * (
                    r + gamma * nxt - float(Q2[s, a]))
            n_steps += 1
            s = s2
    Q = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            Q[s, a] = 0.5 * (float(Q1[s, a]) + float(Q2[s, a]))
    pol = np.zeros(S)
    V = np.zeros(S)
    for s in range(S):
        b = _greedy(Q[s], A)
        pol[s] = float(b)
        V[s] = float(Q[s, b])
    return RichResult(payload={
        "estimate": Q,
        "q1": Q1,
        "q2": Q2,
        "policy": pol,
        "v": V,
        "n_steps": n_steps,
        "n_episodes": int(n_episodes),
        "method": _METHOD,
    })


ddqn = double_q_learning


def cheatsheet():
    return "ddqn(P, R, gamma, ...) -> tabular Double Q-learning (Sutton-Barto 2018 Sec 6.7; van Hasselt 2010)."

# public names resolved by fn/_lazy_map.json
double_dqn = double_q_learning
doubledqn = double_q_learning
