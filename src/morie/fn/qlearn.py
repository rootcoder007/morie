# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tabular Q-learning (off-policy TD control)."""

from . import _array_core as np

from ._richresult import RichResult
from .mdpval import _mdp_args

__all__ = ["qlearn", "q_learning"]

_METHOD = "Tabular Q-learning, epsilon-greedy off-policy TD control"


def _greedy(Qs, A):
    b = 0
    for a in range(1, A):
        if Qs[a] > Qs[b]:
            b = a
    return b


def _sample_row(row, S, u):
    # Inverse-CDF draw on a stochastic row: one uniform, cumulative scan,
    # exactly the convention of _array_core Generator.choice with p=
    # (mirrored bit-exactly by .ghc_choice_p in the R arm).
    c = 0.0
    for s2 in range(S):
        c += float(row[s2])
        if u <= c:
            return s2
    return S - 1


def q_learning(P, R, gamma, alpha=0.1, epsilon=0.1, n_episodes=100,
               start=0, terminal=(), max_steps=1000, seed=0, Q0=None):
    r"""
    Tabular Q-learning on an explicit finite MDP.

    Runs the boxed algorithm "Q-learning (off-policy TD control)" of
    Sutton and Barto (2018), Section 6.5, p. 131, whose update is their
    eq. (6.8):

        Q(S, A) <- Q(S, A) + alpha [ R + gamma max_a Q(S2, a) - Q(S, A) ]

    with behaviour actions chosen epsilon-greedily from Q.  Convergence
    of the iterate to q* (given infinite visits and stochastic-
    approximation step sizes) is the theorem of Watkins and Dayan
    (1992).  Q(terminal, .) is fixed at 0 and max_a Q(S2, a) is 0 when
    S2 is terminal, per the box's initialization line.

    Determinism conventions (mirrored bit-exactly in the R arm): the
    greedy argmax breaks ties to the lowest action index; each step
    consumes from the SplitMix64 stream, in order, (1) one uniform for
    the epsilon test, (2) one uniform for the uniform-random action if
    exploring (a = floor(u * A)), (3) one uniform for the next-state
    draw by inverse CDF on the P row.  With ``epsilon = 0`` and
    deterministic transition rows the trajectory is fully deterministic
    and the Q table after N steps can be computed by hand.

    Parameters
    ----------
    P : sequence of A matrices, each (S, S)
        Transition probabilities ``P[a][s, s2] = P(s2 | s, a)``.
    R : array-like (S, A), or sequence of A matrices (S, S)
        Expected one-step rewards r(s, a) (per-transition rewards are
        averaged under P).
    gamma : float
        Discount factor in [0, 1].
    alpha : float
        Step size in (0, 1].
    epsilon : float
        Exploration probability in [0, 1].
    n_episodes : int
        Number of episodes.
    start : int
        0-based start state of every episode.
    terminal : sequence of int
        0-based terminal (absorbing) states; entering one ends the
        episode.  Empty means episodes end after ``max_steps`` steps.
    max_steps : int
        Step cap per episode.
    seed : int
        SplitMix64 seed for the behaviour policy and transitions.
    Q0 : array-like (S, A), optional
        Initial action values (default zeros; terminal rows forced 0).

    Returns
    -------
    result : dict
        Keys: ``estimate`` ((S, A) learned Q table), ``policy`` (greedy
        0-based action per state), ``v`` (greedy state values),
        ``n_steps`` (total updates), ``n_episodes``, ``method``.

    References
    ----------
    Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An
    Introduction, 2nd ed., MIT Press, Section 6.5, boxed algorithm
    p. 131, eq. (6.8).  Local source:
    fetched-wave3/sutton-barto-2018-reinforcement-learning-2nd-ed.pdf.
    Watkins, C. J. C. H. and Dayan, P. (1992). Q-learning. Machine
    Learning 8, 279-292.  Local source:
    fetched-wave3/watkins-dayan-1992-qlearning-ML8.pdf.
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
    Q = np.zeros((S, A))
    if Q0 is not None:
        Q0 = np.asarray(Q0, dtype=float)
        for s in range(S):
            for a in range(A):
                Q[s, a] = 0.0 if s in term else float(Q0[s, a])
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
                a = _greedy(Q[s], A)
            u3 = float(rng.uniform())
            s2 = _sample_row(Pm[a][s], S, u3)
            r = float(R[s, a])
            nxt = 0.0
            if s2 not in term:
                nxt = float(Q[s2, _greedy(Q[s2], A)])
            Q[s, a] = float(Q[s, a]) + alpha * (r + gamma * nxt - float(Q[s, a]))
            n_steps += 1
            s = s2
    pol = np.zeros(S)
    V = np.zeros(S)
    for s in range(S):
        b = _greedy(Q[s], A)
        pol[s] = float(b)
        V[s] = float(Q[s, b])
    return RichResult(payload={
        "estimate": Q,
        "policy": pol,
        "v": V,
        "n_steps": n_steps,
        "n_episodes": int(n_episodes),
        "method": _METHOD,
    })


qlearn = q_learning


def cheatsheet():
    return "qlearn(P, R, gamma, alpha, epsilon, n_episodes) -> tabular Q-learning per Sutton-Barto 2018 eq 6.8."
