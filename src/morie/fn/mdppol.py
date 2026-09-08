# SPDX-License-Identifier: AGPL-3.0-or-later
"""Policy iteration for a finite MDP."""

from . import _array_core as np

from ._richresult import RichResult
from .mdpval import _mdp_args

__all__ = ["mdppol", "mdp_policy_iteration"]

_METHOD = "Policy iteration (iterative policy evaluation + greedy improvement)"


def mdp_policy_iteration(P, R, gamma, tol=1e-12, max_eval=100000,
                         max_improve=1000, pi0=None):
    r"""
    Howard-style policy iteration for a finite Markov decision process.

    Alternates (2) iterative policy evaluation -- sweeps of

        V(s) <- sum_{s2} P(s2|s,pi(s)) [ r + gamma V(s2) ]

    until the largest change is below ``tol`` -- with (3) greedy policy
    improvement pi(s) <- argmax_a sum_{s2} P(s2|s,a)[ r + gamma V(s2) ],
    stopping when the policy is stable.  This is the boxed algorithm
    "Policy Iteration (using iterative policy evaluation)" in Sutton and
    Barto (2018), Section 4.3, p. 80; policy iteration itself is due to
    Howard (1960).  Ties in the argmax go to the lowest action index,
    and a state counts as stable when the incumbent action value is
    within 1e-12 of the maximum, which removes the tie-flipping
    non-termination noted in Sutton and Barto Exercise 4.4.

    Parameters
    ----------
    P : sequence of A matrices, each (S, S)
        Transition probabilities ``P[a][s, s2] = P(s2 | s, a)``.
    R : array-like (S, A), or sequence of A matrices (S, S)
        Expected one-step rewards r(s, a), or per-transition rewards
        averaged under P.
    gamma : float
        Discount factor in [0, 1).
    tol : float
        Policy-evaluation sweep threshold (theta in the source box).
    max_eval : int
        Cap on evaluation sweeps per improvement round.
    max_improve : int
        Cap on improvement rounds.
    pi0 : array-like of shape (S,), optional
        Initial deterministic policy (0-based actions; default all 0).

    Returns
    -------
    result : dict
        Keys: ``estimate`` (V for the final policy), ``policy``
        (0-based optimal actions), ``q`` ((S, A) action values under
        the final V), ``n_improve``, ``n_eval``, ``policy_stable``,
        ``method``.

    References
    ----------
    Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An
    Introduction, 2nd ed., MIT Press, Section 4.3, boxed algorithm
    p. 80.  Local source:
    fetched-wave3/sutton-barto-2018-reinforcement-learning-2nd-ed.pdf.
    Howard, R. A. (1960). Dynamic Programming and Markov Processes,
    MIT Press.
    """
    Pm, R, S, A = _mdp_args(P, R)
    gamma = float(gamma)
    tol = float(tol)
    pol = [0] * S
    if pi0 is not None:
        pi0 = np.asarray(pi0, dtype=float)
        for s in range(S):
            a = int(pi0[s])
            if a < 0 or a >= A:
                raise ValueError("pi0[%d] out of range" % s)
            pol[s] = a
    V = np.zeros(S)
    n_eval = 0
    stable = False
    rounds = 0
    for rounds in range(1, int(max_improve) + 1):
        # 2. iterative policy evaluation (in place, sweep order s = 0..S-1)
        for _ in range(int(max_eval)):
            n_eval += 1
            delta = 0.0
            for s in range(S):
                v = float(V[s])
                a = pol[s]
                Vs = float(R[s, a]) + gamma * float(np.sum(Pm[a][s] * V))
                V[s] = Vs
                d = abs(v - Vs)
                if d > delta:
                    delta = d
            if delta < tol:
                break
        # 3. policy improvement
        stable = True
        for s in range(S):
            old = pol[s]
            qs = [float(R[s, a]) + gamma * float(np.sum(Pm[a][s] * V))
                  for a in range(A)]
            b = 0
            for a in range(1, A):
                if qs[a] > qs[b]:
                    b = a
            if qs[b] > qs[old] + 1e-12:
                pol[s] = b
                stable = False
        if stable:
            break
    Q = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            Q[s, a] = float(R[s, a]) + gamma * float(np.sum(Pm[a][s] * V))
    return RichResult(payload={
        "estimate": V,
        "policy": np.asarray([float(a) for a in pol]),
        "q": Q,
        "n_improve": rounds,
        "n_eval": n_eval,
        "policy_stable": bool(stable),
        "method": _METHOD,
    })


mdppol = mdp_policy_iteration


def cheatsheet():
    return "mdppol(P, R, gamma) -> optimal policy/V by Howard policy iteration (Sutton-Barto 2018 Sec 4.3)."
