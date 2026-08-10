# SPDX-License-Identifier: AGPL-3.0-or-later
"""Value iteration for a finite MDP."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["mdpval", "mdp_value_iteration"]

_METHOD = "Value iteration (Bellman optimality sweeps) for a finite MDP"


def _mdp_args(P, R):
    # P: length-A sequence of (S, S) row-stochastic matrices, so that
    # P[a][s, s2] = P(s2 | s, a).  R: (S, A) expected one-step reward,
    # or a length-A sequence of (S, S) per-transition rewards reduced by
    # r(s, a) = sum_s2 P(s2|s,a) R[a][s, s2].
    Pm = [np.asarray(Pa, dtype=float) for Pa in P]
    A = len(Pm)
    if A < 1 or Pm[0].ndim != 2 or Pm[0].shape[0] != Pm[0].shape[1]:
        raise ValueError("P must be a sequence of square (S, S) matrices")
    S = Pm[0].shape[0]
    for a in range(A):
        if Pm[a].shape != (S, S):
            raise ValueError("P[%d] is not (S, S)" % a)
        for s in range(S):
            if abs(float(np.sum(Pm[a][s])) - 1.0) > 1e-8:
                raise ValueError("P[%d] row %d does not sum to 1" % (a, s))
    if isinstance(R, (list, tuple)) and len(R) == A and \
            np.asarray(R[0], dtype=float).ndim == 2:
        Rsa = np.zeros((S, A))
        for a in range(A):
            Ra = np.asarray(R[a], dtype=float)
            for s in range(S):
                Rsa[s, a] = float(np.sum(Pm[a][s] * Ra[s]))
        R = Rsa
    else:
        R = np.asarray(R, dtype=float)
        if R.shape != (S, A):
            raise ValueError("R must be (S, A) or a length-A list of (S, S)")
    return Pm, R, S, A


def mdp_value_iteration(P, R, gamma, tol=1e-10, max_iter=100000, V0=None):
    r"""
    Value iteration for a finite Markov decision process.

    Repeats the Bellman optimality sweep

        V(s) <- max_a sum_{s2} P(s2|s,a) [ r(s,a,s2) + gamma V(s2) ]

    until the largest change in a sweep, Delta = max_s |v - V(s)|, drops
    below ``tol``, then outputs the deterministic greedy policy
    pi(s) = argmax_a sum_{s2} P(s2|s,a) [ r + gamma V(s2) ] with ties to
    the lowest action index.  This is the boxed algorithm "Value
    Iteration, for estimating pi ~ pi*" in Sutton and Barto (2018),
    Section 4.4, p. 83 (their eq. 4.10), which turns the Bellman
    optimality equation (their eq. 4.1) into an update rule; the method
    originates with Bellman (1957) and is treated as value iteration in
    Puterman (1994), Section 6.3.

    Parameters
    ----------
    P : sequence of A matrices, each (S, S)
        Transition probabilities ``P[a][s, s2] = P(s2 | s, a)``; every
        row must sum to 1.  Make a terminal state absorbing (self-loop
        with reward 0).
    R : array-like (S, A), or sequence of A matrices (S, S)
        Expected one-step reward r(s, a), or per-transition rewards
        averaged under P.
    gamma : float
        Discount factor in [0, 1] (1 only sensible with absorbing
        zero-reward terminal states).
    tol : float
        Sweep-change threshold (theta in the source box).
    max_iter : int
        Hard cap on sweeps.
    V0 : array-like of shape (S,), optional
        Initial value function (defaults to zeros).

    Returns
    -------
    result : dict
        Keys: ``estimate`` (V*, length S), ``policy`` (greedy 0-based
        action per state), ``q`` ((S, A) state-action values),
        ``n_iter``, ``delta``, ``converged``, ``method``.

    References
    ----------
    Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An
    Introduction, 2nd ed., MIT Press, Section 4.4, boxed algorithm
    p. 83, eq. (4.10).  Local source:
    fetched-wave3/sutton-barto-2018-reinforcement-learning-2nd-ed.pdf.
    Bellman, R. (1957). Dynamic Programming, Princeton University Press.
    Puterman, M. L. (1994). Markov Decision Processes: Discrete
    Stochastic Dynamic Programming, Wiley, Section 6.3.
    """
    Pm, R, S, A = _mdp_args(P, R)
    gamma = float(gamma)
    tol = float(tol)
    V = np.zeros(S)
    if V0 is not None:
        V0 = np.asarray(V0, dtype=float)
        for s in range(S):
            V[s] = float(V0[s])
    delta = float("inf")
    it = 0
    while it < int(max_iter):
        it += 1
        delta = 0.0
        for s in range(S):
            v = float(V[s])
            best = -float("inf")
            for a in range(A):
                q = float(R[s, a]) + gamma * float(np.sum(Pm[a][s] * V))
                if q > best:
                    best = q
            V[s] = best
            d = abs(v - best)
            if d > delta:
                delta = d
        if delta < tol:
            break
    Q = np.zeros((S, A))
    pol = np.zeros(S)
    for s in range(S):
        for a in range(A):
            Q[s, a] = float(R[s, a]) + gamma * float(np.sum(Pm[a][s] * V))
        b = 0
        for a in range(1, A):
            if Q[s, a] > Q[s, b]:
                b = a
        pol[s] = float(b)
    return RichResult(payload={
        "estimate": V,
        "policy": pol,
        "q": Q,
        "n_iter": it,
        "delta": delta,
        "converged": bool(delta < tol),
        "method": _METHOD,
    })


mdpval = mdp_value_iteration


def cheatsheet():
    return "mdpval(P, R, gamma) -> V*, greedy policy by Bellman optimality sweeps (Sutton-Barto 2018 Sec 4.4)."
