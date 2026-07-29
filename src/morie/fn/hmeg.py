# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Epsilon-greedy exploration strategy."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_epsilon_greedy"]


def geron_epsilon_greedy(Q, s, epsilon, seed=0):
    """
    Epsilon-greedy exploration strategy.

    Formula: a = argmax_a Q(s,a) w.p. 1-eps; random w.p. eps

    The full action distribution is returned, not just a draw:
    ``eps/A`` on every action plus ``(1 - eps)/|argmax|`` shared among the
    greedy ones, so ties are handled explicitly rather than by numpy's
    argmax tie-break. The sampled action uses a deterministic LCG seeded
    by ``seed``, so the same call always gives the same action.

    Parameters
    ----------
    Q : array-like, shape (S, A) or (A,)
        Action-value table.
    s : int
        State index (ignored if ``Q`` is 1-D).
    epsilon : float
        Exploration rate in [0, 1].
    seed : int, default 0
        Seed for the deterministic sampler.

    Returns
    -------
    result : RichResult
        Keys: action, probabilities, greedy_action, greedy_actions,
        q_values, is_exploratory, estimate, n, method.

    Examples
    --------
    >>> r = geron_epsilon_greedy([[1.0, 5.0, 2.0]], s=0, epsilon=0.3)
    >>> [round(p, 12) for p in r["probabilities"]]
    [0.1, 0.8, 0.1]
    >>> r["greedy_action"]
    1
    >>> geron_epsilon_greedy([[1.0, 5.0]], 0, 0.0)["probabilities"]
    [0.0, 1.0]
    >>> geron_epsilon_greedy([[1.0, 5.0]], 0, 0.0)["action"]
    1
    >>> [round(p, 12) for p in geron_epsilon_greedy([[2.0, 2.0]], 0, 0.5)["probabilities"]]
    [0.5, 0.5]

    References
    ----------
    Géron Ch 19
    """
    Qa = np.asarray(Q, dtype=float)
    if Qa.ndim == 1:
        Qa = Qa[None, :]
    if Qa.ndim != 2 or Qa.size == 0:
        raise ValueError(f"geron_epsilon_greedy: Q must be a non-empty 1-D or 2-D table, got shape {Qa.shape}")
    if not np.all(np.isfinite(Qa)):
        raise ValueError("geron_epsilon_greedy: Q contains non-finite values")
    si = int(s)
    if not (0 <= si < Qa.shape[0]):
        raise ValueError(f"geron_epsilon_greedy: state {si} out of range for {Qa.shape[0]} states")
    eps = float(epsilon)
    if not (0.0 <= eps <= 1.0):
        raise ValueError(f"geron_epsilon_greedy: epsilon must lie in [0, 1], got {epsilon!r}")

    q = Qa[si]
    A = q.size
    best = np.flatnonzero(q == q.max())
    p = np.full(A, eps / A)
    p[best] += (1.0 - eps) / best.size

    st = (int(seed) * 1664525 + 1013904223) % 2**32
    u = (st + 0.5) / 2**32
    a = int(np.searchsorted(np.cumsum(p), u, side="right"))
    a = min(a, A - 1)

    return RichResult(
        title="Epsilon-greedy policy",
        summary_lines=[("epsilon", eps), ("Action", a), ("Greedy action", int(best[0]))],
        interpretation="Every action keeps probability at least eps/A, which is what guarantees continued exploration.",
        payload={
            "action": a,
            "probabilities": p.tolist(),
            "greedy_action": int(best[0]),
            "greedy_actions": best.tolist(),
            "q_values": q.tolist(),
            "is_exploratory": bool(a not in set(best.tolist())),
            "epsilon": eps,
            "estimate": float(p.max()),
            "n": int(A),
            "method": "epsilon-greedy action distribution with deterministic LCG sampling",
        },
    )


def cheatsheet():
    return "hmeg: Epsilon-greedy exploration strategy"
