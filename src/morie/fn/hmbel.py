# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bellman optimality equation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_bellman_optimality"]


def geron_bellman_optimality(V, P, R, gamma, tol=1e-10, max_iter=10000):
    """
    Bellman optimality equation, solved by value iteration.

    Formula: V*(s) = max_a [R(s,a) + gamma * sum_{s'} P(s'|s,a) V*(s')]

    Parameters
    ----------
    V : array-like, shape (S,)
        Initial value estimates (any starting point; the contraction is a
        fixed point regardless).
    P : array-like, shape (S, A, S)
        Transition kernel; each P[s, a, :] must be a probability vector.
    R : array-like, shape (S, A)
        Expected immediate reward for taking action a in state s.
    gamma : float
        Discount factor in [0, 1).
    tol : float
        Sup-norm convergence tolerance.
    max_iter : int
        Iteration cap.

    Returns
    -------
    result : RichResult
        Keys: V, policy, Q, iterations, residual, converged, estimate, n, method.

    Examples
    --------
    >>> r = geron_bellman_optimality([0.0], [[[1.0]]], [[1.0]], 0.5)
    >>> round(float(r["V"][0]), 9)
    2.0
    >>> P = [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]]
    >>> R = [[0.0, 1.0], [0.0, 0.0]]
    >>> r2 = geron_bellman_optimality([0.0, 0.0], P, R, 0.0)
    >>> [float(x) for x in r2["V"]], [int(a) for a in r2["policy"]]
    ([1.0, 0.0], [1, 0])

    References
    ----------
    Géron Ch 19
    """
    Vv = np.asarray(V, dtype=float).ravel().copy()
    Pm = np.asarray(P, dtype=float)
    Rm = np.asarray(R, dtype=float)
    if Pm.ndim != 3:
        raise ValueError(f"geron_bellman_optimality: P must be 3-D (S, A, S), got ndim={Pm.ndim}")
    S, A, S2 = Pm.shape
    if S == 0 or A == 0:
        raise ValueError("geron_bellman_optimality: P must have at least one state and one action")
    if S != S2:
        raise ValueError(f"geron_bellman_optimality: P must be square in the state axes, got {S} vs {S2}")
    if Rm.shape != (S, A):
        raise ValueError(f"geron_bellman_optimality: R must have shape {(S, A)}, got {Rm.shape}")
    if Vv.size != S:
        raise ValueError(f"geron_bellman_optimality: V has {Vv.size} entries but P has {S} states")
    if np.any(Pm < 0):
        raise ValueError("geron_bellman_optimality: P contains negative probabilities")
    rowsum = Pm.sum(axis=2)
    if not np.allclose(rowsum, 1.0, atol=1e-8):
        bad = np.argwhere(np.abs(rowsum - 1.0) > 1e-8)[0]
        raise ValueError(
            f"geron_bellman_optimality: P[{bad[0]}, {bad[1]}, :] sums to "
            f"{rowsum[bad[0], bad[1]]!r}, not 1"
        )
    g = float(gamma)
    if not (0.0 <= g < 1.0):
        raise ValueError(f"geron_bellman_optimality: gamma must lie in [0, 1), got {g}")

    residual = np.inf
    it = 0
    Q = Rm + g * (Pm @ Vv)
    for it in range(1, int(max_iter) + 1):
        Q = Rm + g * (Pm @ Vv)
        Vn = Q.max(axis=1)
        residual = float(np.max(np.abs(Vn - Vv)))
        Vv = Vn
        if residual <= tol:
            break
    Q = Rm + g * (Pm @ Vv)
    policy = Q.argmax(axis=1)

    return RichResult(
        title="Bellman optimality (value iteration)",
        summary_lines=[("States", S), ("Actions", A), ("Iterations", it), ("Sup-norm residual", residual)],
        payload={
            "V": Vv,
            "policy": policy,
            "Q": Q,
            "iterations": it,
            "residual": residual,
            "converged": bool(residual <= tol),
            "gamma": g,
            "estimate": float(np.max(Vv)),
            "n": int(S),
            "method": "Value iteration on the Bellman optimality operator",
        },
    )


def cheatsheet():
    return "hmbel: Bellman optimality equation"
