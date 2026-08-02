# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bellman optimality equation for Q*."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_bellman_optimality"]

_METHOD = "Q-value iteration (Bellman optimality)"


def geron_bellman_optimality(Q, transitions, rewards, gamma, max_iter=1000, tol=1e-10):
    r"""Iterate the Bellman optimality operator on Q until it stops moving.

    .. math::
        Q^*(s,a) = \sum_{s'} T(s,a,s')\bigl[R(s,a,s')
                 + \gamma \max_{a'} Q^*(s',a')\bigr]

    For :math:`\gamma < 1` the operator is a
    :math:`\gamma`-contraction in the sup norm, so this converges to the
    unique fixed point from any start, and the residual shrinks by at
    least a factor :math:`\gamma` per sweep.  With :math:`\gamma = 1` no
    such guarantee exists and the iteration may not terminate -- the
    residual is returned so the caller can tell.

    Parameters
    ----------
    Q : array-like, shape (S, A)
        Initial Q-table.
    transitions : array-like, shape (S, A, S)
        ``T[s, a, s']`` transition probabilities; each ``T[s, a, :]``
        must be non-negative and sum to 1 (or to 0 for an unavailable
        action, whose Q stays at ``-inf``-free zero).
    rewards : array-like, shape (S, A, S) or (S, A)
        Rewards. A 2-D array is read as ``R(s, a)`` and broadcast.
    gamma : float
        Discount factor in ``[0, 1]``.
    max_iter : int, optional
        Sweep cap, default 1000.
    tol : float, optional
        Sup-norm convergence tolerance, default ``1e-10``.

    Returns
    -------
    RichResult
        Payload keys ``Q`` (final table), ``V`` (state values),
        ``policy`` (greedy action per state), ``residual``,
        ``iterations``, ``converged``, ``estimate`` (mean state value),
        ``n``, ``method``.

    References
    ----------
    Géron Ch 19, Bellman Optimality Equation.

    Examples
    --------
    One state, one action, self-loop paying 1 per step at
    ``gamma = 0.5``: the geometric series gives ``1/(1-0.5) = 2``.

    >>> r = geron_bellman_optimality([[0.0]], [[[1.0]]], [[[1.0]]], 0.5)
    >>> round(r["Q"][0][0], 8)
    2.0
    >>> r["converged"]
    True

    Two actions, the second paying double, picks the second:

    >>> T = [[[1.0], [1.0]]]
    >>> R = [[[1.0], [2.0]]]
    >>> r2 = geron_bellman_optimality([[0.0, 0.0]], T, R, 0.5)
    >>> [round(q, 8) for q in r2["Q"][0]]
    [3.0, 4.0]
    >>> r2["policy"]
    [1]
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    T = np.asarray(transitions, dtype=float)
    R = np.asarray(rewards, dtype=float)
    if Q.ndim != 2 or Q.size == 0:
        raise ValueError(f"Q must be a non-empty 2-D (S, A) array, got shape {Q.shape}.")
    S, A = Q.shape
    if T.shape != (S, A, S):
        raise ValueError(f"transitions must have shape {(S, A, S)}, got {T.shape}.")
    if R.shape == (S, A):
        R = np.repeat(R[:, :, None], S, axis=2)
    elif R.shape != (S, A, S):
        raise ValueError(
            f"rewards must have shape {(S, A, S)} or {(S, A)}, got {R.shape}."
        )
    if np.any(T < 0):
        raise ValueError("transition probabilities must be non-negative.")
    rowsum = T.sum(axis=2)
    bad = np.argwhere(~(np.isclose(rowsum, 1.0) | np.isclose(rowsum, 0.0)))
    if bad.size:
        s0, a0 = bad[0]
        raise ValueError(
            f"transitions[{s0}, {a0}] sums to {rowsum[s0, a0]:.6g}; each (s, a) row "
            "must sum to 1 (available) or 0 (unavailable)."
        )
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Q)):
        raise ValueError("Q and rewards must be finite.")
    gamma = float(gamma)
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}.")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}.")
    tol = float(tol)
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")

    ER = np.sum(T * R, axis=2)  # (S, A) expected immediate reward
    residual = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        V = Q.max(axis=1)
        Q_new = ER + gamma * (T @ V)
        residual = float(np.max(np.abs(Q_new - Q)))
        Q = Q_new
        if residual <= tol:
            break

    V = Q.max(axis=1)
    policy = Q.argmax(axis=1)
    converged = residual <= tol

    return RichResult(
        title="Bellman optimality (Q-value iteration)",
        summary_lines=[("Sweeps", it), ("Sup-norm residual", residual)],
        warnings=[] if converged else [
            f"did not converge in {max_iter} sweeps (residual {residual:.3g}); "
            f"gamma={gamma} may be too close to 1."
        ],
        payload={
            "Q": Q.tolist(),
            "V": V.tolist(),
            "policy": policy.tolist(),
            "residual": residual,
            "iterations": it,
            "converged": bool(converged),
            "gamma": gamma,
            "estimate": float(V.mean()),
            "n": int(S),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbo: Bellman optimality Q*(s,a)=E[r+gamma*max_a' Q*(s',a')], solved by value iteration"
