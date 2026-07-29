# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Temporal-difference (TD) learning update."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_td_learning"]


def geron_td_learning(V, s, r, s_next, alpha=0.1, gamma=0.9, terminal=None):
    """
    Temporal-difference (TD) learning update.

    Formula: V(s) <- V(s) + alpha*(r + gamma*V(s') - V(s))

    Applies TD(0) to one transition or, when the arguments are sequences,
    sweeps a whole trajectory in order (each update sees the values left
    by the previous one -- that ordering is what makes TD bootstrap).
    Terminal successors are handled explicitly: their bootstrap term is
    dropped, so ``target = r``.

    Parameters
    ----------
    V : array-like
        Current value table, one entry per state.
    s, r, s_next : scalar or array-like
        Transition(s). `s` and `s_next` are state indices.
    alpha : float, default 0.1
        Step size in (0, 1].
    gamma : float, default 0.9
        Discount factor in [0, 1].
    terminal : array-like of bool, optional
        Marks transitions whose successor is terminal.

    Returns
    -------
    result : RichResult
        Keys: V, td_error, target, updates, estimate, n, method.

    Examples
    --------
    V = [0, 1], reward 1, gamma 0.9, alpha 0.5: the TD target is
    1 + 0.9*1 = 1.9, the error is 1.9 - 0 = 1.9, so V(0) moves to 0.95.

    >>> r = geron_td_learning([0.0, 1.0], 0, 1.0, 1, alpha=0.5, gamma=0.9)
    >>> [round(float(v), 12) for v in r["V"]]
    [0.95, 1.0]
    >>> round(float(r["td_error"][0]), 12)
    1.9

    A terminal successor drops the bootstrap term:

    >>> r2 = geron_td_learning([0.0, 1.0], 0, 1.0, 1, alpha=0.5, gamma=0.9, terminal=[True])
    >>> round(float(r2["target"][0]), 12)
    1.0

    References
    ----------
    Géron Ch 19
    """
    v = np.array(np.atleast_1d(np.asarray(V, dtype=float)), copy=True)
    if v.size == 0:
        raise ValueError("geron_td_learning: V is empty")
    if not np.all(np.isfinite(v)):
        raise ValueError("geron_td_learning: V contains non-finite values")
    sa = np.atleast_1d(np.asarray(s)).astype(int)
    sn = np.atleast_1d(np.asarray(s_next)).astype(int)
    rr = np.atleast_1d(np.asarray(r, dtype=float))
    if not (sa.size == sn.size == rr.size):
        raise ValueError(
            f"geron_td_learning: s, r and s_next must be the same length, got {sa.size}, {rr.size}, {sn.size}"
        )
    if sa.size == 0:
        raise ValueError("geron_td_learning: no transitions supplied")
    for nm, arr in (("s", sa), ("s_next", sn)):
        if arr.min() < 0 or arr.max() >= v.size:
            raise ValueError(f"geron_td_learning: {nm} indexes outside the value table of size {v.size}")
    a = float(alpha)
    g = float(gamma)
    if not (0.0 < a <= 1.0):
        raise ValueError(f"geron_td_learning: alpha must lie in (0, 1], got {a}")
    if not (0.0 <= g <= 1.0):
        raise ValueError(f"geron_td_learning: gamma must lie in [0, 1], got {g}")
    term = np.zeros(sa.size, dtype=bool) if terminal is None else np.atleast_1d(np.asarray(terminal)).astype(bool)
    if term.size != sa.size:
        raise ValueError(f"geron_td_learning: terminal has {term.size} flags but there are {sa.size} transitions")

    errors = np.empty(sa.size)
    targets = np.empty(sa.size)
    for t in range(sa.size):
        boot = 0.0 if term[t] else g * v[sn[t]]
        targets[t] = rr[t] + boot
        errors[t] = targets[t] - v[sa[t]]
        v[sa[t]] += a * errors[t]

    return RichResult(
        title="TD(0) update",
        summary_lines=[
            ("Transitions", int(sa.size)),
            ("alpha", a),
            ("gamma", g),
            ("Mean |TD error|", float(np.mean(np.abs(errors)))),
        ],
        interpretation=(
            "TD bootstraps: it updates towards its own current estimate of the next state, so it "
            "learns online without waiting for the episode to end -- at the cost of bias while V is wrong."
        ),
        payload={
            "V": v,
            "td_error": errors,
            "target": targets,
            "updates": int(sa.size),
            "alpha": a,
            "gamma": g,
            "estimate": float(np.mean(np.abs(errors))),
            "n": int(sa.size),
            "method": "TD(0) value update applied sequentially",
        },
    )


def cheatsheet():
    return "hmtd: Temporal-difference (TD) learning update"
