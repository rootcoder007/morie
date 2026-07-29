# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Warm restarts: cosine decay with periodic restarts."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_warm_restarts"]


def geron_warm_restarts(t, T0=10, factor=2.0, eta_max=0.1, eta_min=0.0):
    """
    Warm restarts: cosine decay with periodic restarts.

    Formula: after period T_i restart to eta_max; T_{i+1} = T_i * factor

    SGDR (Loshchilov & Hutter). Within cycle i the learning rate follows
    the cosine schedule

    ``eta = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi * T_cur / T_i))``

    so it starts at `eta_max`, ends at `eta_min`, and the restart is a
    genuine jump back up -- that jump is the mechanism, not a side
    effect: it kicks the optimiser out of a narrow minimum and the
    schedule then anneals into whatever basin it lands in. Cycle lengths
    grow geometrically, ``T_{i+1} = factor * T_i``, so late cycles get
    long anneals.

    `t` may be a scalar or an array of steps; the containing cycle is
    located by accumulating cycle lengths, so the answer is exact rather
    than approximated by a modulo of the first period.

    Parameters
    ----------
    t : int or array-like
        Global step(s), >= 0.
    T0 : int, default 10
        Length of the first cycle (>= 1).
    factor : float, default 2.0
        Geometric growth of the cycle length (>= 1).
    eta_max, eta_min : float
        Learning-rate bounds; ``eta_min <= eta_max`` and both >= 0.

    Returns
    -------
    result : RichResult
        Keys: eta, cycle, cycle_length, step_in_cycle, restarts,
        estimate, n, method.

    Examples
    --------
    Step 0 is the top of the first cycle; halfway through, the cosine is
    exactly at the midpoint; the step after the cycle ends restarts.

    >>> r = geron_warm_restarts([0, 5, 10], T0=10, factor=2.0, eta_max=0.1)
    >>> [round(float(v), 12) for v in r["eta"]]
    [0.1, 0.05, 0.1]
    >>> [int(c) for c in r["cycle"]]
    [0, 0, 1]
    >>> [int(c) for c in r["cycle_length"]]
    [10, 10, 20]

    The second cycle is twice as long, so its midpoint is at step 20:

    >>> round(float(geron_warm_restarts(20, T0=10, factor=2.0, eta_max=0.1)["eta"][0]), 12)
    0.05

    References
    ----------
    Géron Ch 11
    """
    steps = np.atleast_1d(np.asarray(t)).ravel()
    if steps.size == 0:
        raise ValueError("geron_warm_restarts: t is empty")
    if not np.all(np.equal(np.mod(steps.astype(float), 1), 0)):
        raise ValueError("geron_warm_restarts: t must contain whole step numbers")
    steps = steps.astype(int)
    if np.any(steps < 0):
        raise ValueError("geron_warm_restarts: t must be non-negative")
    T = int(T0)
    if T < 1:
        raise ValueError(f"geron_warm_restarts: T0 must be >= 1, got {T}")
    f = float(factor)
    if not np.isfinite(f) or f < 1.0:
        raise ValueError(f"geron_warm_restarts: factor must be >= 1 (cycles never shrink), got {f}")
    hi, lo = float(eta_max), float(eta_min)
    if not (np.isfinite(hi) and np.isfinite(lo)):
        raise ValueError("geron_warm_restarts: eta_max and eta_min must be finite")
    if lo < 0 or hi < 0:
        raise ValueError("geron_warm_restarts: learning rates must be non-negative")
    if lo > hi:
        raise ValueError(f"geron_warm_restarts: eta_min ({lo}) exceeds eta_max ({hi})")

    etas = np.empty(steps.size)
    cyc = np.empty(steps.size, dtype=int)
    clen = np.empty(steps.size, dtype=int)
    scur = np.empty(steps.size, dtype=int)
    for k, step in enumerate(steps):
        i = 0
        start = 0
        length = T
        while step >= start + length:
            start += length
            length = max(1, int(round(length * f)))
            i += 1
        cur = int(step - start)
        etas[k] = lo + 0.5 * (hi - lo) * (1.0 + np.cos(np.pi * cur / length))
        cyc[k] = i
        clen[k] = length
        scur[k] = cur

    return RichResult(
        title="Cosine schedule with warm restarts (SGDR)",
        summary_lines=[
            ("Steps queried", int(steps.size)),
            ("Cycles touched", int(np.max(cyc) + 1)),
            ("eta range", f"[{lo}, {hi}]"),
        ],
        interpretation=(
            "Each restart deliberately raises the learning rate again: the schedule trades a temporarily "
            "worse loss for a chance to leave a sharp minimum, and the growing cycles anneal harder each time."
        ),
        payload={
            "eta": etas,
            "cycle": cyc,
            "cycle_length": clen,
            "step_in_cycle": scur,
            "restarts": int(np.max(cyc)),
            "T0": T,
            "factor": f,
            "estimate": float(etas[-1]),
            "n": int(steps.size),
            "method": "SGDR: cosine annealing within geometrically growing cycles",
        },
    )


def cheatsheet():
    return "hmwrst: Warm restarts: cosine decay with periodic restarts"
