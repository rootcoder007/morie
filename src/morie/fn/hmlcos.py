# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cosine annealing learning rate schedule."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_cosine_annealing"]

_METHOD = "Cosine annealing learning-rate schedule"


def geron_cosine_annealing(t, T, eta_max, eta_min=0.0):
    """
    Cosine annealing learning rate schedule.

    Formula: eta_t = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi*t/T))

    The half-cosine leaves ``eta_max`` with zero slope and arrives at
    ``eta_min`` with zero slope, which is the point: no discontinuity at
    either end, unlike step decay, and most of the decay happens in the
    middle of training rather than at the start.  The whole schedule for
    steps ``0..T`` is returned alongside the value at ``t``.

    Parameters
    ----------
    t : int or array-like of int
        Step(s) at which to evaluate, ``0 <= t <= T``.
    T : int
        Total number of steps in the cycle (positive).
    eta_max : float
        Learning rate at ``t = 0``.
    eta_min : float
        Learning rate at ``t = T``; must not exceed ``eta_max``.

    Returns
    -------
    result : RichResult
        Keys: eta, schedule, T, eta_max, eta_min, estimate, n, method.

    Examples
    --------
    Endpoints and midpoint are exact: ``cos(0) = 1``, ``cos(pi/2) = 0``,
    ``cos(pi) = -1``:

    >>> r = geron_cosine_annealing([0, 2, 4], T=4, eta_max=0.1, eta_min=0.0)
    >>> [round(float(v), 10) for v in r["eta"]]
    [0.1, 0.05, 0.0]

    A non-zero floor shifts the whole curve:

    >>> s = geron_cosine_annealing(1, T=2, eta_max=1.0, eta_min=0.2)
    >>> round(float(s["eta"]), 10)
    0.6

    The schedule is monotone decreasing over the cycle:

    >>> import numpy as np
    >>> sch = geron_cosine_annealing(0, T=10, eta_max=1.0)["schedule"]
    >>> bool(np.all(np.diff(sch) < 0)), len(sch)
    (True, 11)

    References
    ----------
    Géron Ch 11
    """
    T_int = int(T)
    if T_int < 1:
        raise ValueError(f"geron_cosine_annealing: T must be a positive number of steps, got {T!r}")
    hi, lo = float(eta_max), float(eta_min)
    if not np.isfinite(hi) or not np.isfinite(lo):
        raise ValueError("geron_cosine_annealing: eta_max and eta_min must be finite")
    if lo < 0:
        raise ValueError(f"geron_cosine_annealing: eta_min must be non-negative, got {lo}")
    if hi < lo:
        raise ValueError(f"geron_cosine_annealing: eta_max ({hi}) must be at least eta_min ({lo})")

    tt = np.asarray(t)
    scalar = tt.ndim == 0
    tt = np.atleast_1d(tt).astype(float)
    if tt.size == 0:
        raise ValueError("geron_cosine_annealing: t is empty")
    if np.any(tt < 0) or np.any(tt > T_int):
        raise ValueError(f"geron_cosine_annealing: every t must lie in 0..{T_int}, got {tt.tolist()}")

    def _eta(steps):
        return lo + 0.5 * (hi - lo) * (1.0 + np.cos(np.pi * steps / T_int))

    eta = _eta(tt)
    schedule = _eta(np.arange(T_int + 1, dtype=float))

    return RichResult(
        title="Cosine annealing",
        summary_lines=[("Cycle length", T_int), ("eta_max", hi), ("eta_min", lo)],
        interpretation="Smooth at both ends: the rate leaves eta_max and reaches eta_min with zero slope.",
        payload={
            "eta": float(eta[0]) if scalar else eta,
            "schedule": schedule,
            "T": T_int,
            "eta_max": hi,
            "eta_min": lo,
            "estimate": float(eta[0]) if scalar else float(eta[0]),
            "n": int(tt.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlcos: cosine annealing eta_t = eta_min + 0.5*(eta_max-eta_min)*(1+cos(pi t/T))"
