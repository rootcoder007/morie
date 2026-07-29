# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Learning rate schedule used with SGD: eta_t = eta_0 / (t + t_0)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_learning_rate_schedule"]

_METHOD = "Power (1/t) learning-rate schedule"


def geron_learning_rate_schedule(t, eta0, t0):
    """
    Learning rate schedule used with SGD: eta_t = eta_0 / (t + t_0).

    Formula: eta_t = eta_0 / (t + t_0)

    The classic stochastic-approximation schedule.  It satisfies the two
    Robbins-Monro conditions -- ``sum eta_t`` diverges (so the iterate
    can still travel any distance) while ``sum eta_t^2`` converges (so
    the noise is eventually damped) -- and both partial sums up to the
    requested step are returned so that this is visible rather than
    asserted.  ``t0`` keeps the first few steps from being enormous.

    Parameters
    ----------
    t : int or array-like
        Step(s), non-negative.
    eta0 : float
        Numerator (positive).
    t0 : float
        Offset (positive); ``t + t0`` must never be zero.

    Returns
    -------
    result : RichResult
        Keys: eta, schedule, sum_eta, sum_eta_squared, estimate, n, method.

    Examples
    --------
    ``5 / (0 + 50) = 0.1`` and ``5 / (50 + 50) = 0.05``:

    >>> r = geron_learning_rate_schedule([0, 50], eta0=5.0, t0=50.0)
    >>> [float(v) for v in r["eta"]]
    [0.1, 0.05]

    Halving happens after ``t0`` steps, whatever ``t0`` is:

    >>> float(geron_learning_rate_schedule(4, eta0=1.0, t0=4.0)["eta"])
    0.125

    The running sums up to step 3 with eta0=1, t0=1 are
    ``1 + 1/2 + 1/3 + 1/4`` and ``1 + 1/4 + 1/9 + 1/16``:

    >>> s = geron_learning_rate_schedule(3, eta0=1.0, t0=1.0)
    >>> round(s["sum_eta"], 6), round(s["sum_eta_squared"], 6)
    (2.083333, 1.423611)

    References
    ----------
    Géron Ch 4
    """
    e0 = float(eta0)
    off = float(t0)
    if not np.isfinite(e0) or e0 <= 0:
        raise ValueError(f"geron_learning_rate_schedule: eta0 must be positive and finite, got {eta0!r}")
    if not np.isfinite(off) or off <= 0:
        raise ValueError(
            f"geron_learning_rate_schedule: t0 must be positive (it is what keeps eta_0 finite at t=0), got {t0!r}"
        )

    tt = np.asarray(t)
    scalar = tt.ndim == 0
    tt = np.atleast_1d(tt).astype(float)
    if tt.size == 0:
        raise ValueError("geron_learning_rate_schedule: t is empty")
    if np.any(tt < 0) or not np.all(np.isfinite(tt)):
        raise ValueError("geron_learning_rate_schedule: t must be finite and non-negative")

    eta = e0 / (tt + off)
    upto = np.arange(0, int(np.max(tt)) + 1, dtype=float)
    schedule = e0 / (upto + off)

    return RichResult(
        title="1/t learning-rate schedule",
        summary_lines=[("eta0", e0), ("t0", off), ("eta at max t", float(eta.max() if scalar else eta[-1]))],
        interpretation=(
            "Robbins-Monro: sum eta_t diverges so the iterate can still move, "
            "sum eta_t^2 converges so the gradient noise is damped."
        ),
        payload={
            "eta": float(eta[0]) if scalar else eta,
            "schedule": schedule,
            "sum_eta": float(np.sum(schedule)),
            "sum_eta_squared": float(np.sum(schedule**2)),
            "eta0": e0,
            "t0": off,
            "estimate": float(eta[0]),
            "n": int(tt.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlrs: SGD schedule eta_t = eta_0/(t + t_0) with Robbins-Monro partial sums"
