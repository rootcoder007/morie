# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exponential learning rate decay."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_lr_exponential"]

_METHOD = "Exponential learning-rate decay"


def geron_lr_exponential(eta0, decay, t):
    """
    Exponential learning rate decay.

    Formula: eta_t = eta_0 * decay^t

    Constant *relative* decay: every step multiplies the rate by the
    same factor, so the rate never reaches zero and the number of steps
    to drop by a factor of ten is fixed at ``log(10)/(-log(decay))``.
    That half-life-style constant is returned, because ``decay`` alone
    is hard to read -- 0.99 and 0.999 look similar and differ tenfold in
    how long they take to matter.

    Parameters
    ----------
    eta0 : float
        Initial learning rate (positive).
    decay : float
        Per-step multiplier, ``0 < decay <= 1``.
    t : int or array-like
        Step(s), non-negative.

    Returns
    -------
    result : RichResult
        Keys: eta, steps_per_decade, eta0, decay, estimate, n, method.

    Examples
    --------
    ``0.1 * 0.5^3 = 0.0125``:

    >>> r = geron_lr_exponential(0.1, 0.5, 3)
    >>> float(r["eta"])
    0.0125

    With ``decay = 0.5`` a tenfold drop takes ``log(10)/log(2)`` steps:

    >>> round(r["steps_per_decade"], 6)
    3.321928

    ``decay = 1`` is a constant schedule, and the decade count is then
    infinite rather than an error:

    >>> c = geron_lr_exponential(0.3, 1.0, [0, 100])
    >>> [float(v) for v in c["eta"]]
    [0.3, 0.3]
    >>> c["steps_per_decade"]
    inf

    References
    ----------
    Géron Ch 11
    """
    e0 = float(eta0)
    d = float(decay)
    if not np.isfinite(e0) or e0 <= 0:
        raise ValueError(f"geron_lr_exponential: eta0 must be a positive finite learning rate, got {eta0!r}")
    if not np.isfinite(d) or d <= 0 or d > 1:
        raise ValueError(f"geron_lr_exponential: decay must lie in (0, 1], got {decay!r}")

    tt = np.asarray(t)
    scalar = tt.ndim == 0
    tt = np.atleast_1d(tt).astype(float)
    if tt.size == 0:
        raise ValueError("geron_lr_exponential: t is empty")
    if np.any(tt < 0) or not np.all(np.isfinite(tt)):
        raise ValueError("geron_lr_exponential: t must be finite and non-negative")

    eta = e0 * d**tt
    spd = float("inf") if d == 1.0 else float(np.log(10.0) / -np.log(d))

    return RichResult(
        title="Exponential LR decay",
        summary_lines=[("eta0", e0), ("decay", d), ("Steps per 10x drop", spd)],
        interpretation="Constant relative decay: the rate approaches but never reaches zero.",
        payload={
            "eta": float(eta[0]) if scalar else eta,
            "steps_per_decade": spd,
            "eta0": e0,
            "decay": d,
            "estimate": float(eta[0]),
            "n": int(tt.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlrex: exponential decay eta_t = eta_0 * decay^t, plus steps per tenfold drop"
