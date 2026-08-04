# morie.fn -- function file (rootcoder007/morie)
"""Curve registration by shift.

Ramsay and Silverman (2005), *Functional Data Analysis*, 2nd ed.,
Springer, Chapter 7 "Registration: aligning features": two curves that
differ only in phase are aligned by finding the warping h that
minimises

    min_h integral ( y_1(t) - y_2(h(t)) )^2 dt,

which is the criterion named in the stub docstring.  Section 7.2
treats the simplest and commonest case, SHIFT registration, where
h(t) = t + delta and only a single constant is estimated per curve.
That is what this function does.

Search strategy, chosen so that both language arms land on identical
numbers rather than merely the same optimum: the criterion is evaluated
at every integer sample lag in [-max_lag, max_lag] over the overlapping
part of the two curves, normalised by the length of the overlap so
that lags are comparable, and the integer minimiser is then refined by
fitting a parabola through the criterion at the minimiser and its two
neighbours.  No optimiser, no random restarts, no tolerance.

The parabolic step is the standard three-point vertex
delta = 0.5 (c_{-1} - c_{+1}) / (c_{-1} - 2 c_0 + c_{+1}), which is
zero when the criterion is symmetric about the integer minimiser, so
an exactly-shifted pair recovers its shift exactly.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["function_register"]


def function_register(y1, y2, max_lag=None):
    """Align y2 to y1 by a shift.

    Parameters
    ----------
    y1, y2 : array-like
        The target and the curve to be registered, sampled on a common
        equally spaced grid.
    max_lag : int, optional
        Largest lag searched.  Defaults to floor(n / 2).

    Returns
    -------
    estimate : the refined shift in samples (positive = y2 lags y1)
    shift    : the integer minimising lag
    criterion : the normalised criterion at the integer minimiser
    profile  : the criterion at every searched lag
    lags     : the searched lags
    """
    a = k.vec(y1)
    b = k.vec(y2)
    n = len(a)
    if n == 0:
        raise ValueError("function_register: y1 is empty")
    if len(b) != n:
        raise ValueError("function_register: y1 and y2 must have the same length")
    if n < 3:
        raise ValueError("function_register: need at least three sampling points")
    M = n // 2 if max_lag is None else int(max_lag)
    if M < 1:
        raise ValueError("function_register: max_lag must be at least 1")
    if M > n - 2:
        M = n - 2
    lags = list(range(-M, M + 1))
    prof = []
    for d in lags:
        lo = max(0, -d)
        hi = min(n, n - d)
        m = hi - lo
        if m <= 0:
            prof.append(float("inf"))
            continue
        s = 0.0
        for i in range(lo, hi):
            r = a[i] - b[i + d]
            s += r * r
        prof.append(s / float(m))
    best = 0
    for i in range(len(lags)):
        if prof[i] < prof[best]:
            best = i
    ref = 0.0
    if 0 < best < len(lags) - 1:
        cm, c0, cp = prof[best - 1], prof[best], prof[best + 1]
        den = cm - 2.0 * c0 + cp
        if den > 0.0:
            ref = 0.5 * (cm - cp) / den
    return RichResult(
        title="Shift registration",
        summary_lines=[("points", n), ("shift", lags[best]), ("criterion", prof[best])],
        payload={
            "estimate": lags[best] + ref,
            "shift": lags[best],
            "refinement": ref,
            "criterion": prof[best],
            "profile": prof,
            "lags": [float(v) for v in lags],
            "n": n,
            "method": "Ramsay-Silverman (2005) Sect. 7.2 shift registration, integer lag search plus three-point parabolic refinement",
        },
    )


def cheatsheet():
    return "freg: curve registration by shift"
