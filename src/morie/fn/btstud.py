# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Studentised (bootstrap-t) confidence interval.

Source: Davison, A. C. and Hinkley, D. V. (1997), *Bootstrap Methods and
their Application*, Cambridge University Press, Section 5.2, which
defines the studentised statistic z* = (t* - t)/v*^{1/2} and gives the
limits

    [ t - z*_{(1-alpha/2)} v^{1/2} ,  t - z*_{(alpha/2)} v^{1/2} ].

Also Hall, P. (1988), "Theoretical comparison of bootstrap confidence
intervals", *The Annals of Statistics* 16(3), 927-953,
doi:10.1214/aos/1176350933, which is the second-order accuracy result
that motivates studentising in the first place.

Like the basic interval this one is *reversed*: the upper quantile of z*
sets the lower endpoint.  Feeding it the raw replicates t* instead of the
studentised z* is the classic misuse and produces an interval that looks
plausible and is wrong; the module therefore takes ``t_b`` and ``se_hat``
separately and never divides for the caller.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_studentized_ci"]


def boot_studentized_ci(theta_hat, se_hat, t_b, alpha=0.05):
    """Bootstrap-t interval.

    Parameters
    ----------
    theta_hat : float
        Estimate on the original data.
    se_hat : float
        Its standard error on the original data; must be positive.
    t_b : array-like
        The studentised replicates z*_b = (t*_b - t)/se*_b.
    alpha : float
        Two-sided error rate.

    Returns
    -------
    lo, hi : the endpoints
    z_lo, z_hi : the studentised quantiles used
    """
    v = core.vec(t_b)
    n = len(v)
    if n == 0:
        raise ValueError("boot_studentized_ci: no studentised replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_studentized_ci: alpha must lie strictly between 0 and 1")
    s = float(se_hat)
    if not (s > 0.0):
        raise ValueError("boot_studentized_ci: se_hat must be positive")
    t = float(theta_hat)
    zlo = core.quantile7(v, a / 2.0)
    zhi = core.quantile7(v, 1.0 - a / 2.0)
    lo = t - zhi * s
    hi = t - zlo * s
    return RichResult(
        title="Studentised bootstrap interval",
        summary_lines=[("lo", lo), ("hi", hi)],
        payload={
            "lo": lo,
            "hi": hi,
            "estimate": hi - lo,
            "z_lo": zlo,
            "z_hi": zhi,
            "se_hat": s,
            "theta_hat": t,
            "B": n,
            "method": "t - z*_(1-a/2) se, t - z*_(a/2) se; Davison and Hinkley (1997) Sect. 5.2",
        },
    )


def cheatsheet():
    return "btstud: Studentised bootstrap (bootstrap-t) CI"


# compact alias per ledger/NAMING.md
bootstudentizedci = boot_studentized_ci
