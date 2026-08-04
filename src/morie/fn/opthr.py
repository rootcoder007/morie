# morie.fn -- k02 batch (rootcoder007/morie)
"""Huber tuning constant for a target asymptotic efficiency.

Source consulted: Huber, P.J. (1964), Robust estimation of a location
parameter, *Annals of Mathematical Statistics* 35(1), 73-101.  For the
monotone psi_k(x) = max(-k, min(k, x)) the asymptotic variance of the
M-estimator at the standard normal is E[psi^2]/(E[psi'])^2, and both
expectations are available in closed form:

    E[psi'] = 2 Phi(k) - 1
    E[psi^2] = 2 Phi(k) - 1 - 2 k phi(k) + 2 k^2 (1 - Phi(k))

so the efficiency relative to the sample mean is

    ARE(k) = (2 Phi(k) - 1)^2 / ( 2 Phi(k) - 1 - 2 k phi(k) + 2 k^2 (1 - Phi(k)) )

ARE is increasing in k, so the constant achieving a target efficiency is found
by bisection.  The classical answer k = 1.345 for 95 per cent efficiency drops
out of this equation and is asserted in the canonical test.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as _st

from ._richresult import RichResult

__all__ = ["optimal_huber_k"]


def _are(k):
    if k <= 0.0:
        return 0.0
    phi = float(_st.norm.cdf(k))
    den = float(_st.norm.pdf(k))
    a = 2.0 * phi - 1.0
    e2 = a - 2.0 * k * den + 2.0 * k * k * (1.0 - phi)
    return a * a / e2


def optimal_huber_k(efficiency=0.95, lower=1e-6, upper=20.0, iters=200):
    """Huber's k for a given normal-model efficiency.

    Parameters
    ----------
    efficiency : float, default 0.95
        Target asymptotic relative efficiency at the normal.
    lower, upper : float
        Bisection bracket on k.
    iters : int, default 200
        Bisection steps (fixed, so the answer is deterministic).

    Returns
    -------
    RichResult
        estimate (k), efficiency, achieved, asymptotic_variance,
        breakdown_hint, n, method.
    """
    target = float(efficiency)
    lo = float(lower)
    hi = float(upper)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if _are(mid) < target:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    ach = _are(k)
    return RichResult(
        payload={
            "estimate": float(k),
            "efficiency": target,
            "achieved": float(ach),
            "asymptotic_variance": float(1.0 / ach),
            "breakdown_hint": 0.5,
            "n": 0,
            "method": "Huber tuning constant for a target normal efficiency (Huber 1964)",
        }
    )


# CANONICAL TEST
# >>> r = optimal_huber_k(0.95)
# >>> assert abs(r["estimate"] - 1.345) < 5e-4    # the published Huber constant
# >>> assert abs(r["achieved"] - 0.95) < 1e-10
# >>> assert optimal_huber_k(0.90)["estimate"] < r["estimate"]


def cheatsheet():
    return "opthr(efficiency): Huber tuning constant for a target efficiency."


optimalhuberk = optimal_huber_k
