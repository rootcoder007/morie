# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Normal-approximation bootstrap confidence interval.

Source: Davison, A. C. and Hinkley, D. V. (1997), *Bootstrap Methods and
their Application*, Cambridge University Press, Section 5.2.  The
bootstrap supplies two numbers -- a bias estimate and a standard error --
and the interval is the ordinary normal one built from them:

    bias* = mean(t*) - t,     se* = sd(t*),
    [ t - bias* - z_{1-alpha/2} se* ,  t - bias* + z_{1-alpha/2} se* ].

The bias is *subtracted*, not added: mean(t*) - t estimates E(T) - theta,
so the corrected estimate is t - bias*.  This interval assumes the
replicate distribution is normal and is the first thing to abandon when
it is not; the returned ``skew`` (the standardised third moment of the
replicates) is there so that assumption is visible rather than implicit.

sd(t*) uses the n-1 divisor, matching stats::sd, which is the anchor.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["boot_normal_ci"]


def boot_normal_ci(theta_hat, theta_b, alpha=0.05):
    """Normal-approximation interval from bootstrap replicates.

    Returns
    -------
    lo, hi : the endpoints
    bias : mean(t*) - t
    se_b : sd(t*), the bootstrap standard error
    skew : standardised third moment of the replicates
    """
    v = core.vec(theta_b)
    n = len(v)
    if n < 2:
        raise ValueError("boot_normal_ci: need at least two bootstrap replicates")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("boot_normal_ci: alpha must lie strictly between 0 and 1")
    t = float(theta_hat)
    m = core.mean(v)
    se = core.sd(v, 1)
    bias = m - t
    z = core.qnorm(1.0 - a / 2.0)
    centre = t - bias
    m2 = 0.0
    m3 = 0.0
    for x in v:
        d = x - m
        m2 += d * d
        m3 += d * d * d
    m2 = m2 / n
    m3 = m3 / n
    skew = m3 / (m2 ** 1.5) if m2 > 0.0 else float("nan")
    return RichResult(
        title="Normal-approximation bootstrap interval",
        summary_lines=[("lo", centre - z * se), ("hi", centre + z * se)],
        payload={
            "lo": centre - z * se,
            "hi": centre + z * se,
            "estimate": 2.0 * z * se,
            "bias": bias,
            "se_b": se,
            "centre": centre,
            "skew": skew,
            "z": z,
            "B": n,
            "method": "t - bias* +/- z_{1-a/2} se*, Davison and Hinkley (1997) Sect. 5.2",
        },
    )


def cheatsheet():
    return "btnorm: Normal-approximation bootstrap CI"


# compact alias per ledger/NAMING.md
bootnormalci = boot_normal_ci
