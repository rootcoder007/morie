# morie.fn -- slice s03 (rootcoder007/morie)
"""TMLE for the marginal risk ratio.

Source consulted: van der Laan, M. J. and Rubin, D. (2006).  Targeted
maximum likelihood learning.  *The International Journal of
Biostatistics* 2(1), article 11.  The initial outcome fit is fluctuated
along a parametric submodel whose score spans the efficient influence
curve; for the ATE the clever covariate is

    H(D, X) = D / g(X) - (1 - D) / (1 - g(X))

and the fluctuation is fitted on the logistic scale, which keeps the
targeted predictions inside [0, 1].  The 2006 article is open access
but was not retrievable here; the clever covariate and the logistic
fluctuation are quoted in their standard published form and are
reproduced identically in every account of TMLE.

The marginal risk ratio is a smooth function of the two targeted means,

    RR = E[Y(1)] / E[Y(0)]

so, by the delta method, the influence curve of log RR is

    IC_logRR = IC_1 / E[Y(1)] - IC_0 / E[Y(0)]

with IC_d the influence curve of the corresponding mean.  The standard
error is reported on the log scale and the interval exponentiated back,
because that is the scale on which the normal approximation holds.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["tmle_marginal_rr"]


def tmle_marginal_rr(y, D, X=None, alpha=0.05, trim=0.0):
    """Targeted estimate of E[Y(1)] / E[Y(0)].

    Returns
    -------
    RichResult with payload:
        estimate : the risk ratio
        log_rr, se_log : on the log scale
        ci_lo, ci_hi : interval for the ratio
        ey1, ey0
    """
    fit = k.tmle_ate(y, D, X, trim)
    yv = k.vec(y)
    d = k.vec(D)
    n = len(yv)
    g = fit["g"]
    q1 = fit["q1"]
    q0 = fit["q0"]
    lo = fit["shift"]
    rng = fit["scale"]
    m1 = 0.0
    m0 = 0.0
    for i in range(n):
        m1 += (lo + rng * q1[i]) / n
        m0 += (lo + rng * q0[i]) / n
    ic = []
    for i in range(n):
        qa1 = lo + rng * q1[i]
        qa0 = lo + rng * q0[i]
        i1 = (d[i] / g[i]) * (yv[i] - qa1) + qa1 - m1
        i0 = ((1.0 - d[i]) / (1.0 - g[i])) * (yv[i] - qa0) + qa0 - m0
        ic.append(i1 / m1 - i0 / m0 if m1 != 0.0 and m0 != 0.0 else float("nan"))
    v = 0.0
    for x in ic:
        v += x * x
    se = math.sqrt(v / (n * n)) if n else float("nan")
    rr = m1 / m0 if m0 != 0.0 else float("nan")
    lrr = math.log(rr) if rr > 0.0 else float("nan")
    z = k.qnorm(1.0 - float(alpha) / 2.0)
    return RichResult(
        title="TMLE marginal risk ratio",
        summary_lines=[("RR", rr), ("SE(log RR)", se)],
        payload={
            "estimate": rr,
            "rr": rr,
            "log_rr": lrr,
            "se_log": se,
            "ci_lo": math.exp(lrr - z * se) if lrr == lrr else float("nan"),
            "ci_hi": math.exp(lrr + z * se) if lrr == lrr else float("nan"),
            "ey1": m1,
            "ey0": m0,
            "eps": fit["eps"],
            "n": n,
            "method": "TMLE for the marginal risk ratio, delta-method influence curve on the log scale",
        },
    )


def cheatsheet():
    return "tmlmrr: TMLE for marginal risk ratio"


tmlemarginalrr = tmle_marginal_rr
