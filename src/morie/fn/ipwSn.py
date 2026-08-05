# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""IPW sensitivity analysis for a non-ignorable selection mechanism.

Robins, Rotnitzky and Scharfstein (2000), "Sensitivity analysis for
selection bias and unmeasured confounding in missing data and causal
inference models", in *Statistical Models in Epidemiology, the
Environment, and Clinical Trials*, IMA Volumes in Mathematics and its
Applications 116, Springer, pp. 1-94, doi:10.1007/978-1-4612-1284-3_1,
and Scharfstein, Rotnitzky and Robins (1999), "Adjusting for
nonignorable drop-out using semiparametric nonresponse models", Journal
of the American Statistical Association 94(448):1096-1120,
doi:10.1080/01621459.1999.10473862.

Their selection-bias function q(y) indexes departures from missing at
random.  With the exponential tilt q(y) = lambda * y the observed-data
weights become

    u_i(lambda) = C_i * exp(-lambda * Y_i) / pi_hat(X_i),
    mu(lambda)  = sum_i u_i Y_i / sum_i u_i,

where pi_hat is the logistic regression estimate of P(C = 1 | X).
lambda = 0 is exactly the MAR inverse-probability-weighted (Hajek)
estimator; lambda > 0 encodes "large Y are less likely to be observed",
lambda < 0 the reverse.  The output is the curve mu(lambda) over the
supplied grid and its range, which is the quantity a sensitivity
analysis reports.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ipw_sensitivity"]


def ipw_sensitivity(Y, X, C, lam_grid):
    """Sensitivity curve of the IPW mean to an exponential selection tilt.

    Parameters
    ----------
    Y : array-like
        Outcome; entries with C = 0 are not used and may be anything.
    X : array-like or None
        Covariate block for the selection model; an intercept is added.
    C : array-like
        Selection (observed) indicator, 0 or 1.
    lam_grid : array-like
        Values of the selection-bias parameter lambda.
    """
    yv = core.vec(Y)
    n = len(yv)
    if n == 0:
        raise ValueError("ipw_sensitivity: Y is empty")
    c = core.vec(C)
    if len(c) != n:
        raise ValueError("ipw_sensitivity: Y and C have different lengths")
    for v in c:
        if v not in (0.0, 1.0):
            raise ValueError("ipw_sensitivity: C must be 0 or 1")
    if sum(c) == 0:
        raise ValueError("ipw_sensitivity: no observed units")
    lam = core.vec(lam_grid)
    if len(lam) == 0:
        raise ValueError("ipw_sensitivity: lam_grid is empty")
    Z = core.design(X, n)
    if len(Z) != n:
        raise ValueError("ipw_sensitivity: X and Y have different lengths")
    gam = core.logit_irls(Z, c, 60)
    pi = [core.sigmoid(sum(Z[i][j] * gam[j] for j in range(len(gam)))) for i in range(n)]
    mus = []
    for L in lam:
        num = 0.0
        den = 0.0
        for i in range(n):
            if c[i] == 1.0:
                u = math.exp(-L * yv[i]) / pi[i]
                num += u * yv[i]
                den += u
        if den <= 0:
            raise ValueError("ipw_sensitivity: tilted weights vanished; lambda too extreme")
        mus.append(num / den)
    zero = None
    for k, L in enumerate(lam):
        if L == 0.0:
            zero = mus[k]
    if zero is None:
        zero = mus[0]
    lo = min(mus)
    hi = max(mus)
    return RichResult(
        title="IPW sensitivity to non-ignorable selection",
        summary_lines=[("n observed", int(sum(c))), ("mu at lambda=0", zero), ("range", hi - lo)],
        payload={
            "estimate": zero,
            "mu": mus,
            "lambda": lam,
            "mu_min": lo,
            "mu_max": hi,
            "range": hi - lo,
            "propensity": pi,
            "gamma": gam,
            "n_observed": int(sum(c)),
            "n": n,
            "method": "mu(lam) = sum C exp(-lam Y) Y / pi / sum C exp(-lam Y) / pi, Robins, Rotnitzky & Scharfstein (2000)",
        },
    )


def cheatsheet():
    return "ipwSn: IPW sensitivity (Robins-Rotnitzky-Scharfstein)"


# compact alias per ledger/NAMING.md
ipwsensitivity = ipw_sensitivity
