# morie.fn -- function file (rootcoder007/morie)
r"""ARDL long-run coefficients and the Pesaran-Shin-Smith bounds test.

Testing for a long-run relationship normally requires knowing first
whether each series is I(0) or I(1) -- and unit-root tests are exactly
the tests with poor power near the boundary, so the pre-test contaminates
everything after it. The bounds procedure sidesteps the pre-test: the
same F statistic on the lagged levels of a conditional error-correction
model is compared with TWO critical values, one computed as if every
regressor were I(0) and one as if every regressor were I(1).

Above the upper bound a long-run relationship is present whatever the
orders are; below the lower bound it is absent; between them the test is
INCONCLUSIVE, and reporting that honestly is the point of the method
rather than a shortcoming of it.

The long-run coefficients follow from the ARDL(p, q) fit as
:math:`\theta_j = \sum_l \beta_{jl} / (1 - \sum_i \phi_i)`, which is why
the sum of the autoregressive coefficients is returned alongside them: as
it approaches one the long-run estimate diverges, and the number makes
that visible instead of leaving a large coefficient unexplained.

References
----------
Pesaran, M. H. and Shin, Y. (1998) "An autoregressive distributed lag
modelling approach to cointegration analysis", in *Econometrics and
Economic Theory in the 20th Century: The Ragnar Frisch Centennial
Symposium*, ed. S. Strom, Cambridge University Press, Ch. 11, 371-413,
doi:10.1017/CCOLial0521633230.011.

Pesaran, M. H., Shin, Y. and Smith, R. J. (2001) "Bounds testing
approaches to the analysis of level relationships", *Journal of Applied
Econometrics* **16**(3), 289-326, doi:10.1002/jae.616. The bounds test
and its two critical-value surfaces.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["ardl_bounds"]

_EPS = 1e-12


def _ols(X, y):
    n, p = len(X), len(X[0])
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)]
           for a in range(p)]
    # numerical floor scaled to the matrix: a fixed absolute ridge does
    # nothing when the design is nearly collinear and the coefficients small
    scale = sum(XtX[a][a] for a in range(p)) / p
    ridge = 1e-8 * scale if scale > 1e-300 else 1e-10
    for a in range(p):
        XtX[a][a] += ridge
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    beta = k.cholsolve(XtX, Xty)
    fit = [sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    res = [y[i] - fit[i] for i in range(n)]
    return beta, fit, res, XtX


def ardl_bounds(y, x, p=1, q=1):
    r"""Fit the conditional ECM and run the bounds test on the levels."""
    yv = [float(v) for v in k.vec(y)]
    Xm = [[float(v) for v in r] for r in k.mat(x)]
    n0 = len(yv)
    if n0 != len(Xm):
        raise ValueError("pesdol: %d responses but %d regressor rows"
                         % (n0, len(Xm)))
    m = len(Xm[0])
    p, q = int(p), int(q)
    if p < 1 or q < 0:
        raise ValueError("pesdol: need p >= 1 and q >= 0")
    start = max(p, q) + 1
    if n0 - start < p + q * m + m + 3:
        raise ValueError("pesdol: too few observations for ARDL(%d, %d) "
                         "with %d regressors" % (p, q, m))

    rows, dep = [], []
    for t in range(start, n0):
        r = [1.0]
        r.append(yv[t - 1])                                  # level of y
        for j in range(m):
            r.append(Xm[t - 1][j])                           # levels of x
        for i in range(1, p):
            r.append(yv[t - i] - yv[t - i - 1])              # dy lags
        for j in range(m):
            for l in range(0, q + 1):
                r.append(Xm[t - l][j] - Xm[t - l - 1][j])    # dx lags
        rows.append(r)
        dep.append(yv[t] - yv[t - 1])

    beta, fit, res, _ = _ols(rows, dep)
    n = len(rows)
    kk = len(rows[0])
    rss_u = sum(v * v for v in res)

    # restricted fit: drop the lagged LEVELS (columns 1 .. 1+m)
    keep = [0] + list(range(1 + m + 1, kk))
    rrows = [[r[j] for j in keep] for r in rows]
    _, _, rres, _ = _ols(rrows, dep)
    rss_r = sum(v * v for v in rres)
    n_rest = 1 + m
    dfe = n - kk
    F = ((rss_r - rss_u) / n_rest) / (rss_u / dfe) if dfe > 0 and rss_u > _EPS \
        else float("nan")

    phi = beta[1]                       # coefficient on y_{t-1}
    theta = [(-beta[1 + 1 + j] / phi) if abs(phi) > _EPS else float("nan")
             for j in range(m)]

    # Pesaran, Shin & Smith (2001) Table CI(iii), case III (unrestricted
    # intercept, no trend), 5% level, indexed by the number of regressors.
    TAB = {1: (4.94, 5.73), 2: (3.79, 4.85), 3: (3.23, 4.35),
           4: (2.86, 4.01), 5: (2.62, 3.79)}
    lo, hi = TAB.get(m, (float("nan"), float("nan")))
    if F != F or lo != lo:
        verdict = "unavailable"
    elif F > hi:
        verdict = "cointegrated"
    elif F < lo:
        verdict = "no long-run relationship"
    else:
        verdict = "inconclusive"

    return RichResult(payload={
        "estimate": theta, "long_run": theta,
        "coefficients": beta, "residuals": res, "fitted": fit,
        "speed_of_adjustment": phi,
        "f_statistic": F, "bound_lower": lo, "bound_upper": hi,
        "verdict": verdict, "n_used": n, "n_params": kk,
        "rss_unrestricted": rss_u, "rss_restricted": rss_r,
        "p": p, "q": q, "n_regressors": m,
        "method": "ARDL conditional ECM with the Pesaran-Shin-Smith bounds "
                  "test (Pesaran & Shin 1998; Pesaran, Shin & Smith 2001)",
        "note": "the bounds test avoids a unit-root PRE-TEST; between the "
                "two critical values the answer is INCONCLUSIVE, which is "
                "the method working rather than failing",
    })


def cheatsheet():
    return ("pesdol: ardl_bounds(y, x, p, q) -> ARDL long-run coefficients "
            "and the bounds test (Pesaran & Shin 1998; Pesaran, Shin & "
            "Smith 2001, J. Appl. Econometrics 16(3), 289-326)")
