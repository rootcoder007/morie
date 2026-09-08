"""Quandt likelihood ratio (sup-F) structural-break test.

Quandt (1960); Andrews (1993); p-values per Hansen (1997).
"""

import math

from ._stats_core import chi2
from ._richresult import RichResult

__all__ = ["qlrtst", "quandt_likelihood_ratio"]

# Hansen (1997), Table 2 (SupF distribution, m >= 1), transcribed from
# the rendered page 64 (fetched-wave3/hansen-1997-structural-change-
# pvalues.pdf): for each m and symmetric trimming pi0, the p-value
# approximation of his Eq. 8 is p = P(chi^2_eta > theta0 + theta1 * x)
# where x is the sup Wald statistic.  Rows: m -> {pi0: (theta0,
# theta1, eta)}.
_HANSEN_T2 = {
    1:  {.01: (-1.79, 1.17, 4.5), .05: (-1.39, 1.07, 3.6),
         .15: (-0.99, 1.02, 3.0), .25: (-0.73, 0.98, 2.5),
         .35: (-0.50, 0.96, 2.1)},
    2:  {.01: (-3.06, 1.18, 6.1), .05: (-2.38, 1.11, 5.4),
         .15: (-1.65, 1.06, 4.7), .25: (-1.16, 1.02, 4.1),
         .35: (-0.78, 0.97, 3.5)},
    3:  {.01: (-4.09, 1.21, 7.8), .05: (-3.31, 1.10, 6.5),
         .15: (-2.05, 1.13, 6.8), .25: (-1.61, 1.03, 5.5),
         .35: (-1.06, 1.01, 4.9)},
    4:  {.01: (-5.33, 1.21, 8.9), .05: (-4.08, 1.14, 8.2),
         .15: (-2.52, 1.11, 8.0), .25: (-1.91, 1.04, 7.0),
         .35: (-1.45, 0.97, 5.7)},
    5:  {.01: (-6.39, 1.18, 9.4), .05: (-4.84, 1.15, 9.3),
         .15: (-3.46, 1.07, 8.3), .25: (-2.63, 1.02, 7.5),
         .35: (-1.82, 1.00, 7.0)},
    6:  {.01: (-7.08, 1.26, 11.8), .05: (-5.37, 1.19, 11.2),
         .15: (-4.05, 1.08, 9.5), .25: (-2.94, 1.05, 9.0),
         .35: (-1.79, 1.03, 8.6)},
    7:  {.01: (-8.49, 1.17, 11.1), .05: (-6.21, 1.21, 12.6),
         .15: (-4.42, 1.10, 11.0), .25: (-3.23, 1.05, 10.1),
         .35: (-2.21, 1.01, 9.3)},
    8:  {.01: (-9.20, 1.17, 12.2), .05: (-7.24, 1.13, 11.9),
         .15: (-5.36, 1.08, 11.3), .25: (-3.65, 1.06, 11.4),
         .35: (-1.69, 1.10, 12.2)},
    9:  {.01: (-10.22, 1.14, 12.3), .05: (-8.07, 1.11, 12.4),
         .15: (-5.43, 1.10, 13.1), .25: (-4.38, 1.01, 11.3),
         .35: (-2.83, 1.00, 11.1)},
    10: {.01: (-11.01, 1.14, 13.3), .05: (-8.84, 1.11, 13.2),
         .15: (-6.47, 1.06, 12.8), .25: (-4.97, 1.01, 12.0),
         .35: (-2.92, 1.05, 13.0)},
}


def _ols_ssr(x_rows, y):
    # SSR via normal equations with partial-pivot elimination
    n = len(y)
    k = len(x_rows[0])
    a = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for xi, yi in zip(x_rows, y):
        for r in range(k):
            b[r] += xi[r] * yi
            for c in range(k):
                a[r][c] += xi[r] * xi[c]
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-300:
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(k):
            if r != c and m[r][c] != 0.0:
                f = m[r][c] / m[c][c]
                for j in range(c, k + 1):
                    m[r][j] -= f * m[c][j]
    beta = [m[i][k] / m[i][i] for i in range(k)]
    ssr = 0.0
    for xi, yi in zip(x_rows, y):
        e = yi - sum(bb * v for bb, v in zip(beta, xi))
        ssr += e * e
    return ssr


def qlrtst(y, X=None, trim=0.15):
    """
    Quandt likelihood ratio (sup-Wald) test for a structural break at
    unknown date.

    For every candidate break k in the symmetric trimmed range
    [trim*T, (1-trim)*T], fit the regression separately on the two
    subsamples and form the Wald (Chow) statistic

        W(k) = (T - 2m) (SSR_0 - SSR_1(k)) / SSR_1(k),

    where SSR_0 is the full-sample and SSR_1(k) the split-sample sum
    of squared residuals and m the number of regression coefficients.
    The test statistic is SupF = max_k W(k) (Quandt 1960; Andrews
    1993, whose Eq. defines SupF_n = sup_k F_n(k)); its asymptotic
    null distribution is nonstandard.  The p-value uses Hansen
    (1997), Eq. 8 with his Table 2: p = P(chi^2_eta > theta0 +
    theta1 SupF), coefficients selected by (m, pi0).

    Sources
    -------
    Quandt, R. E. (1960). Tests of the hypothesis that a linear
    regression system obeys two separate regimes. *JASA*, 55,
    324-330.
    Andrews, D. W. K. (1993). Tests for parameter instability and
    structural change with unknown change point. *Econometrica*,
    61, 821-856.
    Hansen, B. E. (1997). Approximate asymptotic p values for
    structural-change tests. *JBES*, 15, 60-67, Eq. 8 and Table 2
    (local copy
    fetched-wave3/hansen-1997-structural-change-pvalues.pdf).

    Parameters
    ----------
    y : sequence of float
        Dependent variable.
    X : sequence of sequences, optional
        Regressor rows WITHOUT intercept (added automatically);
        default mean-shift model (intercept only).
    trim : float
        Symmetric trimming fraction pi0; Hansen coefficients exist
        for 0.01, 0.05, 0.15 (default), 0.25, 0.35.

    Returns
    -------
    RichResult
        Keys: statistic (SupF), breakpoint (0-based index of first
        obs of second regime), p_value (None if m > 10), f_path,
        m, trim.
    """
    yv = [float(v) for v in y]
    n = len(yv)
    if X is None:
        rows = [[1.0] for _ in range(n)]
    else:
        rows = [[1.0] + [float(v) for v in r] for r in X]
        if len(rows) != n:
            raise ValueError("X must have one row per observation")
    m = len(rows[0])
    tr = float(trim)
    if not any(abs(tr - t) < 1e-12 for t in (.01, .05, .15, .25, .35)):
        raise ValueError("trim must be one of 0.01, 0.05, 0.15, "
                         "0.25, 0.35 (Hansen Table 2)")
    lo = max(m + 1, int(math.ceil(tr * n)))
    hi = min(n - m - 1, int(math.floor((1.0 - tr) * n)))
    if hi <= lo:
        raise ValueError("sample too short for this trimming")
    ssr0 = _ols_ssr(rows, yv)
    if ssr0 is None:
        raise ValueError("singular full-sample design")
    best = (-1.0, None)
    path = []
    for k in range(lo, hi + 1):
        s1 = _ols_ssr(rows[:k], yv[:k])
        s2 = _ols_ssr(rows[k:], yv[k:])
        if s1 is None or s2 is None:
            path.append(float("nan"))
            continue
        ssr1 = s1 + s2
        if ssr1 <= 0:
            path.append(float("inf"))
            w = float("inf")
        else:
            w = (n - 2.0 * m) * (ssr0 - ssr1) / ssr1
            path.append(w)
        if w > best[0]:
            best = (w, k)
    sup_f, kbest = best
    p = None
    if m in _HANSEN_T2:
        t0, t1, eta = _HANSEN_T2[m][round(tr, 2)]
        z = t0 + t1 * sup_f
        p = 1.0 if z <= 0 else float(chi2.sf(z, eta))
    return RichResult(payload={
        "statistic": sup_f,
        "breakpoint": kbest,
        "p_value": p,
        "f_path": path,
        "path_start": lo,
        "m": m,
        "trim": tr,
        "n": n,
        "method": "Quandt/Andrews sup-Wald; Hansen (1997) p-value",
    })


# long descriptive alias (stub-era name)
quandt_likelihood_ratio = qlrtst


def cheatsheet():
    return "qlrtst: SupF = max_k (T-2m)(SSR0-SSR1)/SSR1; p via Hansen Eq.8"
