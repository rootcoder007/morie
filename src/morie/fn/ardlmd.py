"""ARDL bounds test for the existence of a level relationship."""

from . import _array_core as np

from ._richresult import RichResult
from .zaurts import _ols_coef_se

__all__ = ["ardl_bounds_test"]


def ardl_bounds_test(y, X, p=1, q=1, case=3):
    """
    ARDL bounds test for a level relationship

    Formula: F-stat for H0: pi_yy = 0 and pi_yx = 0 in the conditional ECM

    The bounds test asks whether a long-run level relationship exists
    between ``y`` and ``X`` without first having to decide whether the
    regressors are I(0) or I(1).  It is run on the conditional
    error-correction form

        dy_t = c0 [+ c1 t] + pi_yy y_{t-1} + sum_j pi_yx_j x_{j,t-1}
               + sum_{i=1}^{p-1} psi_i dy_{t-i}
               + sum_j sum_{i=0}^{q-1} beta_ji dx_{j,t-i} + u_t,

    in which the lagged *levels* carry the long-run information and the
    differences absorb the short-run dynamics.  Two statistics are
    reported:

    * ``f_statistic`` -- the Wald/F statistic for the joint null
      ``pi_yy = 0`` and ``pi_yx_j = 0`` for every j, that is, no level
      relationship.  This is the bounds test proper.
    * ``t_statistic`` -- the t-ratio for ``pi_yy = 0`` alone, the
      complementary test on the speed of adjustment.

    Neither statistic has a standard null distribution: under the null
    both limits depend on whether the regressors are I(0) or I(1), which
    is why the test is read against a *pair* of critical values -- a
    lower bound assuming all regressors are I(0) and an upper bound
    assuming all are I(1).  A statistic above the upper bound rejects, one
    below the lower bound does not, and one between the two is
    inconclusive.

    **The critical value bounds are not returned.**  They are tabulated in
    Tables CI(i)-CI(v) of Pesaran, Shin & Smith (2001), one panel per
    deterministic case, and no accessible copy of those tables could be
    obtained when this function was written -- the article is paywalled
    and no authoritative open restatement was found.  Reproducing them
    from memory would be inventing numbers, so the statistics are
    returned for the caller to compare against the published table for
    their ``case`` and number of regressors.  ``f_pvalue_iid`` is
    reported only as a diagnostic: it is the p-value from the ordinary
    F distribution, which is *not* the null distribution of this test and
    is always too small.

    ``case`` follows the paper's numbering of the deterministic terms:
    2 restricted intercept and no trend, 3 unrestricted intercept and no
    trend, 4 unrestricted intercept and restricted trend, 5 unrestricted
    intercept and unrestricted trend.  Cases 3 and 5 are fitted here (an
    unrestricted intercept, with a trend for case 5); cases 2 and 4 place
    the deterministic term inside the error-correction term and are
    rejected rather than silently fitted as case 3.

    Parameters
    ----------
    y : array-like
        The dependent series.
    X : array-like
        ``(n, k)`` regressors.  A 1-D input is read as a single
        regressor.
    p : int
        Lag order of ``y``: ``p - 1`` lagged differences of ``y`` enter.
    q : int
        Lag order of each regressor: ``q - 1`` lagged differences of each
        ``x`` enter, alongside the contemporaneous difference.
    case : {3, 5}
        Deterministic specification, in the numbering above.

    Returns
    -------
    result : RichResult
        Keys: f_statistic, t_statistic, df_num, df_den, n_used, k,
        f_pvalue_iid, pi_yy, case, p, q, method.

    References
    ----------
    Pesaran M H, Shin Y & Smith R J (2001).  Bounds testing approaches to
    the analysis of level relationships.  Journal of Applied Econometrics
    16(3), 289-326.  The conditional error-correction model above is
    their equation (16); the two statistics are their F and t bounds
    tests.
    """
    from . import _stats_core as stats

    yv = [float(v) for v in np.atleast_1d(np.asarray(y, dtype=float)).tolist()]
    xa = np.asarray(X, dtype=float).tolist()
    if len(xa) == 0:
        raise ValueError("X must be non-empty")
    if not isinstance(xa[0], list):
        xm = [[float(v)] for v in xa]
    else:
        xm = [[float(v) for v in row] for row in xa]
    n = len(yv)
    if len(xm) != n:
        raise ValueError("y and X must have the same number of rows")
    k = len(xm[0])
    p = int(p)
    q = int(q)
    if p < 1 or q < 1:
        raise ValueError("p and q must be at least 1")
    if case not in (3, 5):
        raise ValueError("only cases 3 and 5 are fitted here; cases 2 and 4 "
                         "restrict the deterministic term inside the error "
                         "correction term and need a different estimator")

    dy = [yv[t] - yv[t - 1] for t in range(1, n)]
    dx = [[xm[t][j] - xm[t - 1][j] for j in range(k)] for t in range(1, n)]
    # Row t (0-based in y) needs y_{t-1}, dy_{t-1}..dy_{t-p+1} and
    # dx_t..dx_{t-q+1}, so it starts at max(p, q).
    start = p if p > q else q
    rows = list(range(start, n))
    if len(rows) < 3:
        raise ValueError("not enough observations after lagging")

    resp = [dy[t - 1] for t in rows]
    design = []
    for t in rows:
        row = [1.0]
        if case == 5:
            row.append(float(t + 1))
        row.append(yv[t - 1])                      # pi_yy
        for j in range(k):
            row.append(xm[t - 1][j])               # pi_yx_j
        for i in range(1, p):
            row.append(dy[t - i - 1])              # psi_i
        for j in range(k):
            for i in range(0, q):
                row.append(dx[t - i - 1][j])       # beta_ji
        design.append(row)

    ndet = 2 if case == 5 else 1
    fit = _ols_coef_se(design, resp)
    if fit is None:
        raise ValueError("the conditional ECM design is rank deficient; "
                         "reduce p or q")
    beta, se = fit
    m = len(design[0])
    nu = len(rows)
    rss_u = 0.0
    for i in range(nu):
        f = 0.0
        for a in range(m):
            f += design[i][a] * beta[a]
        d = resp[i] - f
        rss_u += d * d

    # Restricted fit: drop the k + 1 lagged level columns.
    keep = [a for a in range(m) if not (ndet <= a < ndet + k + 1)]
    rdes = [[row[a] for a in keep] for row in design]
    rfit = _ols_coef_se(rdes, resp)
    if rfit is None:
        raise ValueError("the restricted design is rank deficient")
    rbeta = rfit[0]
    rss_r = 0.0
    for i in range(nu):
        f = 0.0
        for a in range(len(keep)):
            f += rdes[i][a] * rbeta[a]
        d = resp[i] - f
        rss_r += d * d

    df_num = k + 1
    df_den = nu - m
    if df_den <= 0:
        raise ValueError("no residual degrees of freedom; reduce p or q")
    fstat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    tstat = (beta[ndet] - 0.0) / se[ndet]

    return RichResult(
        payload={
            "f_statistic": float(fstat),
            "t_statistic": float(tstat),
            "df_num": df_num,
            "df_den": df_den,
            "n_used": nu,
            "k": k,
            "f_pvalue_iid": float(stats.f.sf(fstat, df_num, df_den)),
            "pi_yy": float(beta[ndet]),
            "case": case,
            "p": p,
            "q": q,
            "method": "ARDL bounds test for a level relationship "
                      "(Pesaran, Shin & Smith 2001, eq. 16)",
        }
    )


def cheatsheet():
    return "ardlmd: ARDL bounds test for a level relationship (PSS 2001)"


# compact alias per ledger/NAMING.md
ardlboundstest = ardl_bounds_test
