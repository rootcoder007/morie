"""Zivot-Andrews unit root test with an endogenous structural break."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["zivot_andrews_unit_root"]

# Zivot & Andrews (1992) asymptotic critical values at 1%, 5% and 10%,
# in the encoding used by urca::ur.za.
_CVAL = {
    "intercept": (-5.34, -4.80, -4.58),
    "trend": (-4.93, -4.42, -4.11),
    "both": (-5.57, -5.08, -4.82),
}


def _ols_coef_se(xmat, yvec):
    """OLS coefficients and standard errors by QR, or None if singular.

    QR rather than the normal equations: squaring the design costs half
    the working precision, and R's ``lm`` -- against which this test is
    checked -- factorises the design directly.
    """
    n = len(yvec)
    k = len(xmat[0])
    if n <= k:
        return None
    q, r = np.linalg.qr(xmat)
    rm = np.atleast_2d(r).tolist()
    qm = np.atleast_2d(q).tolist()
    scale = 0.0
    for a in range(k):
        d = rm[a][a]
        if d < 0.0:
            d = -d
        if d > scale:
            scale = d
    if scale <= 0.0:
        return None
    for a in range(k):
        d = rm[a][a]
        if d < 0.0:
            d = -d
        if d <= 1e-10 * scale:
            return None
    # qty = Q^T y
    qty = [0.0] * k
    for a in range(k):
        s = 0.0
        for i in range(n):
            s += qm[i][a] * yvec[i]
        qty[a] = s
    # beta by back substitution on R beta = Q^T y
    beta = [0.0] * k
    for a in range(k - 1, -1, -1):
        s = qty[a]
        for b in range(a + 1, k):
            s -= rm[a][b] * beta[b]
        beta[a] = s / rm[a][a]
    rss = 0.0
    for i in range(n):
        fit = 0.0
        for a in range(k):
            fit += xmat[i][a] * beta[a]
        d = yvec[i] - fit
        rss += d * d
    sigma2 = rss / (n - k)
    # (X'X)^{-1} = R^{-1} R^{-T}: build R^{-1} by back substitution.
    rinv = [[0.0] * k for _ in range(k)]
    for c in range(k):
        rinv[c][c] = 1.0 / rm[c][c]
        for a in range(c - 1, -1, -1):
            s = 0.0
            for b in range(a + 1, c + 1):
                s += rm[a][b] * rinv[b][c]
            rinv[a][c] = -s / rm[a][a]
    se = []
    for a in range(k):
        d = 0.0
        for b in range(a, k):
            d += rinv[a][b] * rinv[a][b]
        v = sigma2 * d
        se.append(np.sqrt(v) if v > 0.0 else float("nan"))
    return beta, se


def zivot_andrews_unit_root(x, model="intercept", lags=0):
    """
    Zivot-Andrews unit root test with an endogenous break

    Formula: t_alpha_min = min over TB of t_alpha(TB)

    The null is a unit root with no break.  The alternative is a
    trend-stationary process whose intercept, trend, or both shift at a
    single break date TB that is *not* known in advance but chosen by the
    data.  The regression run at each candidate break is, in levels,

        y_t = mu + alpha * y_{t-1} + beta * t
              + theta * DU_t(TB) [+ gamma * DT_t(TB)]
              + sum_{j=1}^{k} c_j * dy_{t-j} + e_t,

    with the break dummies

        DU_t(TB) = 1 for t > TB, else 0        (intercept shift)
        DT_t(TB) = t - TB for t > TB, else 0   (trend shift)

    ``model = "intercept"`` includes DU only (Zivot & Andrews model A),
    ``"trend"`` includes DT only (model B), and ``"both"`` includes both
    (model C).  The test statistic is the minimum over all candidate
    break dates of the t-ratio for ``alpha = 1``,

        t_alpha(TB) = (alpha_hat(TB) - 1) / se(alpha_hat(TB)),

    and the reported break point is the date attaining that minimum.
    Because TB is chosen to minimise the statistic, the null distribution
    is not the Dickey-Fuller one and the critical values below are those
    tabulated by Zivot & Andrews.

    Candidate break dates run over ``1, ..., n - 1`` without trimming,
    matching ``urca::ur.za``.  Candidates whose design matrix is rank
    deficient (which happens near the ends of the sample, where a dummy
    is almost constant) are skipped rather than allowed to produce a
    spurious minimum.

    Parameters
    ----------
    x : array-like
        The series to test.
    model : {"intercept", "trend", "both"}
        Which component is allowed to break.
    lags : int
        Number of lagged differences ``dy_{t-j}`` included.  Default 0.

    Returns
    -------
    result : RichResult
        Keys: statistic, break_point, model, lags, cval_1pct, cval_5pct,
        cval_10pct, tstats, n, method.

    Notes
    -----
    The layout of the regression, the choice not to trim the candidate
    break dates and the critical values all follow ``urca::ur.za``
    (package urca 1.3-4, file R/ur-za.R), which is the reference
    implementation of this test in R and was read directly when this
    function was written.

    References
    ----------
    Zivot E & Andrews D W K (1992).  Further evidence on the great crash,
    the oil-price shock, and the unit-root hypothesis.  Journal of
    Business & Economic Statistics 10(3), 251-270.
    """
    xv = [float(v) for v in np.atleast_1d(np.asarray(x, dtype=float)).tolist()]
    xv = [v for v in xv if v == v]
    n = len(xv)
    if model not in _CVAL:
        raise ValueError("model must be one of 'intercept', 'trend', 'both'")
    lags = int(lags)
    if lags < 0:
        raise ValueError("lags must be a non-negative integer")
    # Columns: y.l1, trend, lagged differences, then the break dummies.
    ncol = lags + 3
    if n < ncol + 2:
        raise ValueError("insufficient number of observations")

    dy = [xv[i + 1] - xv[i] for i in range(n - 1)]
    # Row t (0-based) is usable once y_{t-1} and all lagged differences
    # exist, i.e. from t = lags + 1 onward.
    start = lags + 1
    rows = list(range(start, n))
    yvec = [xv[t] for t in rows]
    base = []
    for t in rows:
        # intercept, y_{t-1}, trend
        row = [1.0, xv[t - 1], float(t + 1)]
        for j in range(1, lags + 1):
            # dy_{t-j} = y_{t-j} - y_{t-j-1}
            row.append(dy[t - j - 1])
        base.append(row)

    tstats = []
    for z in range(1, n):
        extra = []
        for t in rows:
            # urca: du <- c(rep(0, z), rep(1, n - z)) so unit t (1-based
            # t + 1) is in the post-break regime when t + 1 > z.
            e = []
            if model in ("intercept", "both"):
                e.append(1.0 if (t + 1) > z else 0.0)
            if model in ("trend", "both"):
                e.append(float((t + 1) - z) if (t + 1) > z else 0.0)
            extra.append(e)
        xmat = [base[i] + extra[i] for i in range(len(rows))]
        fit = _ols_coef_se(xmat, yvec)
        if fit is None:
            tstats.append(float("nan"))
            continue
        beta, se = fit
        s = se[1]
        if not (s > 0.0) or s != s:
            tstats.append(float("nan"))
            continue
        tstats.append((beta[1] - 1.0) / s)

    best = None
    bpoint = None
    for i, v in enumerate(tstats):
        if v != v:
            continue
        if best is None or v < best:
            best = v
            bpoint = i + 1
    if best is None:
        raise ValueError("no candidate break date gave an estimable regression")

    cv = _CVAL[model]
    return RichResult(
        payload={
            "statistic": float(best),
            "break_point": int(bpoint),
            "model": model,
            "lags": lags,
            "cval_1pct": cv[0],
            "cval_5pct": cv[1],
            "cval_10pct": cv[2],
            "tstats": tstats,
            "n": n,
            "method": "Zivot-Andrews unit root test with endogenous break",
        }
    )


def cheatsheet():
    return "zaurts: Zivot-Andrews unit root test with endogenous break"

