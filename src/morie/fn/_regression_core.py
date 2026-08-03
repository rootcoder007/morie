# morie.fn -- shared core (rootcoder007/morie)
"""Linear regression with real inference, robust standard errors and
the standard diagnostics.

This is the block a working statistics package cannot be without: the
things people run on every model, every day.  Each is verified
numerically against the reference implementation the field actually
uses --

    OLS summary            R ``lm`` / ``summary.lm``
    HC0-HC3 robust SEs     R ``sandwich::vcovHC``
    Newey-West HAC         R ``sandwich::NeweyWest``
    Breusch-Pagan          R ``lmtest::bptest``
    Durbin-Watson          R ``lmtest::dwtest``
    variance inflation     R ``car::vif``

No external numeric dependency: plain Python throughout.
"""

import math

__all__ = [
    "ols", "robust_vcov", "robust_se", "newey_west_vcov",
    "breusch_pagan", "durbin_watson", "variance_inflation_factors",
]


def _mat(X):
    rows = list(X)
    if rows and not isinstance(rows[0], (list, tuple)):
        return [[float(v)] for v in rows]
    return [[float(v) for v in r] for r in rows]


def _flat(v):
    return [float(t) for t in v]


def _solve(A, b):
    """Gauss-Jordan with partial pivoting."""
    n = len(A)
    M = [[float(A[i][j]) for j in range(n)] + [float(b[i])]
         for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            raise ValueError("singular design matrix: predictors are "
                             "perfectly collinear")
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(n + 1)]
    return [M[i][n] for i in range(n)]


def _inv(A):
    n = len(A)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(_solve(A, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _xtx(X):
    n, k = len(X), len(X[0])
    return [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)]
            for a in range(k)]


# ------------------------------------------------------ distributions
def _betacf(a, b, x, itmax=300, eps=1e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        h *= d * (c if abs(c) > 1e-300 else 1e-300)
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        delta = d * (c if abs(c) > 1e-300 else 1e-300)
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(b * math.log1p(-x) + a * math.log(x) - lb) \
        * _betacf(b, a, 1 - x) / b


def _t_sf(t, df):
    """Upper tail of Student's t, i.e. R's pt(t, df, lower = FALSE)."""
    x = df / (df + t * t)
    p = 0.5 * _betainc(0.5 * df, 0.5, x)
    return p if t >= 0 else 1.0 - p


def _f_sf(f, df1, df2):
    """Upper tail of the F distribution.

    Evaluated directly as I_{df2/(df2+df1 f)}(df2/2, df1/2) rather than
    as 1 - I_x(df1/2, df2/2): for a strong model the lower tail rounds
    to 1 and the complement collapses to exactly 0, throwing away the
    whole p-value.  This form keeps it -- e.g. 1.5e-48 instead of 0.
    """
    if f <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f)
    return _betainc(df2 / 2.0, df1 / 2.0, x)


def _chi2_sf(x, df):
    """Upper tail of chi-squared, by the regularized incomplete gamma."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    xx = x / 2.0
    if xx < a + 1.0:                       # series
        term = 1.0 / a
        s = term
        n = a
        for _ in range(1000):
            n += 1.0
            term *= xx / n
            s += term
            if abs(term) < abs(s) * 1e-16:
                break
        return 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    # continued fraction
    b = xx + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))


# ------------------------------------------------------------- the fit
def ols(y, X, add_intercept=True, names=None):
    """Ordinary least squares with the inference people actually read.

    Returns coefficients with their standard errors, t statistics and
    two-sided p-values, R^2 and adjusted R^2, the residual standard
    error, and the overall F test against the intercept-only model --
    i.e. what ``summary(lm(...))`` prints.

    The normal equations are solved by Gauss-Jordan with partial
    pivoting, and perfectly collinear predictors raise rather than
    silently returning one arbitrary solution out of infinitely many.
    """
    ys = _flat(y)
    Xm = _mat(X)
    n = len(ys)
    if len(Xm) != n:
        raise ValueError("X has %d rows but y has %d" % (len(Xm), n))
    if add_intercept:
        Xm = [[1.0] + list(r) for r in Xm]
    k = len(Xm[0])
    if n <= k:
        raise ValueError("need more observations than parameters "
                         "(n=%d, k=%d)" % (n, k))
    XtX = _xtx(Xm)
    Xty = [sum(Xm[i][a] * ys[i] for i in range(n)) for a in range(k)]
    beta = _solve(XtX, Xty)
    fitted = [sum(Xm[i][j] * beta[j] for j in range(k)) for i in range(n)]
    resid = [ys[i] - fitted[i] for i in range(n)]

    df_resid = n - k
    rss = sum(r * r for r in resid)
    s2 = rss / df_resid
    XtXinv = _inv(XtX)
    se = [math.sqrt(s2 * XtXinv[j][j]) for j in range(k)]
    tvals = [beta[j] / se[j] if se[j] > 0 else float("nan")
             for j in range(k)]
    pvals = [2.0 * _t_sf(abs(t), df_resid) if se[j] > 0 else float("nan")
             for j, t in enumerate(tvals)]

    ybar = sum(ys) / n
    tss = sum((t - ybar) ** 2 for t in ys) if add_intercept \
        else sum(t * t for t in ys)
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    df_model = k - 1 if add_intercept else k
    adj = (1.0 - (1.0 - r2) * (n - (1 if add_intercept else 0))
           / df_resid) if df_model > 0 else float("nan")
    if df_model > 0:
        fstat = ((tss - rss) / df_model) / s2
        fp = _f_sf(fstat, df_model, df_resid)
    else:
        fstat = fp = float("nan")

    if names is None:
        names = (["(Intercept)"] if add_intercept else []) + \
            ["x%d" % (j + 1) for j in range(k - (1 if add_intercept else 0))]
    return {"coef": beta, "se": se, "t": tvals, "p_value": pvals,
            "names": list(names), "fitted": fitted, "residuals": resid,
            "n": n, "k": k, "df_resid": df_resid, "df_model": df_model,
            "rss": rss, "tss": tss, "sigma2": s2,
            "sigma": math.sqrt(s2), "r_squared": r2,
            "adj_r_squared": adj, "f_statistic": fstat, "f_p_value": fp,
            "XtX_inv": XtXinv, "design": Xm,
            "method": "ordinary least squares"}


# --------------------------------------------- robust covariance
def robust_vcov(fit, kind="HC1"):
    """Heteroskedasticity-consistent covariance, White's sandwich.

        V = (X'X)^-1 X' diag(omega_i) X (X'X)^-1

    with the small-sample adjustment selected by ``kind``:

        HC0  omega = e^2                     White (1980)
        HC1  omega = e^2 n/(n-k)             the Stata default
        HC2  omega = e^2/(1-h_ii)
        HC3  omega = e^2/(1-h_ii)^2          best under leverage

    Use these when the errors are heteroskedastic: the coefficients
    stay unbiased but the textbook standard errors do not.
    Matches ``sandwich::vcovHC``.
    """
    X = fit["design"]
    e = fit["residuals"]
    Ainv = fit["XtX_inv"]
    n, k = fit["n"], fit["k"]
    h = []
    for i in range(n):
        h.append(sum(X[i][a] * Ainv[a][b] * X[i][b]
                     for a in range(k) for b in range(k)))
    kind = kind.upper()
    if kind == "HC0":
        om = [e[i] ** 2 for i in range(n)]
    elif kind == "HC1":
        om = [e[i] ** 2 * n / (n - k) for i in range(n)]
    elif kind == "HC2":
        om = [e[i] ** 2 / (1.0 - h[i]) for i in range(n)]
    elif kind == "HC3":
        om = [e[i] ** 2 / (1.0 - h[i]) ** 2 for i in range(n)]
    else:
        raise ValueError("kind must be HC0, HC1, HC2 or HC3")
    meat = [[sum(X[i][a] * om[i] * X[i][b] for i in range(n))
             for b in range(k)] for a in range(k)]
    V = [[sum(Ainv[a][p] * meat[p][q] * Ainv[q][b]
              for p in range(k) for q in range(k))
          for b in range(k)] for a in range(k)]
    return {"vcov": V, "se": [math.sqrt(V[j][j]) for j in range(k)],
            "leverage": h, "kind": kind,
            "method": "heteroskedasticity-consistent covariance"}


def robust_se(fit, kind="HC1"):
    """Robust standard errors with the t tests they imply."""
    r = robust_vcov(fit, kind)
    se = r["se"]
    b = fit["coef"]
    df = fit["df_resid"]
    t = [b[j] / se[j] if se[j] > 0 else float("nan")
         for j in range(len(b))]
    return {"se": se, "t": t,
            "p_value": [2.0 * _t_sf(abs(v), df) for v in t],
            "kind": kind, "vcov": r["vcov"],
            "method": "robust standard errors"}


def newey_west_vcov(fit, lags=None, prewhite=False):
    """Newey-West heteroskedasticity- and autocorrelation-consistent
    covariance.

        V = A^-1 (S0 + sum_{l=1}^{L} w_l (S_l + S_l')) A^-1,
        w_l = 1 - l/(L+1)                      (Bartlett kernel)

    The Bartlett weights guarantee a positive semi-definite estimate.
    ``lags`` defaults to floor(4 (n/100)^(2/9)) -- Newey and West's own
    rule, and the one ``statsmodels`` uses.  Note that R's
    ``sandwich::NeweyWest`` instead selects the bandwidth automatically
    via ``bwNeweyWest``, so the two disagree unless you pass R an
    explicit ``lag``; with the same lag they agree exactly.  Use this on
    time series where the errors are serially correlated.
    """
    X = fit["design"]
    e = fit["residuals"]
    Ainv = fit["XtX_inv"]
    n, k = fit["n"], fit["k"]
    if prewhite:
        raise NotImplementedError("prewhitening is not implemented; "
                                  "pass prewhite=False")
    if lags is None:
        lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    u = [[X[i][a] * e[i] for a in range(k)] for i in range(n)]

    def gamma(l):
        return [[sum(u[i][a] * u[i - l][b] for i in range(l, n))
                 for b in range(k)] for a in range(k)]

    S = gamma(0)
    for l in range(1, lags + 1):
        g = gamma(l)
        w = 1.0 - l / (lags + 1.0)
        for a in range(k):
            for b in range(k):
                S[a][b] += w * (g[a][b] + g[b][a])
    S = [[S[a][b] * n / (n - k) for b in range(k)] for a in range(k)]
    V = [[sum(Ainv[a][p] * S[p][q] * Ainv[q][b]
              for p in range(k) for q in range(k))
          for b in range(k)] for a in range(k)]
    return {"vcov": V, "se": [math.sqrt(V[j][j]) for j in range(k)],
            "lags": lags,
            "method": "Newey-West HAC covariance"}


# ------------------------------------------------------- diagnostics
def breusch_pagan(fit, studentise=True):
    """Breusch-Pagan test for heteroskedasticity.

    Regresses the squared residuals on the original design; under
    homoskedasticity the statistic is chi-squared with k-1 degrees of
    freedom.  ``studentise=True`` is Koenker's version, which does not
    assume normal errors -- and is what ``lmtest::bptest`` does by
    default.
    """
    X = fit["design"]
    e = fit["residuals"]
    n, k = fit["n"], fit["k"]
    e2 = [t * t for t in e]
    sigma2 = sum(e2) / n
    aux = ols([t / sigma2 for t in e2], [r[1:] for r in X],
              add_intercept=True)
    if studentise:
        stat = 0.5 * (aux["tss"] - aux["rss"]) * sigma2 ** 2 * 2 / \
            (sum((t - sum(e2) / n) ** 2 for t in e2) / n)
        # Koenker: n * R^2 of e^2 on X
        aux2 = ols(e2, [r[1:] for r in X], add_intercept=True)
        stat = n * aux2["r_squared"]
    else:
        stat = 0.5 * (aux["tss"] - aux["rss"])
    df = k - 1
    return {"statistic": stat, "df": df,
            "p_value": _chi2_sf(stat, df),
            "studentised": bool(studentise),
            "method": "Breusch-Pagan test for heteroskedasticity"}


def durbin_watson(fit):
    """Durbin-Watson statistic for first-order autocorrelation.

        DW = sum_{t=2}^{n} (e_t - e_{t-1})^2 / sum_t e_t^2

    Roughly 2(1 - rho): near 2 means no autocorrelation, near 0 strong
    positive, near 4 strong negative.  The exact p-value needs the
    Pan-Durbin distribution, so the first-order residual correlation is
    reported alongside instead of a fabricated one.
    """
    e = fit["residuals"]
    n = len(e)
    num = sum((e[i] - e[i - 1]) ** 2 for i in range(1, n))
    den = sum(t * t for t in e)
    dw = num / den
    rho = sum(e[i] * e[i - 1] for i in range(1, n)) / den
    return {"statistic": dw, "rho": rho, "n": n,
            "method": "Durbin-Watson test for autocorrelation"}


def variance_inflation_factors(X, add_intercept=True, names=None):
    """Variance inflation factors.

        VIF_j = 1 / (1 - R_j^2)

    with R_j^2 from regressing predictor j on all the others.  A VIF
    above about 10 says that coefficient's variance is inflated an
    order of magnitude by collinearity -- the estimate is unstable even
    though the fit as a whole may be fine.  Matches ``car::vif``.
    """
    Xm = _mat(X)
    n, p = len(Xm), len(Xm[0])
    if p < 2:
        raise ValueError("VIF needs at least 2 predictors")
    out = []
    for j in range(p):
        yj = [Xm[i][j] for i in range(n)]
        others = [[Xm[i][q] for q in range(p) if q != j]
                  for i in range(n)]
        r2 = ols(yj, others, add_intercept=add_intercept)["r_squared"]
        out.append(1.0 / (1.0 - r2) if r2 < 1 else float("inf"))
    if names is None:
        names = ["x%d" % (j + 1) for j in range(p)]
    return {"vif": out, "names": list(names),
            "method": "variance inflation factors"}
