# morie.fn -- shared core (rootcoder007/morie)
"""Generalised linear models: logistic, Poisson, Gaussian, Gamma.

Fitted by iteratively reweighted least squares, which is Fisher
scoring for these families, and verified against R's ``glm``:

    binomial(logit)     glm(family = binomial)
    poisson(log)        glm(family = poisson)
    gaussian(identity)  glm(family = gaussian)  == lm
    Gamma(log)          glm(family = Gamma(link = "log"))

Reported quantities are the ones a ``summary(glm(...))`` shows:
coefficients with standard errors, Wald z (or t) statistics and
p-values, the null and residual deviance with their degrees of freedom,
and AIC.

No external numeric dependency.
"""

import math

__all__ = ["glm", "glm_predict", "deviance_residuals", "FAMILIES"]


def _mat(X):
    rows = list(X)
    if rows and not isinstance(rows[0], (list, tuple)):
        return [[float(v)] for v in rows]
    return [[float(v) for v in r] for r in rows]


def _flat(v):
    return [float(t) for t in v]


def _solve(A, b):
    n = len(A)
    M = [[float(A[i][j]) for j in range(n)] + [float(b[i])]
         for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            raise ValueError("singular information matrix: predictors "
                             "are collinear or a category is empty")
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [M[r][q] - f * M[c][q] for q in range(n + 1)]
    return [M[i][n] for i in range(n)]


def _inv(A):
    n = len(A)
    cols = [_solve(A, [1.0 if i == j else 0.0 for i in range(n)])
            for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


_EPS = 1e-10


def _clip01(p):
    return min(max(p, _EPS), 1.0 - _EPS)


#: Link, inverse link, variance function and deviance contribution for
#: each supported family.  Keeping them together is what lets one IRLS
#: loop serve all of them.
FAMILIES = {
    "binomial": {
        "link": lambda mu: math.log(_clip01(mu) / (1 - _clip01(mu))),
        "linkinv": lambda e: 1.0 / (1.0 + math.exp(-max(min(e, 700.0),
                                                        -700.0))),
        "variance": lambda mu: max(mu * (1 - mu), _EPS),
        "mu_eta": lambda e: (lambda p: max(p * (1 - p), _EPS))(
            1.0 / (1.0 + math.exp(-max(min(e, 700.0), -700.0)))),
        "dev_resid": lambda y, mu: 2.0 * (
            (y * math.log(y / _clip01(mu)) if y > 0 else 0.0)
            + ((1 - y) * math.log((1 - y) / (1 - _clip01(mu)))
               if y < 1 else 0.0)),
        "start": lambda y: (y + 0.5) / 2.0,
        "dispersion_fixed": True,
    },
    "poisson": {
        "link": lambda mu: math.log(max(mu, _EPS)),
        "linkinv": lambda e: math.exp(min(e, 700.0)),
        "variance": lambda mu: max(mu, _EPS),
        "mu_eta": lambda e: math.exp(min(e, 700.0)),
        "dev_resid": lambda y, mu: 2.0 * (
            (y * math.log(y / max(mu, _EPS)) if y > 0 else 0.0)
            - (y - mu)),
        "start": lambda y: y + 0.1,
        "dispersion_fixed": True,
    },
    "gaussian": {
        "link": lambda mu: mu,
        "linkinv": lambda e: e,
        "variance": lambda mu: 1.0,
        "mu_eta": lambda e: 1.0,
        "dev_resid": lambda y, mu: (y - mu) ** 2,
        "start": lambda y: y,
        "dispersion_fixed": False,
    },
    "gamma": {
        "link": lambda mu: math.log(max(mu, _EPS)),
        "linkinv": lambda e: math.exp(min(e, 700.0)),
        "variance": lambda mu: max(mu, _EPS) ** 2,
        "mu_eta": lambda e: math.exp(min(e, 700.0)),
        "dev_resid": lambda y, mu: 2.0 * (
            -math.log(max(y, _EPS) / max(mu, _EPS))
            + (y - mu) / max(mu, _EPS)),
        "start": lambda y: max(y, _EPS),
        "dispersion_fixed": False,
    },
}


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def _betacf(a, b, x, itmax=300, eps=1e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (1e-300 if abs(d) < 1e-300 else d)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (1e-300 if abs(d) < 1e-300 else d)
        h *= d * (1e-300 if abs(c) < 1e-300 else c)
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (1e-300 if abs(d) < 1e-300 else d)
        delta = d * (1e-300 if abs(c) < 1e-300 else c)
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
    x = df / (df + t * t)
    p = 0.5 * _betainc(0.5 * df, 0.5, x)
    return p if t >= 0 else 1.0 - p


def glm(y, X, family="binomial", add_intercept=True, weights=None,
        offset=None, max_iter=25, tol=1e-8):
    """Fit a generalised linear model by iteratively reweighted least
    squares.

    Each step solves a weighted least-squares problem in the working
    response

        z = eta - offset + (y - mu) / (dmu/deta),
        w = prior_weight * (dmu/deta)^2 / V(mu)

    which is Fisher scoring for the canonical links used here, so it
    converges quadratically and the final weighted cross-product is the
    expected information --- hence the standard errors come straight
    out of its inverse.

    For binomial and Poisson the dispersion is fixed at 1 and the Wald
    statistics are normal; for gaussian and Gamma it is estimated by the
    Pearson statistic over the residual degrees of freedom and the
    statistics are t, exactly as R's ``summary.glm`` does.
    """
    fam = FAMILIES.get(str(family).lower())
    if fam is None:
        raise ValueError("family must be one of %s"
                         % ", ".join(sorted(FAMILIES)))
    ys = _flat(y)
    Xm = _mat(X)
    n = len(ys)
    if len(Xm) != n:
        raise ValueError("X has %d rows but y has %d" % (len(Xm), n))
    if add_intercept:
        Xm = [[1.0] + list(r) for r in Xm]
    p = len(Xm[0])
    if n <= p:
        raise ValueError("need more observations than parameters")
    pw = [1.0] * n if weights is None else _flat(weights)
    off = [0.0] * n if offset is None else _flat(offset)
    if str(family).lower() == "binomial" and any(
            t < 0 or t > 1 for t in ys):
        raise ValueError("binomial response must lie in [0, 1]")
    if str(family).lower() == "poisson" and any(t < 0 for t in ys):
        raise ValueError("Poisson response must be non-negative")

    mu = [fam["start"](t) for t in ys]
    eta = [fam["link"](m) for m in mu]
    beta = [0.0] * p
    converged = False
    dev_old = None
    w_fit = None
    for _ in range(int(max_iter)):
        w, z = [], []
        for i in range(n):
            g = fam["mu_eta"](eta[i])
            v = fam["variance"](mu[i])
            w.append(pw[i] * g * g / v)
            z.append(eta[i] - off[i] + (ys[i] - mu[i]) / g)
        A = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in range(n))
              for b in range(p)] for a in range(p)]
        rhs = [sum(w[i] * Xm[i][a] * z[i] for i in range(n))
               for a in range(p)]
        beta = _solve(A, rhs)
        w_fit = w          # the weights that produced this beta
        eta = [off[i] + sum(Xm[i][j] * beta[j] for j in range(p))
               for i in range(n)]
        mu = [fam["linkinv"](e) for e in eta]
        dev = sum(pw[i] * fam["dev_resid"](ys[i], mu[i])
                  for i in range(n))
        if dev_old is not None and abs(dev - dev_old) / (
                abs(dev) + 0.1) < tol:
            converged = True
            dev_old = dev
            break
        dev_old = dev

    deviance = dev_old
    # null model: intercept only (or offset only)
    if add_intercept:
        mu0 = sum(pw[i] * ys[i] for i in range(n)) / sum(pw)
        null_dev = sum(pw[i] * fam["dev_resid"](ys[i], mu0)
                       for i in range(n))
        df_null = n - 1
    else:
        null_dev = sum(pw[i] * fam["dev_resid"](
            ys[i], fam["linkinv"](off[i])) for i in range(n))
        df_null = n

    df_resid = n - p
    # Standard errors come from the weighted cross-product of the FINAL
    # IRLS solve -- the weights evaluated at the eta that produced beta,
    # not recomputed at the converged eta.  The two differ by one
    # Fisher-scoring step, which is invisible in beta but shows up in
    # the eighth digit of every standard error; R's summary.glm inverts
    # the stored QR, i.e. the former, so this is what agreement means.
    w = w_fit
    A = [[sum(w[i] * Xm[i][a] * Xm[i][b] for i in range(n))
          for b in range(p)] for a in range(p)]
    V = _inv(A)

    pearson = sum(pw[i] * (ys[i] - mu[i]) ** 2 / fam["variance"](mu[i])
                  for i in range(n))
    if fam["dispersion_fixed"]:
        disp = 1.0
    else:
        disp = pearson / df_resid if df_resid > 0 else float("nan")
    V = [[V[a][b] * disp for b in range(p)] for a in range(p)]
    se = [math.sqrt(V[j][j]) for j in range(p)]
    stat = [beta[j] / se[j] for j in range(p)]
    if fam["dispersion_fixed"]:
        pv = [2.0 * _norm_sf(abs(s)) for s in stat]
        stat_name = "z"
    else:
        pv = [2.0 * _t_sf(abs(s), df_resid) for s in stat]
        stat_name = "t"

    # AIC as R defines it per family
    fl = str(family).lower()
    if fl == "binomial":
        ll = sum(pw[i] * (ys[i] * math.log(_clip01(mu[i]))
                          + (1 - ys[i]) * math.log(1 - _clip01(mu[i])))
                 for i in range(n))
        aic = -2 * ll + 2 * p
    elif fl == "poisson":
        ll = sum(pw[i] * (ys[i] * math.log(max(mu[i], _EPS)) - mu[i]
                          - math.lgamma(ys[i] + 1.0))
                 for i in range(n))
        aic = -2 * ll + 2 * p
    elif fl == "gaussian":
        s2 = deviance / n
        ll = -0.5 * n * (math.log(2 * math.pi * s2) + 1.0)
        aic = -2 * ll + 2 * (p + 1)
    else:
        ll = float("nan")
        aic = float("nan")

    return {"coef": beta, "se": se, "statistic": stat,
            "statistic_name": stat_name, "p_value": pv,
            "fitted": mu, "linear_predictor": eta,
            "residuals": [ys[i] - mu[i] for i in range(n)],
            "deviance": deviance, "null_deviance": null_dev,
            "df_residual": df_resid, "df_null": df_null,
            "dispersion": disp, "pearson_chi2": pearson,
            "aic": aic, "loglik": ll, "converged": converged,
            "family": fl, "n": n, "k": p, "vcov": V,
            "method": "generalised linear model (IRLS)"}


def glm_predict(fit, X, add_intercept=True, type="response",
                offset=None):
    """Predict from a fitted GLM.

    ``type="link"`` returns the linear predictor, ``"response"`` the
    mean on the scale of the data --- the distinction that trips people
    up when a logistic model appears to predict outside [0, 1].
    """
    Xm = _mat(X)
    if add_intercept:
        Xm = [[1.0] + list(r) for r in Xm]
    b = fit["coef"]
    if len(Xm[0]) != len(b):
        raise ValueError("X has %d columns but the fit has %d "
                         "coefficients" % (len(Xm[0]), len(b)))
    off = [0.0] * len(Xm) if offset is None else _flat(offset)
    eta = [off[i] + sum(Xm[i][j] * b[j] for j in range(len(b)))
           for i in range(len(Xm))]
    if type == "link":
        return eta
    if type != "response":
        raise ValueError('type must be "link" or "response"')
    inv = FAMILIES[fit["family"]]["linkinv"]
    return [inv(e) for e in eta]


def deviance_residuals(fit, y):
    """Signed deviance residuals, sign(y - mu) sqrt(d_i).

    Their sum of squares is the deviance, and unlike raw residuals they
    are roughly symmetric for non-normal families, which is what makes
    them the right thing to plot.
    """
    ys = _flat(y)
    mu = fit["fitted"]
    dr = FAMILIES[fit["family"]]["dev_resid"]
    out = []
    for i in range(len(ys)):
        d = max(dr(ys[i], mu[i]), 0.0)
        out.append(math.copysign(math.sqrt(d), ys[i] - mu[i]))
    return out
