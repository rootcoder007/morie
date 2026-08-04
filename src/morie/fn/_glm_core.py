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


# --------------------------------------------------------------- power
# Statistical power for the t, z and one-way ANOVA F tests.  The API is
# statsmodels.stats.power's (``.power`` / ``.solve_power``) because that
# is what morie.inference calls; the implementation is native, using the
# noncentral t and noncentral F already in morie.fn._stats_core.
#
# Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
# Sciences* (2nd ed.), chapters 2 (t), 6 (z on proportions), 8 (F).


def _bisect(fn, lo, hi, tol=1e-7):
    """Bracketed bisection on an increasing function. ponytail: bisection
    is ~40 evaluations here; Brent would halve that and save no bugs."""
    flo, fhi = fn(lo), fn(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError(
            "solve_power: no solution in bracket [%g, %g]" % (lo, hi))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hi - lo <= tol * max(1.0, abs(mid)):
            return mid
        fm = fn(mid)
        if fm == 0.0:
            return mid
        if flo * fm < 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


class _PowerBase:
    """Shared ``solve_power``: power is monotone increasing in each of
    effect_size, nobs and alpha, so whichever one is left ``None`` is
    recovered by bisection on ``power(...) - target``."""

    _nobs_name = "nobs"
    _es_bracket = (1e-8, 50.0)
    _nobs_hi = 1e7

    def _nobs_lo(self, **kwargs):
        return 2.0 + 1e-6

    def solve_power(self, effect_size=None, alpha=0.05, power=None,
                    **kwargs):
        nobs = kwargs.pop(self._nobs_name, None)
        known = dict(kwargs)

        def pw(es, n, a):
            known[self._nobs_name] = n
            return self.power(effect_size=es, alpha=a, **known)

        if power is None:
            if effect_size is None or nobs is None:
                raise ValueError(
                    "solve_power: exactly one argument may be None")
            return pw(effect_size, nobs, alpha)

        missing = [k for k, v in (("effect_size", effect_size),
                                  (self._nobs_name, nobs),
                                  ("alpha", alpha)) if v is None]
        if len(missing) != 1:
            raise ValueError(
                "solve_power: exactly one of effect_size, %s, alpha, power "
                "must be None (got %d unknowns)"
                % (self._nobs_name, len(missing)))
        what = missing[0]
        if what == "effect_size":
            lo, hi = self._es_bracket
            return _bisect(lambda es: pw(es, nobs, alpha) - power, lo, hi)
        if what == "alpha":
            return _bisect(lambda a: pw(effect_size, nobs, a) - power,
                           1e-12, 1.0 - 1e-12)
        lo = self._nobs_lo(**kwargs)
        return _bisect(lambda n: pw(effect_size, n, alpha) - power,
                       lo, self._nobs_hi)


def _two_tail_power(crit_hi, crit_lo, sf, cdf, alternative):
    """Rejection probability under the alternative for a shifted
    (noncentral) statistic: upper tail, lower tail, or both."""
    if alternative in ("two-sided", "two_sided", "2s"):
        return sf(crit_hi) + cdf(crit_lo)
    if alternative in ("larger", "greater", "one-sided", "1s"):
        return sf(crit_hi)
    if alternative in ("smaller", "less"):
        return cdf(crit_lo)
    raise ValueError("alternative must be 'two-sided', 'larger' or "
                     "'smaller', got %r" % (alternative,))


class TTestPower(_PowerBase):
    """Power of the one-sample (or paired) t test.

    ``ncp = effect_size * sqrt(nobs)`` on ``df = nobs - 1`` degrees of
    freedom, so the test statistic is noncentral t (Cohen 1988, ch. 2).
    """

    def power(self, effect_size, nobs, alpha=0.05, df=None,
              alternative="two-sided"):
        from morie.fn._stats_core import nct, t as _t
        nobs = float(nobs)
        if df is None:
            df = nobs - 1.0
        if df <= 0:
            raise ValueError("nobs must exceed 1 for a one-sample t test")
        ncp = float(effect_size) * math.sqrt(nobs)
        if alternative in ("two-sided", "two_sided", "2s"):
            hi = _t.ppf(1.0 - alpha / 2.0, df)
            lo = -hi
        else:
            hi = _t.ppf(1.0 - alpha, df)
            lo = -hi
        return _two_tail_power(
            hi, lo,
            lambda c: float(nct.sf(c, df, ncp)),
            lambda c: float(nct.cdf(c, df, ncp)),
            alternative)


class TTestIndPower(_PowerBase):
    """Power of the two-independent-sample t test.

    ``ncp = effect_size * sqrt(nobs1 * ratio / (1 + ratio))`` on
    ``df = nobs1 * (1 + ratio) - 2`` (Cohen 1988, ch. 2); with
    ``ratio = 1`` this is the familiar ``d * sqrt(n/2)``.
    """

    _nobs_name = "nobs1"

    def _nobs_lo(self, ratio=1.0, **kwargs):
        return (2.0 + 1e-6) / (1.0 + float(ratio))

    def power(self, effect_size, nobs1, alpha=0.05, ratio=1.0, df=None,
              alternative="two-sided"):
        from morie.fn._stats_core import nct, t as _t
        nobs1 = float(nobs1)
        ratio = float(ratio)
        if df is None:
            df = nobs1 * (1.0 + ratio) - 2.0
        if df <= 0:
            raise ValueError("nobs1 too small: df = %g" % df)
        ncp = float(effect_size) * math.sqrt(nobs1 * ratio / (1.0 + ratio))
        if alternative in ("two-sided", "two_sided", "2s"):
            hi = _t.ppf(1.0 - alpha / 2.0, df)
            lo = -hi
        else:
            hi = _t.ppf(1.0 - alpha, df)
            lo = -hi
        return _two_tail_power(
            hi, lo,
            lambda c: float(nct.sf(c, df, ncp)),
            lambda c: float(nct.cdf(c, df, ncp)),
            alternative)


class NormalIndPower(_PowerBase):
    """Normal (z) approximation to :class:`TTestIndPower`.

    ``power = Phi(delta - z_crit) + Phi(-delta - z_crit)`` with
    ``delta = effect_size * sqrt(nobs1 * ratio / (1 + ratio))``; matches
    the published two-sample z tables (es = 0.2, n1 = n2 = 100,
    alpha = .05 -> 0.293).  Also used on Cohen's h for proportions.
    """

    _nobs_name = "nobs1"

    def _nobs_lo(self, **kwargs):
        return 1e-6

    def power(self, effect_size, nobs1, alpha=0.05, ratio=1.0,
              alternative="two-sided"):
        from morie.fn._stats_core import norm as _norm
        ratio = float(ratio)
        delta = float(effect_size) * math.sqrt(
            float(nobs1) * ratio / (1.0 + ratio))
        if alternative in ("two-sided", "two_sided", "2s"):
            crit = _norm.ppf(1.0 - alpha / 2.0)
        else:
            crit = _norm.ppf(1.0 - alpha)
        return _two_tail_power(
            crit, -crit,
            lambda c: float(_norm.sf(c - delta)),
            lambda c: float(_norm.cdf(c - delta)),
            alternative)


class FTestAnovaPower(_PowerBase):
    """Power of the one-way ANOVA F test.

    Cohen's f is the effect size: ``ncp = f**2 * nobs`` with
    ``dfn = k_groups - 1`` and ``dfd = nobs - k_groups``, where ``nobs``
    is the TOTAL sample size (Cohen 1988, ch. 8).
    """

    _es_bracket = (1e-8, 20.0)

    def _nobs_lo(self, k_groups=2, **kwargs):
        return float(k_groups) + 1e-6

    def power(self, effect_size, nobs, alpha=0.05, k_groups=2):
        from morie.fn._stats_core import ncf, f as _f
        nobs = float(nobs)
        dfn = float(k_groups) - 1.0
        dfd = nobs - float(k_groups)
        if dfn <= 0 or dfd <= 0:
            raise ValueError(
                "need k_groups >= 2 and nobs > k_groups (got %g, %g)"
                % (dfn, dfd))
        ncp = float(effect_size) ** 2 * nobs
        crit = _f.ppf(1.0 - alpha, dfn, dfd)
        return float(ncf.sf(crit, dfn, dfd, ncp))
