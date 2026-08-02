"""morie glm core: statsmodels subset (OLS / WLS / Logit / GLM +
formula API + families).

Native replacement for the statsmodels surface morie uses (inventory:
sm.families 36, sm.add_constant 21, sm.OLS 11, smf.glm 11, sm.GLM,
smf.wls, sm.Logit, smf.ols). IRLS for GLM families with the standard
links; OLS/WLS closed-form with full inference (bse / tvalues /
pvalues / conf_int / rsquared / llf / aic / bic). Equivalence-tested
against statsmodels in tests/fn/test_glm_core.py.
"""

from __future__ import annotations

import builtins as _bi
import math as _math

from . import _array_core as _ac
from . import _stats_core as _stats


def _tolists(X):
    a = _ac.atleast_2d(X)
    return [list(map(float, r)) for r in a.data]


def _tolist1(y):
    return [float(v) for v in _ac.asarray(y)._flat()]


def add_constant(X, prepend=True):
    if hasattr(X, "_cols"):            # native DataFrame
        out = X.copy()
        col = [1.0] * X.shape[0]
        if prepend:
            out._cols = {"const": col, **out._cols}
        else:
            out["const"] = col
        return out
    a = _ac.asarray(X)
    if len(a.shape) == 1:
        rows = [[v] for v in a._flat()]
    else:
        rows = [list(r) for r in a.data]
    if prepend:
        rows = [[1.0] + r for r in rows]
    else:
        rows = [r + [1.0] for r in rows]
    return _ac.marr(rows)


# ------------------------------------------------------------ families

class _Link:
    pass


class _LogitLink(_Link):
    name = "logit"

    def g(self, mu):
        return _math.log(mu / (1.0 - mu))

    def ginv(self, eta):
        if eta >= 0:
            return 1.0 / (1.0 + _math.exp(-eta))
        e = _math.exp(eta)
        return e / (1.0 + e)

    def dmu_deta(self, eta):
        p = self.ginv(eta)
        return p * (1.0 - p)


class _LogLink(_Link):
    name = "log"

    def g(self, mu):
        return _math.log(mu)

    def ginv(self, eta):
        return _math.exp(_bi.min(eta, 700.0))

    def dmu_deta(self, eta):
        return self.ginv(eta)


class _IdentityLink(_Link):
    name = "identity"

    def g(self, mu):
        return mu

    def ginv(self, eta):
        return eta

    def dmu_deta(self, eta):
        return 1.0


class _InverseLink(_Link):
    name = "inverse_power"

    def g(self, mu):
        return 1.0 / mu

    def ginv(self, eta):
        return 1.0 / eta

    def dmu_deta(self, eta):
        return -1.0 / (eta * eta)


class Family:
    links = ()

    def __init__(self, link=None):
        self.link = link or self.default_link()


class Gaussian(Family):
    name = "gaussian"

    @staticmethod
    def default_link():
        return _IdentityLink()

    @staticmethod
    def variance(mu):
        return 1.0

    @staticmethod
    def loglike_obs(y, mu, scale):
        return -0.5 * ((y - mu) ** 2 / scale
                       + _math.log(2.0 * _math.pi * scale))


class Binomial(Family):
    name = "binomial"

    @staticmethod
    def default_link():
        return _LogitLink()

    @staticmethod
    def variance(mu):
        return mu * (1.0 - mu)

    @staticmethod
    def loglike_obs(y, mu, scale):
        mu = _bi.min(_bi.max(mu, 1e-10), 1.0 - 1e-10)
        return y * _math.log(mu) + (1.0 - y) * _math.log(1.0 - mu)


class Poisson(Family):
    name = "poisson"

    @staticmethod
    def default_link():
        return _LogLink()

    @staticmethod
    def variance(mu):
        return mu

    @staticmethod
    def loglike_obs(y, mu, scale):
        return y * _math.log(_bi.max(mu, 1e-300)) - mu \
            - _math.lgamma(y + 1.0)


class Gamma(Family):
    name = "gamma"

    @staticmethod
    def default_link():
        return _InverseLink()

    @staticmethod
    def variance(mu):
        return mu * mu

    @staticmethod
    def loglike_obs(y, mu, scale):
        a = 1.0 / scale
        return (a * _math.log(a * y / mu) - a * y / mu
                - _math.log(y) - _math.lgamma(a))


class _FamiliesNS:
    Gaussian = Gaussian
    Binomial = Binomial
    Poisson = Poisson
    Gamma = Gamma

    class links:
        Logit = _LogitLink
        logit = _LogitLink
        Log = _LogLink
        log = _LogLink
        Identity = _IdentityLink
        identity = _IdentityLink
        InversePower = _InverseLink


families = _FamiliesNS()


# ------------------------------------------------------------ results

class RegressionResults:
    def __init__(self, model, params, cov, df_resid, scale,
                 fittedvalues, resid, llf, exog_names):
        self.model = model
        self.params = _NamedVec(params, exog_names)
        self.cov = cov
        self.df_resid = df_resid
        self.scale = scale
        self.fittedvalues = _ac.marr(fittedvalues)
        self.resid = _ac.marr(resid)
        self.llf = llf
        self.exog_names = exog_names
        k = len(params)
        self.df_model = k - (1 if "const" in exog_names else 0)
        self.nobs = df_resid + k
        se = [_math.sqrt(_bi.max(cov[i][i], 0.0)) for i in range(k)]
        self.bse = _NamedVec(se, exog_names)
        tv = [params[i] / se[i] if se[i] > 0 else float("nan")
              for i in range(k)]
        self.tvalues = _NamedVec(tv, exog_names)
        if getattr(model, "_use_t", True):
            pv = [2.0 * _stats.t.sf(abs(v), df_resid) for v in tv]
        else:
            pv = [2.0 * _stats.norm.sf(abs(v)) for v in tv]
        self.pvalues = _NamedVec(pv, exog_names)
        self.aic = -2.0 * llf + 2.0 * k
        self.bic = -2.0 * llf + k * _math.log(self.nobs)

    def conf_int(self, alpha=0.05):
        if getattr(self.model, "_use_t", True):
            crit = _stats.t.ppf(1.0 - alpha / 2.0, self.df_resid)
        else:
            crit = _stats.norm.ppf(1.0 - alpha / 2.0)
        p = list(self.params._flat())
        s = list(self.bse._flat())
        return _ac.marr([[p[i] - crit * s[i], p[i] + crit * s[i]]
                         for i in range(len(p))])

    def predict(self, exog=None):
        return self.model.predict(list(self.params._flat()), exog)

    def summary(self):
        lines = ["%-14s %10s %10s %10s %10s" % (
            "", "coef", "std err", "t", "P>|t|")]
        p = list(self.params._flat())
        s = list(self.bse._flat())
        tv = list(self.tvalues._flat())
        pv = list(self.pvalues._flat())
        for i, nm in enumerate(self.exog_names):
            lines.append("%-14s %10.4f %10.4f %10.3f %10.4f" % (
                nm[:14], p[i], s[i], tv[i], pv[i]))
        return "\n".join(lines)


class _NamedVec(_ac.marr):
    """1-D marr with name-based item access (params['x1'])."""

    def __init__(self, data, names):
        super().__init__([float(v) for v in data])
        self._names = list(names)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[self._names.index(key)]
        return super().__getitem__(key)

    @property
    def index(self):
        return list(self._names)

    def to_dict(self):
        return dict(zip(self._names, self.data))


# ------------------------------------------------------------ models

class OLS:
    _use_t = True

    def __init__(self, endog, exog, weights=None, exog_names=None):
        self.y = _tolist1(endog)
        self.X = _tolists(exog)
        self.w = [float(v) for v in weights] if weights is not None \
            else None
        n = len(self.y)
        k = len(self.X[0])
        self.exog_names = exog_names or (
            ["const"] + ["x%d" % i for i in range(1, k)]
            if all(r[0] == 1.0 for r in self.X)
            else ["x%d" % (i + 1) for i in range(k)])
        self.n, self.k = n, k

    def fit(self, cov_type="nonrobust", **kw):
        del kw
        X, y = self.X, self.y
        n, k = self.n, self.k
        w = self.w or [1.0] * n
        XtWX = [[_math.fsum(w[r] * X[r][i] * X[r][j]
                            for r in range(n))
                 for j in range(k)] for i in range(k)]
        XtWy = [_math.fsum(w[r] * X[r][i] * y[r] for r in range(n))
                for i in range(k)]
        beta = list(_ac.linalg.solve(_ac.marr(XtWX),
                                     _ac.marr(XtWy))._flat())
        fitted = [_math.fsum(X[r][j] * beta[j] for j in range(k))
                  for r in range(n)]
        resid = [y[r] - fitted[r] for r in range(n)]
        df_resid = n - k
        ssr = _math.fsum(w[r] * resid[r] ** 2 for r in range(n))
        scale = ssr / df_resid
        XtWXinv = _ac.linalg.inv(_ac.marr(XtWX)).tolist()
        if cov_type in ("HC0", "HC1", "HC2", "HC3"):
            # sandwich: (X'X)^-1 X' diag(u_i^2 adj) X (X'X)^-1
            meat = [[0.0] * k for _ in range(k)]
            H = None
            if cov_type in ("HC2", "HC3"):
                H = [
                    _math.fsum(X[r][i] * XtWXinv[i][j] * X[r][j]
                               for i in range(k) for j in range(k))
                    for r in range(n)]
            for r in range(n):
                u2 = resid[r] ** 2
                if cov_type == "HC1":
                    u2 *= n / df_resid
                elif cov_type == "HC2":
                    u2 /= (1.0 - H[r])
                elif cov_type == "HC3":
                    u2 /= (1.0 - H[r]) ** 2
                for i in range(k):
                    for j in range(k):
                        meat[i][j] += X[r][i] * u2 * X[r][j]
            cov = [[_math.fsum(XtWXinv[i][a] * meat[a][b]
                               * XtWXinv[b][j]
                               for a in range(k) for b in range(k))
                    for j in range(k)] for i in range(k)]
        else:
            cov = [[XtWXinv[i][j] * scale for j in range(k)]
                   for i in range(k)]
        # gaussian loglik (weights folded into ssr)
        llf = -0.5 * n * (_math.log(2.0 * _math.pi * ssr / n) + 1.0)
        res = RegressionResults(self, beta, cov, df_resid, scale,
                                fitted, resid, llf, self.exog_names)
        ybar = _math.fsum(w[r] * y[r] for r in range(n)) \
            / _math.fsum(w)
        tss = _math.fsum(w[r] * (y[r] - ybar) ** 2 for r in range(n))
        res.rsquared = 1.0 - ssr / tss if tss > 0 else float("nan")
        res.rsquared_adj = 1.0 - (1.0 - res.rsquared) * (n - 1) \
            / df_resid
        res.ssr = ssr
        res.centered_tss = tss
        res.fvalue = (res.rsquared / res.df_model) / (
            (1.0 - res.rsquared) / df_resid) \
            if res.df_model > 0 else float("nan")
        res.f_pvalue = _stats.f.sf(res.fvalue, res.df_model,
                                   df_resid) \
            if res.df_model > 0 else float("nan")
        return res

    def predict(self, params, exog=None):
        X = self.X if exog is None else _tolists(exog)
        return _ac.marr([
            _math.fsum(row[j] * params[j] for j in range(len(params)))
            for row in X])


def WLS(endog, exog, weights=None, **kw):
    del kw
    return OLS(endog, exog, weights=weights)


class GLM:
    _use_t = False

    def __init__(self, endog, exog, family=None, exog_names=None):
        self.y = _tolist1(endog)
        self.X = _tolists(exog)
        self.family = family or Gaussian()
        n = len(self.y)
        k = len(self.X[0])
        self.exog_names = exog_names or (
            ["const"] + ["x%d" % i for i in range(1, k)]
            if all(r[0] == 1.0 for r in self.X)
            else ["x%d" % (i + 1) for i in range(k)])
        self.n, self.k = n, k

    def fit(self, maxiter=100, tol=1e-10, **kw):
        del kw
        X, y = self.X, self.y
        n, k = self.n, self.k
        fam = self.family
        link = fam.link
        beta = [0.0] * k
        # sane start for intercept-only direction
        if isinstance(fam, Binomial):
            p0 = _bi.min(_bi.max(_math.fsum(y) / n, 0.01), 0.99)
            beta[0] = link.g(p0) if self.exog_names[0] == "const" \
                else 0.0
        elif isinstance(fam, (Poisson, Gamma)):
            m0 = _bi.max(_math.fsum(y) / n, 1e-6)
            beta[0] = link.g(m0) if self.exog_names[0] == "const" \
                else 0.0
        llf_old = -_math.inf
        for _it in range(maxiter):
            eta = [_math.fsum(X[r][j] * beta[j] for j in range(k))
                   for r in range(n)]
            mu = [link.ginv(e) for e in eta]
            dmu = [link.dmu_deta(e) for e in eta]
            var = [_bi.max(fam.variance(m), 1e-12) for m in mu]
            wirls = [dmu[r] ** 2 / var[r] for r in range(n)]
            z = [eta[r] + (y[r] - mu[r]) / dmu[r]
                 if abs(dmu[r]) > 1e-300 else eta[r]
                 for r in range(n)]
            XtWX = [[_math.fsum(wirls[r] * X[r][i] * X[r][j]
                                for r in range(n))
                     for j in range(k)] for i in range(k)]
            XtWz = [_math.fsum(wirls[r] * X[r][i] * z[r]
                               for r in range(n)) for i in range(k)]
            beta = list(_ac.linalg.solve(_ac.marr(XtWX),
                                         _ac.marr(XtWz))._flat())
            llf = _math.fsum(fam.loglike_obs(y[r], link.ginv(
                _math.fsum(X[r][j] * beta[j] for j in range(k))),
                1.0) for r in range(n))
            if abs(llf - llf_old) < tol * (abs(llf) + 1.0):
                break
            llf_old = llf
        eta = [_math.fsum(X[r][j] * beta[j] for j in range(k))
               for r in range(n)]
        mu = [link.ginv(e) for e in eta]
        dmu = [link.dmu_deta(e) for e in eta]
        var = [_bi.max(fam.variance(m), 1e-12) for m in mu]
        wirls = [dmu[r] ** 2 / var[r] for r in range(n)]
        XtWX = [[_math.fsum(wirls[r] * X[r][i] * X[r][j]
                            for r in range(n))
                 for j in range(k)] for i in range(k)]
        cov = _ac.linalg.inv(_ac.marr(XtWX)).tolist()
        df_resid = n - k
        # scale: 1 for binomial/poisson; pearson X2/df for others
        if isinstance(fam, (Binomial, Poisson)):
            scale = 1.0
        else:
            scale = _math.fsum((y[r] - mu[r]) ** 2 / var[r]
                               for r in range(n)) / df_resid
            cov = [[c * scale for c in row] for row in cov]
        resid = [y[r] - mu[r] for r in range(n)]
        llf = _math.fsum(fam.loglike_obs(y[r], mu[r], scale)
                         for r in range(n))
        res = RegressionResults(self, beta, cov, df_resid, scale,
                                mu, resid, llf, self.exog_names)
        res.mu = _ac.marr(mu)
        res.deviance = self._deviance(y, mu)
        res.pearson_chi2 = _math.fsum(
            (y[r] - mu[r]) ** 2 / var[r] for r in range(n))
        return res

    def _deviance(self, y, mu):
        fam = self.family
        if isinstance(fam, Binomial):
            d = 0.0
            for yr, mr in zip(y, mu):
                mr = _bi.min(_bi.max(mr, 1e-10), 1.0 - 1e-10)
                if yr > 0:
                    d += yr * _math.log(yr / mr)
                if yr < 1:
                    d += (1.0 - yr) * _math.log((1.0 - yr)
                                                / (1.0 - mr))
            return 2.0 * d
        if isinstance(fam, Poisson):
            d = 0.0
            for yr, mr in zip(y, mu):
                if yr > 0:
                    d += yr * _math.log(yr / mr) - (yr - mr)
                else:
                    d += mr
            return 2.0 * d
        return _math.fsum((yr - mr) ** 2 for yr, mr in zip(y, mu))

    def predict(self, params, exog=None):
        X = self.X if exog is None else _tolists(exog)
        link = self.family.link
        return _ac.marr([link.ginv(
            _math.fsum(row[j] * params[j]
                       for j in range(len(params)))) for row in X])


def Logit(endog, exog, **kw):
    del kw
    return GLM(endog, exog, family=Binomial())


class RLM:
    """Huber-M robust linear model (IRLS with Huber weights)."""

    _use_t = False

    def __init__(self, endog, exog, M=None, exog_names=None):
        del M
        self.y = _tolist1(endog)
        self.X = _tolists(exog)
        k = len(self.X[0])
        self.exog_names = exog_names or ["x%d" % (i + 1)
                                         for i in range(k)]

    def fit(self, maxiter=100, **kw):
        del kw
        X, y = self.X, self.y
        n, k = len(y), len(X[0])
        c = 1.345
        beta = [0.0] * k
        for _ in range(maxiter):
            fitted = [_math.fsum(X[r][j] * beta[j]
                                 for j in range(k)) for r in range(n)]
            resid = [y[r] - fitted[r] for r in range(n)]
            ar = sorted(abs(v) for v in resid)
            mad = ar[n // 2] / 0.6745 if ar[n // 2] > 0 else 1.0
            w = [1.0 if abs(resid[r]) <= c * mad
                 else c * mad / abs(resid[r]) for r in range(n)]
            XtWX = [[_math.fsum(w[r] * X[r][i] * X[r][j]
                                for r in range(n))
                     for j in range(k)] for i in range(k)]
            XtWy = [_math.fsum(w[r] * X[r][i] * y[r]
                               for r in range(n)) for i in range(k)]
            newb = list(_ac.linalg.solve(_ac.marr(XtWX),
                                         _ac.marr(XtWy))._flat())
            if max(abs(a - b) for a, b in zip(newb, beta)) < 1e-10:
                beta = newb
                break
            beta = newb
        fitted = [_math.fsum(X[r][j] * beta[j] for j in range(k))
                  for r in range(n)]
        resid = [y[r] - fitted[r] for r in range(n)]
        scale = sorted(abs(v) for v in resid)[n // 2] / 0.6745
        XtXinv = _ac.linalg.inv(_ac.marr(
            [[_math.fsum(X[r][i] * X[r][j] for r in range(n))
              for j in range(k)] for i in range(k)])).tolist()
        cov = [[XtXinv[i][j] * scale * scale for j in range(k)]
               for i in range(k)]
        llf = float("nan")
        return RegressionResults(self, beta, cov, n - k, scale,
                                 fitted, resid, llf, self.exog_names)


# ------------------------------------------------------------ formula API

def _parse_formula(formula, data):
    """'y ~ x1 + x2 + C(g) + x1:x2 - 1' -> (y, X, names)."""
    lhs, rhs = [s.strip() for s in formula.split("~")]
    y = [float(v) for v in data[lhs]]
    terms = [t.strip() for t in rhs.split("+")]
    intercept = True
    cols = []
    names = []
    n = len(y)
    for t in terms:
        if t in ("1", ""):
            continue
        if t in ("-1", "0"):
            intercept = False
            continue
        if t.startswith("- 1") or t.startswith("-1"):
            intercept = False
            continue
        if ":" in t and not t.startswith("C("):
            a, b = [s.strip() for s in t.split(":")]
            va = [float(v) for v in data[a]]
            vb = [float(v) for v in data[b]]
            cols.append([va[i] * vb[i] for i in range(n)])
            names.append("%s:%s" % (a, b))
        elif t.startswith("C(") and t.endswith(")"):
            var = t[2:-1].strip()
            vals = list(data[var])
            levels = sorted(set(vals), key=str)
            for lev in levels[1:]:
                cols.append([1.0 if v == lev else 0.0 for v in vals])
                names.append("C(%s)[T.%s]" % (var, lev))
        else:
            cols.append([float(v) for v in data[t]])
            names.append(t)
    if intercept:
        cols = [[1.0] * n] + cols
        names = ["Intercept"] + names
    X = [[cols[j][i] for j in range(len(cols))] for i in range(n)]
    return y, X, names


class _Smf:
    @staticmethod
    def ols(formula, data):
        y, X, names = _parse_formula(formula, data)
        return OLS(y, X, exog_names=names)

    @staticmethod
    def wls(formula, data, weights=None):
        y, X, names = _parse_formula(formula, data)
        w = [float(v) for v in weights] if weights is not None \
            else None
        return OLS(y, X, weights=w, exog_names=names)

    @staticmethod
    def glm(formula, data, family=None):
        y, X, names = _parse_formula(formula, data)
        return GLM(y, X, family=family, exog_names=names)

    @staticmethod
    def logit(formula, data):
        y, X, names = _parse_formula(formula, data)
        return GLM(y, X, family=Binomial(), exog_names=names)


formula = _Smf()
ols = _Smf.ols
wls = _Smf.wls
glm = _Smf.glm


# ------------------------------------------------------------ power

def _z_crit(alpha, alternative):
    if alternative == "two-sided":
        return _stats.norm.ppf(1.0 - alpha / 2.0)
    return _stats.norm.ppf(1.0 - alpha)


class NormalIndPower:
    """Two-sample z-test power (statsmodels parametrization)."""

    def power(self, effect_size, nobs1, alpha, ratio=1.0,
              alternative="two-sided"):
        n1 = float(nobs1)
        delta = effect_size * _math.sqrt(
            n1 * ratio / (1.0 + ratio))
        zc = _z_crit(alpha, alternative)
        if alternative == "two-sided":
            return (_stats.norm.sf(zc - delta)
                    + _stats.norm.cdf(-zc - delta))
        if alternative == "larger":
            return _stats.norm.sf(zc - delta)
        return _stats.norm.cdf(-zc - delta)

    def solve_power(self, effect_size=None, nobs1=None, alpha=0.05,
                    power=None, ratio=1.0,
                    alternative="two-sided"):
        if nobs1 is None:
            lo, hi = 2.0, 1e7
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if self.power(effect_size, mid, alpha, ratio,
                              alternative) < power:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        if effect_size is None:
            lo, hi = 1e-8, 10.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if self.power(mid, nobs1, alpha, ratio,
                              alternative) < power:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        return self.power(effect_size, nobs1, alpha, ratio,
                          alternative)


class TTestPower:
    """One-sample / paired t-test power via noncentral t."""

    def power(self, effect_size, nobs, alpha,
              alternative="two-sided"):
        n = float(nobs)
        df = n - 1.0
        nc = effect_size * _math.sqrt(n)
        if alternative == "two-sided":
            tc = _stats.t.ppf(1.0 - alpha / 2.0, df)
            return (_stats.nct.sf(tc, df, nc)
                    + _stats.nct.cdf(-tc, df, nc))
        tc = _stats.t.ppf(1.0 - alpha, df)
        if alternative == "larger":
            return _stats.nct.sf(tc, df, nc)
        return _stats.nct.cdf(-tc, df, nc)

    def solve_power(self, effect_size=None, nobs=None, alpha=0.05,
                    power=None, alternative="two-sided"):
        if nobs is None:
            lo, hi = 3.0, 1e6
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if self.power(effect_size, mid, alpha,
                              alternative) < power:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        return self.power(effect_size, nobs, alpha, alternative)


class TTestIndPower:
    """Two-sample t-test power via noncentral t."""

    def power(self, effect_size, nobs1, alpha, ratio=1.0,
              alternative="two-sided"):
        n1 = float(nobs1)
        n2 = n1 * ratio
        df = n1 + n2 - 2.0
        nc = effect_size * _math.sqrt(n1 * n2 / (n1 + n2))
        if alternative == "two-sided":
            tc = _stats.t.ppf(1.0 - alpha / 2.0, df)
            return (_stats.nct.sf(tc, df, nc)
                    + _stats.nct.cdf(-tc, df, nc))
        tc = _stats.t.ppf(1.0 - alpha, df)
        if alternative == "larger":
            return _stats.nct.sf(tc, df, nc)
        return _stats.nct.cdf(-tc, df, nc)

    def solve_power(self, effect_size=None, nobs1=None, alpha=0.05,
                    power=None, ratio=1.0,
                    alternative="two-sided"):
        if nobs1 is None:
            lo, hi = 3.0, 1e6
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if self.power(effect_size, mid, alpha, ratio,
                              alternative) < power:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        return self.power(effect_size, nobs1, alpha, ratio,
                          alternative)


class FTestAnovaPower:
    """One-way ANOVA power via noncentral F (Cohen's f)."""

    def power(self, effect_size, nobs, alpha, k_groups=2):
        n = float(nobs)
        dfn = k_groups - 1.0
        dfd = n - k_groups
        nc = effect_size ** 2 * n
        fc = _stats.f.ppf(1.0 - alpha, dfn, dfd)
        return _stats.ncf.sf(fc, dfn, dfd, nc)

    def solve_power(self, effect_size=None, nobs=None, alpha=0.05,
                    power=None, k_groups=2):
        if nobs is None:
            lo, hi = k_groups + 2.0, 1e6
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if self.power(effect_size, mid, alpha,
                              k_groups) < power:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        return self.power(effect_size, nobs, alpha, k_groups)


class power:  # namespace mirror for statsmodels.stats.power
    NormalIndPower = NormalIndPower
    TTestPower = TTestPower
    TTestIndPower = TTestIndPower
    FTestAnovaPower = FTestAnovaPower


def proportion_effectsize(prop1, prop2):
    return 2.0 * _math.asin(_math.sqrt(prop1)) \
        - 2.0 * _math.asin(_math.sqrt(prop2))
