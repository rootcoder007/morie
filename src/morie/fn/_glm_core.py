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


def _frame_rows(X):
    """Rows of floats from native OR real-pandas frames, else None."""
    if hasattr(X, "_cols"):            # native DataFrame
        cols = list(X._cols.values())
        return [[float(c[i]) for c in cols]
                for i in range(X.shape[0])]
    if hasattr(X, "columns") and hasattr(X, "values"):  # real pandas
        return [[float(v) for v in row] for row in X.values.tolist()]
    return None


def _tolists(X):
    rows = _frame_rows(X)
    if rows is not None:
        return rows
    if hasattr(X, "_data") and hasattr(X, "index"):  # native Series
        return [[float(v)] for v in X._data]
    a = _ac.atleast_2d(X)
    return [list(map(float, r)) for r in a.data]


def _tolist1(y):
    if hasattr(y, "_data") and hasattr(y, "index"):  # native Series
        return [float(v) for v in y._data]
    if hasattr(y, "values") and hasattr(y, "index"):  # real pandas
        return [float(v) for v in y.values.tolist()]
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
    frows = _frame_rows(X)
    if frows is not None:
        rows = frows
    else:
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
        if getattr(model, "_frame_in", False):
            from . import _frame_core as _fc
            self.fittedvalues = _fc.Series(list(fittedvalues))
            self.resid = _fc.Series(list(resid))
        else:
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
        self._frame_in = hasattr(exog, "_cols") or (
            hasattr(exog, "columns") and hasattr(exog, "values"))
        if self._frame_in and exog_names is None:
            exog_names = [str(c) for c in (
                exog._cols if hasattr(exog, "_cols")
                else list(exog.columns))]
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
        self._frame_in = hasattr(exog, "_cols") or (
            hasattr(exog, "columns") and hasattr(exog, "values"))
        if self._frame_in and exog_names is None:
            exog_names = [str(c) for c in (
                exog._cols if hasattr(exog, "_cols")
                else list(exog.columns))]
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


# ------------------------------------------------------------ formula *

def _expand_star(rhs):
    """Expand a*b into a + b + a:b inside a formula RHS."""
    out = []
    for t in [s.strip() for s in rhs.split("+")]:
        if "*" in t and not t.startswith("- "):
            a, b = [s.strip() for s in t.split("*", 1)]
            out += [a, b, "%s:%s" % (a, b)]
        else:
            out.append(t)
    return " + ".join(out)


_parse_formula_base = _parse_formula


def _parse_formula(formula, data):  # noqa: F811
    lhs, rhs = [s.strip() for s in formula.split("~")]
    return _parse_formula_base("%s ~ %s" % (lhs, _expand_star(rhs)),
                               data)


def _cat_interaction_cols(t, data, n):
    """C(a):C(b) -> product dummies with patsy-style names."""
    left, right = [s.strip() for s in t.split(":")]
    la = left[2:-1].strip()
    lb = right[2:-1].strip()
    va = list(data[la])
    vb = list(data[lb])
    lev_a = sorted(set(va), key=str)[1:]
    lev_b = sorted(set(vb), key=str)[1:]
    cols = []
    names = []
    for A in lev_a:
        for B in lev_b:
            cols.append([1.0 if va[i] == A and vb[i] == B else 0.0
                         for i in range(n)])
            names.append("C(%s)[T.%s]:C(%s)[T.%s]" % (la, A, lb, B))
    return cols, names


# patch base parser to handle C():C() interactions
_parse_formula_base2 = _parse_formula_base


def _parse_formula_base(formula, data):  # noqa: F811
    lhs, rhs = [s.strip() for s in formula.split("~")]
    y = [float(v) for v in data[lhs]]
    n = len(y)
    terms = [t.strip() for t in rhs.split("+")]
    plain = []
    extra_cols = []
    extra_names = []
    for t in terms:
        if ":" in t and t.count("C(") == 2:
            cols, names = _cat_interaction_cols(t, data, n)
            extra_cols += cols
            extra_names += names
        else:
            plain.append(t)
    y2, X, names = _parse_formula_base2(
        "%s ~ %s" % (lhs, " + ".join(plain) if plain else "1"), data)
    for c, nm in zip(extra_cols, extra_names):
        for i in range(n):
            X[i].append(c[i])
        names.append(nm)
    return y2, X, names


# ------------------------------------------------------------ anova_lm

def anova_lm(fit, typ=2):
    """Type-II ANOVA table for a formula-fitted OLS result."""
    from . import _frame_core as _fc
    model = fit.model
    X = model.X
    y = model.y
    names = model.exog_names
    n = len(y)

    def rss_of(cols):
        if not cols:
            ybar = _math.fsum(y) / n
            return _math.fsum((v - ybar) ** 2 for v in y)
        k = len(cols)
        A = [[_math.fsum(X[r][i] * X[r][j] for r in range(n))
              for j in cols] for i in cols]
        b = [_math.fsum(X[r][i] * y[r] for r in range(n))
             for i in cols]
        beta = list(_ac.linalg.solve(_ac.marr(A),
                                     _ac.marr(b))._flat())
        return _math.fsum(
            (y[r] - _math.fsum(X[r][cols[j]] * beta[j]
                               for j in range(k))) ** 2
            for r in range(n))

    # group columns into terms by patsy-style base name
    def term_of(nm):
        if nm == "Intercept":
            return None
        if ":" in nm:
            parts = nm.split(":")
            return ":".join(p.split("[")[0] for p in parts)
        return nm.split("[")[0]

    terms = []
    cols_by_term = {}
    for j, nm in enumerate(names):
        t = term_of(nm)
        if t is None:
            continue
        if t not in cols_by_term:
            cols_by_term[t] = []
            terms.append(t)
        cols_by_term[t].append(j)
    full = list(range(len(names)))
    rss_full = rss_of(full)
    df_resid = n - len(names)
    del typ
    rows_ss, rows_df, rows_F, rows_p = [], [], [], []
    for t in terms:
        # type II: drop the term AND any higher-order term containing it
        drop = set(cols_by_term[t])
        base_parts = set(t.split(":"))
        keep_hi = []
        for u in terms:
            if u != t and base_parts < set(u.split(":")):
                keep_hi += cols_by_term[u]
        reduced = [j for j in full
                   if j not in drop and j not in set(keep_hi)]
        comparison = [j for j in full if j not in set(keep_hi)]
        ss = rss_of(reduced) - rss_of(comparison)
        dft = len(cols_by_term[t])
        Fv = (ss / dft) / (rss_full / df_resid)
        rows_ss.append(ss)
        rows_df.append(float(dft))
        rows_F.append(Fv)
        rows_p.append(_stats.f.sf(Fv, dft, df_resid))
    rows_ss.append(rss_full)
    rows_df.append(float(df_resid))
    rows_F.append(float("nan"))
    rows_p.append(float("nan"))
    df = _fc.DataFrame({"sum_sq": rows_ss, "df": rows_df,
                        "F": rows_F, "PR(>F)": rows_p},
                       index=terms + ["Residual"])
    return df


class _StatsNS:
    anova_lm = staticmethod(anova_lm)


stats = _StatsNS()


# ------------------------------------------------------------ IV2SLS

class IV2SLS:
    """Two-stage least squares, statsmodels sandbox signature:
    IV2SLS(endog_y, exog, instrument)."""

    _use_t = True

    def __init__(self, endog, exog, instrument):
        self.y = _tolist1(endog)
        self.X = _tolists(exog)
        self.Z = _tolists(instrument)

    def fit(self, **kw):
        del kw
        y, X, Z = self.y, self.X, self.Z
        n = len(y)
        kx = len(X[0])
        kz = len(Z[0])
        ZtZ = [[_math.fsum(Z[r][i] * Z[r][j] for r in range(n))
                for j in range(kz)] for i in range(kz)]
        ZtZinv = _ac.linalg.inv(_ac.marr(ZtZ)).tolist()
        ZtX = [[_math.fsum(Z[r][i] * X[r][j] for r in range(n))
                for j in range(kx)] for i in range(kz)]
        Zty = [_math.fsum(Z[r][i] * y[r] for r in range(n))
               for i in range(kz)]
        # Xhat'X = X'Z (Z'Z)^-1 Z'X ; Xhat'y likewise
        XtPZX = [[_math.fsum(ZtX[a][i] * ZtZinv[a][b] * ZtX[b][j]
                             for a in range(kz) for b in range(kz))
                  for j in range(kx)] for i in range(kx)]
        XtPZy = [_math.fsum(ZtX[a][i] * ZtZinv[a][b] * Zty[b]
                            for a in range(kz) for b in range(kz))
                 for i in range(kx)]
        beta = list(_ac.linalg.solve(_ac.marr(XtPZX),
                                     _ac.marr(XtPZy))._flat())
        fitted = [_math.fsum(X[r][j] * beta[j] for j in range(kx))
                  for r in range(n)]
        resid = [y[r] - fitted[r] for r in range(n)]
        df_resid = n - kx
        sigma2 = _math.fsum(v * v for v in resid) / df_resid
        cov = [[_ac.linalg.inv(_ac.marr(XtPZX)).tolist()[i][j]
                * sigma2 for j in range(kx)] for i in range(kx)]
        names = ["x%d" % (i + 1) for i in range(kx)]
        llf = float("nan")
        return RegressionResults(self, beta, cov, df_resid, sigma2,
                                 fitted, resid, llf, names)

    def predict(self, params, exog=None):
        X = self.X if exog is None else _tolists(exog)
        return _ac.marr([
            _math.fsum(row[j] * params[j]
                       for j in range(len(params))) for row in X])


# ------------------------------------------------------------ tukey

def _ptukey_cdf(q, k, df):
    """P(Q <= q) for the studentized range (double quadrature)."""
    if q <= 0:
        return 0.0

    def inner(u):
        # k * int phi(z) [Phi(z) - Phi(z - q*u)]^(k-1) dz
        def f(z):
            base = _stats.norm.cdf(z) - _stats.norm.cdf(z - q * u)
            if base <= 0:
                return 0.0
            return _math.exp(-0.5 * z * z) / _math.sqrt(
                2.0 * _math.pi) * base ** (k - 1)
        total = 0.0
        m = 160
        lo, hi = -8.0, 8.0 + q * u
        for i in range(m):
            z = lo + (i + 0.5) * (hi - lo) / m
            total += f(z)
        return k * total * (hi - lo) / m
    if df > 200:
        return _bi.min(1.0, inner(1.0))
    # integrate over s ~ chi_df / sqrt(df)
    from ._sci_core import quad as _quad

    def g(s):
        # density of s: 2 (df/2)^(df/2) / Gamma(df/2) s^(df-1) e^{-df s^2/2}
        logd = (_math.log(2.0) + 0.5 * df * _math.log(df / 2.0)
                - _math.lgamma(df / 2.0)
                + (df - 1.0) * _math.log(s) - df * s * s / 2.0)
        if logd < -700:
            return 0.0
        return _math.exp(logd) * inner(s)
    val, _ = _quad(g, 1e-4, 4.0, epsabs=1e-9)
    return _bi.min(1.0, val)


def _qsturng(p, k, df):
    lo, hi = 0.01, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _ptukey_cdf(mid, k, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class _TukeyResult:
    def __init__(self, rows, groups):
        self._rows = rows
        self.groupsunique = groups

    def summary(self):
        lines = ["%-10s %-10s %9s %9s %9s %7s" % (
            "group1", "group2", "meandiff", "lower", "upper",
            "reject")]
        for r in self._rows:
            lines.append("%-10s %-10s %9.4f %9.4f %9.4f %7s" % (
                str(r["group1"])[:10], str(r["group2"])[:10],
                r["meandiff"], r["lower"], r["upper"],
                str(r["reject"])))
        return "\n".join(lines)

    @property
    def reject(self):
        return [r["reject"] for r in self._rows]

    @property
    def meandiffs(self):
        return _ac.marr([r["meandiff"] for r in self._rows])

    @property
    def pvalues(self):
        return _ac.marr([r["p-adj"] for r in self._rows])


def pairwise_tukeyhsd(endog, groups, alpha=0.05):
    yv = _tolist1(endog)
    gv = list(groups.tolist() if hasattr(groups, "tolist")
              else groups)
    uniq = sorted(set(gv), key=str)
    k = len(uniq)
    n = len(yv)
    means = {}
    ns = {}
    for g in uniq:
        vals = [yv[i] for i in range(n) if gv[i] == g]
        means[g] = _math.fsum(vals) / len(vals)
        ns[g] = len(vals)
    df = n - k
    mse = _math.fsum((yv[i] - means[gv[i]]) ** 2
                     for i in range(n)) / df
    qcrit = _qsturng(1.0 - alpha, k, df)
    rows = []
    for a in range(k - 1):
        for b in range(a + 1, k):
            g1, g2 = uniq[a], uniq[b]
            diff = means[g2] - means[g1]
            se = _math.sqrt(mse / 2.0 * (1.0 / ns[g1]
                                         + 1.0 / ns[g2]))
            hw = qcrit * se
            qstat = abs(diff) / se
            padj = 1.0 - _ptukey_cdf(qstat, k, df)
            rows.append({"group1": g1, "group2": g2,
                         "meandiff": diff, "lower": diff - hw,
                         "upper": diff + hw,
                         "reject": abs(diff) > hw,
                         "p-adj": _bi.max(0.0, _bi.min(1.0, padj))})
    return _TukeyResult(rows, uniq)


class multicomp:
    pairwise_tukeyhsd = staticmethod(pairwise_tukeyhsd)


stats.multicomp = multicomp


# ------------------------------------------------------------ lowess/KDE

def lowess(endog, exog, frac=2.0 / 3.0, it=3, return_sorted=True):
    """Cleveland LOWESS: local linear, tricube weights, robustness
    iterations."""
    yv = _tolist1(endog)
    xv = _tolist1(exog)
    n = len(xv)
    order = sorted(range(n), key=lambda i: xv[i])
    xs = [xv[i] for i in order]
    ys = [yv[i] for i in order]
    r = _bi.max(2, int(_math.ceil(frac * n)))
    delta = [1.0] * n
    fitted = [0.0] * n
    for _iter in range(it + 1):
        for i in range(n):
            dists = sorted(abs(xs[j] - xs[i]) for j in range(n))
            h = dists[_bi.min(r - 1, n - 1)] or 1e-300
            w = []
            for j in range(n):
                u = abs(xs[j] - xs[i]) / h
                wt = (1.0 - u ** 3) ** 3 if u < 1.0 else 0.0
                w.append(wt * delta[j])
            sw = _math.fsum(w)
            if sw <= 0:
                fitted[i] = ys[i]
                continue
            xm = _math.fsum(w[j] * xs[j] for j in range(n)) / sw
            ym = _math.fsum(w[j] * ys[j] for j in range(n)) / sw
            sxx = _math.fsum(w[j] * (xs[j] - xm) ** 2
                             for j in range(n))
            if sxx <= 1e-300:
                fitted[i] = ym
            else:
                b1 = _math.fsum(w[j] * (xs[j] - xm) * (ys[j] - ym)
                                for j in range(n)) / sxx
                fitted[i] = ym + b1 * (xs[i] - xm)
        if _iter == it:
            break
        resid = [ys[j] - fitted[j] for j in range(n)]
        ar = sorted(abs(v) for v in resid)
        s = ar[n // 2] if n % 2 else 0.5 * (ar[n // 2 - 1]
                                            + ar[n // 2])
        if s <= 0:
            break
        delta = [(1.0 - _bi.min(1.0, (abs(resid[j])
                                      / (6.0 * s))) ** 2) ** 2
                 for j in range(n)]
    if return_sorted:
        return _ac.marr([[xs[i], fitted[i]] for i in range(n)])
    inv = [0] * n
    for pos, i in enumerate(order):
        inv[i] = pos
    return _ac.marr([fitted[inv[i]] for i in range(n)])


class KDEUnivariate:
    def __init__(self, endog):
        from . import _stats_core as _sc
        self._data = _tolist1(endog)
        self._kde = None
        self._sc = _sc

    def fit(self, bw=None, **kw):
        del kw
        self._kde = self._sc.gaussian_kde(
            self._data, bw_method=bw if isinstance(bw, float)
            else None)
        lo = min(self._data)
        hi = max(self._data)
        pad = (hi - lo) * 0.1 or 1.0
        self.support = _ac.marr(
            [lo - pad + (hi - lo + 2 * pad) * i / 255.0
             for i in range(256)])
        self.density = _ac.marr(self._kde(list(
            self.support._flat())))
        return self

    def evaluate(self, points):
        return _ac.marr(self._kde(
            [float(v) for v in _ac.asarray(points)._flat()]))


class nonparametric:
    lowess = staticmethod(lowess)
    KDEUnivariate = KDEUnivariate


# ------------------------------------------------------------ GEE

class NegativeBinomial(Family):
    name = "nbinom"

    def __init__(self, alpha=1.0, link=None):
        self.alpha_nb = alpha
        super().__init__(link)

    @staticmethod
    def default_link():
        return _LogLink()

    def variance(self, mu):
        return mu + self.alpha_nb * mu * mu

    def loglike_obs(self, y, mu, scale):
        a = 1.0 / self.alpha_nb
        return (_math.lgamma(y + a) - _math.lgamma(a)
                - _math.lgamma(y + 1.0)
                + a * _math.log(a / (a + mu))
                + y * _math.log(mu / (a + mu) + 1e-300))


families.NegativeBinomial = NegativeBinomial


class Exchangeable:
    name = "exchangeable"


class Independence:
    name = "independence"


class cov_struct:
    Exchangeable = Exchangeable
    Independence = Independence


class GEE:
    """GEE with independence/exchangeable working correlation and
    cluster-robust (sandwich) covariance."""

    _use_t = False

    def __init__(self, endog, exog, groups, family=None,
                 cov_struct=None, exog_names=None):
        self.y = _tolist1(endog)
        self.X = _tolists(exog)
        self.groups = list(groups.tolist()
                           if hasattr(groups, "tolist") else groups)
        self.family = family or Gaussian()
        self.cs = cov_struct or Independence()
        k = len(self.X[0])
        self.exog_names = exog_names or ["x%d" % (i + 1)
                                         for i in range(k)]

    def fit(self, maxiter=60, **kw):
        del kw
        y, X = self.y, self.X
        n = len(y)
        k = len(X[0])
        fam = self.family
        link = fam.link
        clusters = {}
        for i, g in enumerate(self.groups):
            clusters.setdefault(g, []).append(i)
        # start from GLM
        beta = list(GLM(y, X, family=fam,
                        exog_names=self.exog_names)
                    .fit().params._flat())
        rho = 0.0
        for _it in range(maxiter):
            eta = [_math.fsum(X[r][j] * beta[j] for j in range(k))
                   for r in range(n)]
            mu = [link.ginv(e) for e in eta]
            var = [_bi.max(fam.variance(m), 1e-10) for m in mu]
            pres = [(y[r] - mu[r]) / _math.sqrt(var[r])
                    for r in range(n)]
            if isinstance(self.cs, Exchangeable):
                num = 0.0
                cnt = 0
                for idx in clusters.values():
                    m = len(idx)
                    for a in range(m - 1):
                        for b in range(a + 1, m):
                            num += pres[idx[a]] * pres[idx[b]]
                            cnt += 1
                rho = num / _bi.max(cnt - k, 1) if cnt else 0.0
                rho = _bi.max(-0.49, _bi.min(0.99, rho))
            dmu = [link.dmu_deta(e) for e in eta]
            # score + information cluster by cluster
            U = [0.0] * k
            H = [[0.0] * k for _ in range(k)]
            meat = [[0.0] * k for _ in range(k)]
            for idx in clusters.values():
                m = len(idx)
                D = [[X[i][j] * dmu[i] for j in range(k)]
                     for i in idx]
                A = [_math.sqrt(var[i]) for i in idx]
                # R^{-1} for exchangeable: (I - rho J/(1+(m-1)rho))/(1-rho)
                res = [y[i] - mu[i] for i in idx]
                if isinstance(self.cs, Exchangeable) and m > 1 \
                        and abs(rho) > 1e-12:
                    c1 = 1.0 / (1.0 - rho)
                    c2 = -rho / ((1.0 - rho)
                                 * (1.0 + (m - 1) * rho))
                    def rinvv(v):
                        s = _math.fsum(v)
                        return [c1 * v[a] + c2 * s
                                for a in range(m)]
                else:
                    def rinvv(v):
                        return list(v)
                # V^{-1} r = A^-1 R^-1 A^-1 r
                arn = rinvv([res[a] / A[a] for a in range(m)])
                vinv_r = [arn[a] / A[a] for a in range(m)]
                ui = [_math.fsum(D[a][j] * vinv_r[a]
                                 for a in range(m))
                      for j in range(k)]
                for j in range(k):
                    U[j] += ui[j]
                for j1 in range(k):
                    for j2 in range(k):
                        meat[j1][j2] += ui[j1] * ui[j2]
                # H = D' V^-1 D
                for j2 in range(k):
                    col = rinvv([D[a][j2] / A[a] for a in range(m)])
                    vinv_d = [col[a] / A[a] for a in range(m)]
                    for j1 in range(k):
                        H[j1][j2] += _math.fsum(
                            D[a][j1] * vinv_d[a] for a in range(m))
            step = list(_ac.linalg.solve(_ac.marr(H),
                                         _ac.marr(U))._flat())
            beta = [beta[j] + step[j] for j in range(k)]
            if max(abs(s) for s in step) < 1e-10:
                break
        Hinv = _ac.linalg.inv(_ac.marr(H)).tolist()
        cov = [[_math.fsum(Hinv[i][a] * meat[a][b] * Hinv[b][j]
                           for a in range(k) for b in range(k))
                for j in range(k)] for i in range(k)]
        eta = [_math.fsum(X[r][j] * beta[j] for j in range(k))
               for r in range(n)]
        mu = [link.ginv(e) for e in eta]
        resid = [y[r] - mu[r] for r in range(n)]
        res = RegressionResults(self, beta, cov, n - k, 1.0, mu,
                                resid, float("nan"),
                                self.exog_names)
        res.cov_struct_rho = rho
        return res

    def predict(self, params, exog=None):
        X = self.X if exog is None else _tolists(exog)
        link = self.family.link
        return _ac.marr([link.ginv(
            _math.fsum(row[j] * params[j]
                       for j in range(len(params)))) for row in X])


def _smf_gee(formula_str, groups, data, family=None,
             cov_struct=None):
    y, X, names = _parse_formula(formula_str, data)
    g = data[groups] if isinstance(groups, str) else groups
    return GEE(y, X, list(g), family=family, cov_struct=cov_struct,
               exog_names=names)


_Smf.gee = staticmethod(_smf_gee)


# ------------------------------------------------------------ IV (LM API)

class _IVDiag:
    def __init__(self, fstat):
        self._rows = [{"f.stat": fstat}]

    @property
    def iloc(self):
        return self._rows


class _IVFirstStage:
    def __init__(self, fstat):
        self.diagnostics = _IVDiag(fstat)


class _IVLMResults:
    def __init__(self, params, std_errors, fstat):
        self.params = _ac.marr(params)
        self.std_errors = _ac.marr(std_errors)
        self.first_stage = _IVFirstStage(fstat)


class IV2SLS_LM:
    """linearmodels.iv.IV2SLS-compatible wrapper: named kwargs,
    robust (HC0) covariance, first-stage F diagnostic."""

    def __init__(self, dependent, exog, endog, instruments):
        self.y = _tolist1(dependent)
        self.Xex = _tolists(exog)
        self.Xen = _tolists(endog)
        self.Zin = _tolists(instruments)

    def fit(self, cov_type="robust", **kw):
        del kw
        n = len(self.y)
        X = [self.Xex[r] + self.Xen[r] for r in range(n)]
        Z = [self.Xex[r] + self.Zin[r] for r in range(n)]
        base = IV2SLS(self.y, X, Z)
        res = base.fit()
        beta = list(res.params._flat())
        k = len(beta)
        if cov_type == "robust":
            # HC0 sandwich with Xhat = P_Z X
            kz = len(Z[0])
            ZtZ = [[_math.fsum(Z[r][i] * Z[r][j] for r in range(n))
                    for j in range(kz)] for i in range(kz)]
            ZtZinv = _ac.linalg.inv(_ac.marr(ZtZ)).tolist()
            ZtX = [[_math.fsum(Z[r][i] * X[r][j] for r in range(n))
                    for j in range(k)] for i in range(kz)]
            Xhat = [[_math.fsum(Z[r][a] * ZtZinv[a][b] * ZtX[b][j]
                                for a in range(kz)
                                for b in range(kz))
                     for j in range(k)] for r in range(n)]
            resid = [self.y[r] - _math.fsum(X[r][j] * beta[j]
                                            for j in range(k))
                     for r in range(n)]
            bread = _ac.linalg.inv(_ac.marr(
                [[_math.fsum(Xhat[r][i] * Xhat[r][j]
                             for r in range(n))
                  for j in range(k)] for i in range(k)])).tolist()
            meat = [[_math.fsum(Xhat[r][i] * resid[r] ** 2
                                * Xhat[r][j] for r in range(n))
                     for j in range(k)] for i in range(k)]
            cov = [[_math.fsum(bread[i][a] * meat[a][b]
                               * bread[b][j]
                               for a in range(k) for b in range(k))
                    for j in range(k)] for i in range(k)]
            se = [_math.sqrt(_bi.max(cov[i][i], 0.0))
                  for i in range(k)]
        else:
            se = list(res.bse._flat())
        # first-stage F for the instruments (single endog column)
        en = [row[0] for row in self.Xen]
        fs_full = OLS(en, Z).fit()
        fs_red = OLS(en, self.Xex).fit()
        q = len(self.Zin[0])
        dfr = n - len(Z[0])
        F = ((fs_red.ssr - fs_full.ssr) / q) / (fs_full.ssr / dfr) \
            if fs_full.ssr > 0 else float("inf")
        return _IVLMResults(beta, se, F)


# time-series namespace (native _ts_core)
from ._ts_core import (  # noqa: E402
    ARIMA,
    SARIMAX,
    MarkovRegression,
    UnobservedComponents,
    VECM,
    coint_johansen,
    tsa,
)
