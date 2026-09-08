# morie.fn -- shared core (rootcoder007/morie)
"""Formula interface for the native linear and generalised linear models.

This is the native replacement for the statsmodels formula API
(``statsmodels.formula.api``, imported as ``smf``) that morie used to
depend on. It understands the formula shapes morie actually writes --
``y ~ a + b + c``, with an optional ``- 1`` / ``+ 0`` to drop the
intercept -- and routes them into :mod:`morie.fn._glm_core`.

Every model object follows the statsmodels calling convention: the
constructor takes the formula and the data, and ``.fit()`` returns the
result.
"""

import math

from . import _glm_core

__all__ = ["ols", "wls", "glm", "formula_terms"]


def formula_terms(formula):
    """Split ``"y ~ x1 + x2"`` into ``("y", ["x1", "x2"], intercept)``."""
    if "~" not in formula:
        raise ValueError("formula must contain '~': %r" % formula)
    lhs, rhs = formula.split("~", 1)
    outcome = lhs.strip()
    intercept = True
    terms = []
    for raw in rhs.split("+"):
        t = raw.strip()
        if not t or t == "1":
            continue
        if t in ("0", "-1"):
            intercept = False
            continue
        if t.startswith("-"):
            stripped = t[1:].strip()
            if stripped in ("1", "0"):
                intercept = False
                continue
            raise ValueError("unsupported formula term: %r" % t)
        terms.append(t)
    if not outcome:
        raise ValueError("formula has no outcome: %r" % formula)
    return outcome, terms, intercept


def _term_column(term, data):
    """One design column for a term.

    ``a`` is the column itself; ``a:b`` (and ``a:b:c``) is the
    elementwise product of its parts, which is how an interaction is
    coded. The parts must all be columns of ``data``.
    """
    parts = [q.strip() for q in term.split(":")]
    if len(parts) == 1:
        return [float(v) for v in data[parts[0]]]
    col = None
    for q in parts:
        vals = [float(v) for v in data[q]]
        col = vals if col is None else [a * b for a, b in zip(col, vals)]
    return col


def _expand(terms):
    """Expand ``a*b`` into ``a + b + a:b``, keeping first occurrence."""
    out = []
    for t in terms:
        if "*" in t:
            parts = [q.strip() for q in t.split("*")]
            pieces = list(parts)
            for k in range(2, len(parts) + 1):
                pieces += [":".join(c) for c in _combos(parts, k)]
        else:
            pieces = [t]
        for q in pieces:
            if q not in out:
                out.append(q)
    return out


def _combos(seq, k):
    if k == 0:
        return [[]]
    if k > len(seq):
        return []
    out = []
    for i in range(len(seq) - k + 1):
        for rest in _combos(seq[i + 1:], k - 1):
            out.append([seq[i]] + rest)
    return out


def _design(formula, data):
    outcome, terms, intercept = formula_terms(formula)
    terms = _expand(terms)
    y = [float(v) for v in data[outcome]]
    cols = [_term_column(t, data) for t in terms]
    n = len(y)
    for t, c in zip(terms, cols):
        if len(c) != n:
            raise ValueError("term %r has %d rows, outcome has %d"
                             % (t, len(c), n))
    X = [[c[i] for c in cols] for i in range(n)]
    names = (["Intercept"] if intercept else []) + list(terms)
    return y, X, names, intercept


class _ConfIntLoc(object):
    def __init__(self, ci):
        self._ci = ci

    def _row(self, key):
        if isinstance(key, str):
            try:
                return self._ci._rows[self._ci.names.index(key)]
            except ValueError:
                raise KeyError(key)
        return self._ci._rows[key]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            return self._row(row)[int(col)]
        return self._row(key)


class _ConfInt(object):
    """Confidence limits with the two-column frame shape statsmodels
    returns: ``ci[0]`` is the lower-bound COLUMN (indexed by term) and
    ``ci.loc[term, 0]`` is that term's lower bound."""

    def __init__(self, rows, names=()):
        self._rows = [list(r) for r in rows]
        self.names = list(names)

    def __getitem__(self, col):
        if isinstance(col, str):
            raise KeyError("columns are 0 (lower) and 1 (upper)")
        return _NamedVec([r[int(col)] for r in self._rows], self.names)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter((0, 1))          # a frame iterates its columns

    @property
    def index(self):
        return list(self.names)

    @property
    def loc(self):
        return _ConfIntLoc(self)

    @property
    def values(self):
        return [list(r) for r in self._rows]

    def tolist(self):
        return [list(r) for r in self._rows]

    def __repr__(self):
        return "\n".join("%-24s [%.6g, %.6g]" % (n, r[0], r[1])
                          for n, r in zip(self.names, self._rows))


class _NamedVec(list):
    """A coefficient vector addressable by term name.

    statsmodels returns params/bse as a pandas Series indexed by term,
    and morie's call sites use that (``fit.params[treatment]``). Keeping
    list behaviour means positional access still works everywhere else.
    """

    def __new__(cls, values, names=()):
        return super().__new__(cls, values)

    def __init__(self, values, names=()):
        super().__init__(values)
        self._names = list(names)

    @property
    def index(self):
        return list(self._names)

    @property
    def values(self):
        # An array, not a plain list: callers write
        # np.where(fit.pvalues.values < 0.05, ...) and a list has no
        # elementwise comparison.
        from . import _array_core
        return _array_core.marr([float(v) for v in self])

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return list.__getitem__(self, self._names.index(key))
            except ValueError:
                raise KeyError(key)
        return list.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def to_dict(self):
        return dict(zip(self._names, list(self)))


def _as_series(values):
    """Wrap predictions so the caller gets the frame API (.clip etc.)."""
    if values is None:
        return None
    try:
        from . import _frame_core
        return _frame_core.Series(list(values))
    except Exception:
        return list(values)


class _WaldResult(object):
    """What statsmodels' wald_test returns, as far as morie reads it."""

    def __init__(self, statistic, pvalue, df):
        self.statistic = statistic
        self.pvalue = pvalue
        self.df_constraint = df
        self.df_denom = None

    def __repr__(self):
        return "<Wald chi2=%.6g df=%d p=%.6g>" % (
            self.statistic, self.df_constraint, self.pvalue)


class _WaldTerms(object):
    def __init__(self, table):
        self.table = table


class _Result(object):
    """The subset of the statsmodels result API that morie reads."""

    def __init__(self, params, bse, names, nobs, df_resid,
                 tvalues=None, pvalues=None, fittedvalues=None,
                 resid=None, cov=None, extra=None, model=None):
        self.params = _NamedVec(params, names)
        self.bse = _NamedVec(bse, names)
        self.param_names = list(names)
        self.params_names = list(names)      # statsmodels spells it both ways
        self.nobs = nobs
        self.df_resid = df_resid
        self.df_model = len(self.params) - 1
        self.fittedvalues = fittedvalues
        self.resid = resid
        self.cov_params_ = cov
        self._extra = extra or {}
        self._model = model
        # statsmodels exposes the fit statistics as attributes; the
        # native fit carries them in its payload.
        for _k in ("deviance", "null_deviance", "aic", "loglik",
                   "df_null", "dispersion", "pearson_chi2", "converged"):
            if _k in self._extra:
                setattr(self, _k, self._extra[_k])
        if "loglik" in self._extra and not hasattr(self, "llf"):
            self.llf = self._extra["loglik"]
        if tvalues is None:
            tvalues = [(b / s) if s else float("nan")
                       for b, s in zip(self.params, self.bse)]
        self.tvalues = _NamedVec(tvalues, names)
        if pvalues is None:
            pvalues = [2.0 * _glm_core._norm_sf(abs(t)) if t == t
                       else float("nan") for t in self.tvalues]
        self.pvalues = _NamedVec(pvalues, names)

    def fit(self, *a, **k):
        return self

    def cov_params(self):
        return self.cov_params_

    def conf_int(self, alpha=0.05):
        z = _glm_core._norm_ppf(1.0 - alpha / 2.0)
        return _ConfInt([[b - z * s, b + z * s]
                         for b, s in zip(self.params, self.bse)],
                        self.param_names)

    def predict(self, data=None, **kw):
        """Fitted values, or predictions on new data.

        Mirrors statsmodels: called with no argument it returns the
        in-sample fit; called with a frame it rebuilds the design from
        the model's own terms, so the caller never has to know the
        column order or whether an intercept was added.
        """
        if data is None:
            return _as_series(self.fittedvalues)
        if self._model is None:
            raise ValueError("this result carries no model to predict from")
        return _as_series(self._model.predict_from(data))

    def wald_test(self, r_matrix, scalar=True, **kw):
        """Wald test of R beta = 0.

        W = (R b)' (R V R')^-1 (R b), chi-square on rank(R) degrees of
        freedom. Needs the parameter covariance, so it is available on
        any fit that reported one.
        """
        del kw
        if self.cov_params_ is None:
            raise ValueError("this fit carries no parameter covariance")
        R = [[float(v) for v in row] for row in r_matrix]
        b = [float(v) for v in self.params]
        V = self.cov_params_
        q = len(R)
        k = len(b)
        for row in R:
            if len(row) != k:
                raise ValueError("contrast has %d columns, the fit has "
                                 "%d parameters" % (len(row), k))
        Rb = [sum(R[i][j] * b[j] for j in range(k)) for i in range(q)]
        RV = [[sum(R[i][a] * V[a][j] for a in range(k)) for j in range(k)]
              for i in range(q)]
        M = [[sum(RV[i][a] * R[j][a] for a in range(k)) for j in range(q)]
             for i in range(q)]
        Minv = _glm_core._inv(M)
        stat = sum(Rb[i] * sum(Minv[i][j] * Rb[j] for j in range(q))
                   for i in range(q))
        pval = _glm_core._chi2_sf(stat, q)
        return _WaldResult(float(stat) if scalar else [stat],
                           float(pval) if scalar else [pval], q)

    def wald_test_terms(self, skip_single=False, scalar=True, **kw):
        """Per-term Wald tests, as statsmodels' wald_test_terms.

        One row per model term (the intercept excluded), each testing
        that term's coefficients jointly against zero.
        """
        del kw
        from . import _frame_core
        names = list(self.param_names)
        rows, index = [], []
        for j, nm in enumerate(names):
            if nm == "Intercept":
                continue
            if skip_single and len(names) <= 2:
                continue
            contrast = [[1.0 if i == j else 0.0 for i in range(len(names))]]
            w = self.wald_test(contrast, scalar=scalar)
            index.append(nm)
            rows.append({"statistic": w.statistic,
                         "pvalue": w.pvalue,
                         "df_constraint": w.df_constraint})
        table = _frame_core.DataFrame(
            {c: [r[c] for r in rows]
             for c in ("statistic", "pvalue", "df_constraint")},
            index=index)
        return _WaldTerms(table)

    def summary(self):
        head = "%-24s %12s %12s %10s" % ("term", "estimate", "std.error", "p")
        rows = ["%-24s %12.6g %12.6g %10.4g" % (n, b, s, p)
                for n, b, s, p in zip(self.param_names, self.params,
                                      self.bse, self.pvalues)]
        return "\n".join([head] + rows)

    def __getitem__(self, k):
        return self._extra[k]


def _hc_cov(X, resid, XtX_inv, cov_type):
    """Heteroskedasticity-consistent covariance (MacKinnon & White 1985).

    HC0 uses e_i^2; HC1 scales by n/(n-k); HC2 divides by (1 - h_ii);
    HC3 divides by (1 - h_ii)^2, which is the jackknife approximation
    and the default statsmodels users ask for by name.
    """
    n = len(X)
    k = len(X[0])
    # leverage h_ii = x_i' (X'X)^-1 x_i
    h = []
    for i in range(n):
        xi = X[i]
        acc = 0.0
        for a in range(k):
            s = 0.0
            for b in range(k):
                s += XtX_inv[a][b] * xi[b]
            acc += xi[a] * s
        h.append(acc)
    w = []
    for i in range(n):
        e2 = resid[i] * resid[i]
        if cov_type == "HC0":
            w.append(e2)
        elif cov_type == "HC1":
            w.append(e2 * n / float(n - k))
        elif cov_type == "HC2":
            d = 1.0 - h[i]
            w.append(e2 / d if d > 1e-12 else e2)
        else:                                    # HC3
            d = 1.0 - h[i]
            w.append(e2 / (d * d) if d > 1e-12 else e2)
    # meat = X' diag(w) X
    meat = [[0.0] * k for _ in range(k)]
    for i in range(n):
        xi = X[i]
        wi = w[i]
        for a in range(k):
            xa = xi[a] * wi
            for b in range(k):
                meat[a][b] += xa * xi[b]
    # bread * meat * bread
    tmp = [[sum(XtX_inv[a][c] * meat[c][b] for c in range(k))
            for b in range(k)] for a in range(k)]
    return [[sum(tmp[a][c] * XtX_inv[c][b] for c in range(k))
             for b in range(k)] for a in range(k)]


class _LinearModel(object):
    def __init__(self, formula, data, weights=None):
        self.formula = formula
        self.y, self.X, self.names, self.intercept = _design(formula, data)
        self.terms = _expand(formula_terms(formula)[1])
        if self.intercept:
            self.X = [[1.0] + r for r in self.X]
        self.weights = None if weights is None else \
            [float(v) for v in weights]

    def fit(self, cov_type="nonrobust", **kw):
        y, X = self.y, self.X
        n = len(X)
        k = len(X[0])
        W = self.weights
        XtX, Xty = _glm_core._xtx_xty(X, y, W)
        beta = _glm_core._solve(XtX, Xty)
        fitted = [sum(X[i][j] * beta[j] for j in range(k))
                  for i in range(n)]
        resid = [y[i] - fitted[i] for i in range(n)]
        dfr = n - k
        if dfr <= 0:
            raise ValueError("%d observations cannot support %d parameters"
                             % (n, k))
        XtX_inv = _glm_core._inv(XtX)
        ct = (cov_type or "nonrobust").upper()
        if ct.startswith("HC"):
            # The weighted fit's residuals enter the sandwich in the
            # transformed metric, so scale them the same way the design
            # was scaled.
            if W is None:
                Xe, ee = X, resid
            else:
                rw = [math.sqrt(w) for w in W]
                Xe = [[v * rw[i] for v in X[i]] for i in range(n)]
                ee = [resid[i] * rw[i] for i in range(n)]
            XtX_e = _glm_core._xtx_xty(Xe, [0.0] * n)[0]
            cov = _hc_cov(Xe, ee, _glm_core._inv(XtX_e), ct)
        else:
            if W is None:
                s2 = sum(r * r for r in resid) / dfr
            else:
                s2 = sum(W[i] * resid[i] * resid[i]
                         for i in range(n)) / dfr
            cov = [[s2 * XtX_inv[a][b] for b in range(k)] for a in range(k)]
        bse = [math.sqrt(cov[j][j]) if cov[j][j] > 0 else float("nan")
               for j in range(k)]
        tvals = [(beta[j] / bse[j]) if bse[j] == bse[j] and bse[j]
                 else float("nan") for j in range(k)]
        pvals = [2.0 * _glm_core._t_sf(abs(t), dfr) if t == t
                 else float("nan") for t in tvals]
        self._beta = beta
        return _Result(beta, bse, self.names, n, dfr,
                       tvalues=tvals, pvalues=pvals,
                       fittedvalues=fitted, resid=resid, cov=cov,
                       model=self)

    def predict_from(self, data):
        cols = [_term_column(t, data) for t in self.terms]
        rows = len(cols[0]) if cols else 0
        X = [[c[i] for c in cols] for i in range(rows)]
        if self.intercept:
            X = [[1.0] + r for r in X]
        return [sum(X[i][j] * self._beta[j]
                    for j in range(len(self._beta)))
                for i in range(rows)]


class _GLMModel(object):
    def __init__(self, formula, data, family="gaussian", weights=None):
        self.formula = formula
        self.y, self.X, self.names, self.intercept = _design(formula, data)
        self.terms = _expand(formula_terms(formula)[1])
        self.family = family
        self.weights = None if weights is None else \
            [float(v) for v in weights]

    def fit(self, *a, **kw):
        fam = self.family
        if not isinstance(fam, str):
            fam = getattr(fam, "name", None) or \
                type(fam).__name__.lower()
        fit = _glm_core.glm(self.y, self.X, family=fam,
                            add_intercept=self.intercept,
                            weights=self.weights)
        self._fit = fit
        return _Result(fit["coef"], fit["se"], self.names, len(self.y),
                       fit["df_residual"],
                       tvalues=fit.get("statistic"),
                       pvalues=fit.get("p_value", fit.get("pvalues")),
                       fittedvalues=fit.get("fitted"),
                       cov=fit.get("vcov"), extra=fit,
                       model=self)

    def predict_from(self, data):
        cols = [_term_column(t, data) for t in self.terms]
        rows = len(cols[0]) if cols else 0
        X = [[c[i] for c in cols] for i in range(rows)]
        return _glm_core.glm_predict(self._fit, X,
                                     add_intercept=self.intercept,
                                     type="response")


def ols(formula, data, **kw):
    """Ordinary least squares from a formula."""
    return _LinearModel(formula, data)


def wls(formula, data, weights=None, **kw):
    """Weighted least squares from a formula."""
    return _LinearModel(formula, data, weights=weights)


def glm(formula, data, family="gaussian", weights=None, **kw):
    """Generalised linear model from a formula."""
    return _GLMModel(formula, data, family=family, weights=weights)


def gee(*a, **k):
    raise NotImplementedError(
        "generalised estimating equations are not implemented natively "
        "yet; morie.fn._glm_formula covers ols, wls and glm")
