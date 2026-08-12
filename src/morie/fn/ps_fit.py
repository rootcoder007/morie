# morie.fn -- function file (rootcoder007/morie)
"""
Propensity score estimation via logistic regression.

Implements ``compute_propensity_scores`` -- the foundational propensity score
model used by IPW, AIPW, ATT, ATC, and other causal estimators in morie.
"""

from __future__ import annotations

import math as _math

from . import _frame_core as pd

class _MissingDep:
    """Placeholder for a dependency being nativized (task #141)."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise ImportError(
            "%s is no longer bundled; this code path awaits its native "
            "morie implementation" % self._name)

    def __call__(self, *a, **k):
        raise ImportError(
            "%s is no longer bundled; this code path awaits its native "
            "morie implementation" % self._name)

try:
    from ._ml_core import LogisticRegression
except ImportError:
    LogisticRegression = _MissingDep('LogisticRegression')
try:
    from ._ml_core import LabelEncoder, StandardScaler
except ImportError:
    LabelEncoder = _MissingDep('LabelEncoder')
    StandardScaler = _MissingDep('StandardScaler')


_PS_MODELS = ("mle", "ridge")


def _ps_design(data, covariates):
    """Numeric design matrix: non-numeric columns label-encoded."""
    X = []
    cols = []
    for col in covariates:
        v = list(data[col])
        try:
            cols.append([float(u) for u in v])
        except (TypeError, ValueError):
            levels = sorted({str(u) for u in v})
            idx = {lv: i for i, lv in enumerate(levels)}
            cols.append([float(idx[str(u)]) for u in v])
    n = len(cols[0]) if cols else 0
    for i in range(n):
        X.append([1.0] + [c[i] for c in cols])
    return X


def _ps_standardize(X):
    """Standardise every column but the intercept, population sd."""
    n = len(X)
    p = len(X[0])
    out = [row[:] for row in X]
    for j in range(1, p):
        col = [X[i][j] for i in range(n)]
        m = sum(col) / n
        v = sum((u - m) ** 2 for u in col) / n
        s = v ** 0.5
        if s <= 0.0:
            s = 1.0
        for i in range(n):
            out[i][j] = (X[i][j] - m) / s
    return out


def _ps_irls(X, y, lam=0.0, max_iter=200, tol=1e-12):
    """Logistic IRLS; `lam` is an L2 penalty on the NON-intercept
    coefficients.  lam = 0 is the unpenalised MLE.

    Mirrored exactly by .mor_ps_irls in the R arm, so both propensity
    routes agree across languages rather than each inheriting whatever
    its own ecosystem's logistic regression happens to default to.
    """
    n = len(X)
    p = len(X[0])
    beta = [0.0] * p
    pen = [0.0] + [float(lam)] * (p - 1)
    for _ in range(int(max_iter)):
        eta = [sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
        eta = [max(-30.0, min(30.0, e)) for e in eta]
        mu = [1.0 / (1.0 + _math.exp(-e)) for e in eta]
        w = [max(m * (1.0 - m), 1e-10) for m in mu]
        z = [eta[i] + (y[i] - mu[i]) / w[i] for i in range(n)]
        A = [[sum(w[i] * X[i][a] * X[i][b] for i in range(n))
              + (pen[a] if a == b else 0.0) for b in range(p)]
             for a in range(p)]
        rhs = [sum(w[i] * X[i][a] * z[i] for i in range(n)) for a in range(p)]
        new = _ps_solve(A, rhs)
        delta = max(abs(new[j] - beta[j]) for j in range(p))
        beta = new
        if delta < tol:
            break
    eta = [sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
    eta = [max(-30.0, min(30.0, e)) for e in eta]
    return [1.0 / (1.0 + _math.exp(-e)) for e in eta]


def _ps_solve(A, b):
    """Gaussian elimination with partial pivoting."""
    p = len(b)
    M = [A[i][:] + [b[i]] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            continue
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        for k in range(c, p + 1):
            M[c][k] /= d
        for r in range(p):
            if r == c:
                continue
            f = M[r][c]
            if f == 0.0:
                continue
            for k in range(c, p + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][p] for i in range(p)]


def compute_propensity_scores(data: pd.DataFrame, treatment: str,
                              covariates: list,
                              ps_model: str = "mle",
                              ridge_lambda: float = 1.0) -> pd.Series:
    """Propensity scores by logistic regression.

    Two estimators, both available and both matched exactly by the R
    arm (see .mor_ps_irls in causal.R):

    ``ps_model="mle"``
        Unpenalised logistic maximum likelihood on the raw covariates.
        The textbook propensity model and the default.
    ``ps_model="ridge"``
        L2-penalised logistic on standardised covariates, the penalty
        applying to the non-intercept coefficients with strength
        ``ridge_lambda``.  Useful when covariates are collinear or a
        covariate separates the treatment.

    Before 2026-08-12 this function silently used the ridge route while
    the R arm used the MLE route, which is why their AIPW estimates
    disagreed at ~2e-4.
    """
    if ps_model not in _PS_MODELS:
        raise ValueError("ps_model must be 'mle' or 'ridge'")
    frame = data.loc[:, [treatment, *covariates]].dropna()
    y = [float(v) for v in frame[treatment]]
    X = _ps_design(frame, covariates)
    if ps_model == "ridge":
        X = _ps_standardize(X)
        ps = _ps_irls(X, y, lam=float(ridge_lambda))
    else:
        ps = _ps_irls(X, y, lam=0.0)
    return pd.Series(ps, index=frame.index)

def cheatsheet() -> str:
    return "compute_propensity_scores({}) -> Propensity score estimation via logistic regression."
