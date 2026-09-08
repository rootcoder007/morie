# morie.fn -- function file (rootcoder007/morie)
"""Cinelli-Hazlett omitted-variable-bias sensitivity analysis."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["cinelli_hazlett"]


def ols_with_se(y, X, j):
    """OLS coefficient j, its standard error and the residual df."""
    y = core.vec(y)
    D = core.mat(X)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: y has no observations")
    if len(D) != n:
        raise ValueError("y and X must have the same number of rows")
    A = [[1.0] + list(r) for r in D]
    p = len(A[0])
    if n <= p:
        raise ValueError("need more observations than columns")
    XtX = core.crossprod(A)
    beta = core.cholsolve(XtX, core.matvec(core.tr(A), y))
    fit = core.matvec(A, beta)
    e = [y[i] - fit[i] for i in range(n)]
    df = n - p
    s2 = sum(v * v for v in e) / df
    col = core.cholsolve(XtX, [1.0 if k == j else 0.0 for k in range(p)])
    se = math.sqrt(s2 * col[j])
    return beta[j], se, df


def robustness_value(t, df, q=1.0, alpha=1.0):
    """RV_{q,alpha}: the partial R2 an omitted confounder needs.

    Cinelli & Hazlett (2020) eq. (9).  A confounder explaining RV of both
    treatment and outcome residual variation moves the estimate by q
    times its size; at alpha = 1 the significance version reduces to it.
    """
    fq = q * abs(t) / math.sqrt(df)
    fcrit = abs(t) / math.sqrt(df)
    if alpha < 1.0:
        fq = q * (abs(t) / math.sqrt(df))
    rv = 0.5 * (math.sqrt(fq ** 4 + 4.0 * fq ** 2) - fq ** 2)
    return min(max(rv, 0.0), 1.0), fcrit


def cinelli_hazlett(model, treat=None, cov=None, R2_yu=0.0, R2_du=0.0, q=1.0):
    """
    Cinelli-Hazlett sensitivity to an unobserved confounder

    Formula: adjusted estimate vs (R2_y~u.x, R2_d~u.x)

    The bias an omitted confounder U can produce is bounded by
    |bias| = se * sqrt(df) * sqrt(R2_yu R2_du / (1 - R2_du)),
    so a claim survives U exactly when the adjusted estimate keeps its
    sign.  The robustness value RV_q is the common partial R2 at which
    the adjusted estimate is driven to q times zero: feeding RV back in
    as both R2 values therefore returns an adjusted estimate of exactly
    zero, which is the identity used to check this implementation.

    Parameters
    ----------
    model : array-like
        Outcome vector y.
    treat : array-like
        Treatment vector D.
    cov : array-like or None
        n x k matrix of observed covariates X.
    R2_yu : float
        Hypothesised partial R2 of the confounder with the outcome.
    R2_du : float
        Hypothesised partial R2 of the confounder with the treatment.
    q : float
        Fraction of the estimate the confounder would have to explain.

    Returns
    -------
    result : dict
        Keys: estimate (adjusted effect), tau, se, t, df, bias,
        adjusted_se, rv_q, r2_yd_x, robust, n.

    References
    ----------
    Cinelli & Hazlett (2020), Making Sense of Sensitivity, JRSS B
    82(1):39-67.
    """
    y = core.vec(model)
    d = core.vec(treat)
    n = len(y)
    if n == 0:
        raise ValueError("empty input: model has no observations")
    if len(d) != n:
        raise ValueError("model and treat must have the same length")
    if cov is None:
        Xm = [[] for _ in range(n)]
    else:
        Xm = core.mat(cov)
        if len(Xm) != n:
            raise ValueError("cov must have one row per observation")
    if not (0.0 <= R2_yu <= 1.0 and 0.0 <= R2_du < 1.0):
        raise ValueError("R2_yu must lie in [0, 1] and R2_du in [0, 1)")
    D = [[d[i]] + list(Xm[i]) for i in range(n)]
    tau, se, df = ols_with_se(y, D, 1)
    t = tau / se if se > 0.0 else float("nan")
    bias = se * math.sqrt(df) * math.sqrt(R2_yu * R2_du / (1.0 - R2_du))
    adj = tau - bias if tau >= 0.0 else tau + bias
    adj_se = se * math.sqrt((1.0 - R2_yu) / (1.0 - R2_du)) * \
        math.sqrt(df / (df - 1.0)) if df > 1 else float("nan")
    rv, _f = robustness_value(t, df, q)
    r2_yd = t * t / (t * t + df)
    return RichResult(payload={
        "estimate": adj,
        "tau": tau,
        "se": se,
        "t": t,
        "df": df,
        "bias": bias,
        "adjusted_se": adj_se,
        "rv_q": rv,
        "r2_yd_x": r2_yd,
        "robust": 1 if (adj * tau > 0.0) else 0,
        "n": n,
        "method": "Cinelli-Hazlett omitted-variable-bias sensitivity",
    })


def cheatsheet():
    return "chzlt: Cinelli-Hazlett sensitivity to an unobserved confounder"


# compact alias per ledger/NAMING.md
cinellihazlett = cinelli_hazlett
