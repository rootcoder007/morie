# morie.fn -- function file (rootcoder007/morie)
"""VAR(p) vector autoregression by multivariate least squares.

SOURCE.  Lutkepohl, H. (2005), *New Introduction to Multiple Time
Series Analysis*, Springer; doi:10.1007/978-3-540-27752-1.  The model is
Lutkepohl's Eq. (2.1.1)/(3.2.1),

    y_t = nu + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,   u_t ~ (0, Sigma_u)

and the estimator is the multivariate least squares estimator of his
Section 3.2: stack the regressors of every equation into the same
Z_t = (1, y_{t-1}', ..., y_{t-p}')' and the LS estimator separates,
equation by equation, into B = (Z'Z)^{-1} Z'Y.  This is why a VAR needs
no SUR/GLS step: the regressor set is identical across equations.

Two residual covariances are reported, as Lutkepohl distinguishes them:
the ML/plug-in Sigma-tilde = U'U / T and the unbiased-in-the-univariate
sense Sigma-hat = U'U / (T - Kp - 1).  The Gaussian log-likelihood
(Lutkepohl Eq. 3.4.5) is evaluated at Sigma-tilde,

    log L = -(TK/2) log(2 pi) - (T/2) log|Sigma-tilde| - TK/2,

and the three standard order-selection criteria are the usual
log|Sigma-tilde| + c(T) * (free parameters) / T.

NOT written from the book's own page images -- Lutkepohl (2005) is not
in the local corpus.  The estimator above is the standard multivariate
LS form; it is anchored here against R's own ``lm()`` reached by a
different route (equation-by-equation regression on the lag design),
and against the closed-form univariate OLS AR(1) slope.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["vector_autoregression"]


def _design(A, p, intercept):
    T = len(A)
    K = len(A[0])
    Z = []
    Y = []
    for t in range(p, T):
        row = [1.0] if intercept else []
        for i in range(1, p + 1):
            row.extend(A[t - i])
        Z.append(row)
        Y.append(list(A[t]))
    return Z, Y, K


def vector_autoregression(Y, p=1, intercept=True):
    """Estimate a VAR(p) by multivariate least squares.

    Parameters
    ----------
    Y : array-like, shape (T, K)
        Observations, one row per time point.  A flat sequence is read
        as a single series (K = 1).
    p : int
        Lag order, >= 1.
    intercept : bool
        Include the constant nu.

    Returns
    -------
    RichResult
        ``coef`` (K by 1+Kp, row = equation, columns = [nu, A_1 row,
        ..., A_p row] -- the layout :func:`morie.fn.irfun.impulse_response`
        expects), ``sigma_u`` (T-Kp-1 divisor), ``sigma_ml`` (T divisor),
        ``resid``, ``loglik``, ``aic``, ``bic``, ``hq``, ``n``, ``k``,
        ``p``.

    Raises
    ------
    ValueError
        Empty input, p < 1, or fewer effective observations than
        regressors.

    References
    ----------
    Lutkepohl, H. (2005).  New Introduction to Multiple Time Series
    Analysis.  Springer.  doi:10.1007/978-3-540-27752-1.
    """
    A = core.mat(Y)
    if not A:
        raise ValueError("vector_autoregression: Y is empty")
    p = int(p)
    if p < 1:
        raise ValueError("vector_autoregression: p must be at least 1")
    K = len(A[0])
    for r in A:
        if len(r) != K:
            raise ValueError("vector_autoregression: rows of Y have unequal length")
    T = len(A)
    n = T - p
    if n <= 0:
        raise ValueError("vector_autoregression: not enough observations for p lags")
    Z, Ye, K = _design(A, p, bool(intercept))
    m = len(Z[0])
    if n <= m:
        raise ValueError("vector_autoregression: fewer observations than regressors")
    ZtZ = core.crossprod(Z)
    Zt = core.tr(Z)
    coef = []
    for j in range(K):
        col = [Ye[t][j] for t in range(n)]
        coef.append(core.cholsolve(ZtZ, core.matvec(Zt, col)))
    resid = []
    for t in range(n):
        row = []
        for j in range(K):
            s = 0.0
            for q in range(m):
                s += Z[t][q] * coef[j][q]
            row.append(Ye[t][j] - s)
        resid.append(row)
    S = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            s = 0.0
            for t in range(n):
                s += resid[t][i] * resid[t][j]
            S[i][j] = s
    sigma_ml = [[S[i][j] / n for j in range(K)] for i in range(K)]
    dfd = n - m
    sigma_u = [[S[i][j] / dfd for j in range(K)] for i in range(K)]
    L = core.chol(sigma_ml)
    logdet = 0.0
    for i in range(K):
        logdet += 2.0 * math.log(L[i][i])
    loglik = -0.5 * n * K * math.log(2.0 * math.pi) - 0.5 * n * logdet - 0.5 * n * K
    npar = K * m
    aic = logdet + 2.0 * npar / n
    bic = logdet + math.log(n) * npar / n
    hq = logdet + 2.0 * math.log(math.log(n)) * npar / n
    return RichResult(
        title="VAR(p) by multivariate least squares",
        summary_lines=[("series", K), ("lags", p), ("used obs", n)],
        payload={
            "estimate": loglik,
            "coef": coef,
            "sigma_u": sigma_u,
            "sigma_ml": sigma_ml,
            "resid": resid,
            "loglik": loglik,
            "aic": aic,
            "bic": bic,
            "hq": hq,
            "logdet": logdet,
            "n": n,
            "k": K,
            "p": p,
            "method": "VAR(p) multivariate LS, Lutkepohl (2005) Sec. 3.2",
        },
    )


def cheatsheet():
    return "varest: VAR(p) by multivariate least squares (Lutkepohl 2005 Sec. 3.2)"


varestimate = vector_autoregression
