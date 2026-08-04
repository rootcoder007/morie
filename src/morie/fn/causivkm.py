# morie.fn -- slice k04 (rootcoder007/morie)
"""Moreira conditional likelihood-ratio (CLR) weak-IV-robust test.

Source FETCHED: the reference implementation ``CLR`` in the CRAN package
``ivmodel`` (file ``R/CLR.r``), which implements Moreira, M. J. (2003),
"A conditional likelihood ratio test for structural models",
*Econometrica* 71, 1027-1048, with the conditional p-value integral of
Andrews, Moreira and Stock (2007), *Journal of Econometrics* 138,
46-81.  Kleibergen, F. (2002), *Econometrica* 70, 1781-1803, supplies
the score decomposition the statistic is built from.  The package
source states, with Yadj/Dadj/Zadj the variables after the exogenous
covariates have been partialled out and P_Z the projection on Zadj::

    sigmaHat = (M_Z [Y D])' (M_Z [Y D]) / (n - k - L)
    a0 = (beta0, 1)          b0 = (1, -beta0)
    denomS = b0' sigmaHat b0        denomT = a0' sigmaHat^-1 a0
    QS  = || P_Z [Y D] b0 ||^2 / denomS
    QT  = || P_Z [Y D] sigmaHat^-1 a0 ||^2 / denomT
    QTS = <P_Z [Y D] b0, P_Z [Y D] sigmaHat^-1 a0>
          / (sqrt(denomS) sqrt(denomT))
    LR  = ( QS - QT + sqrt((QS + QT)^2 - 4 (QS QT - QTS^2)) ) / 2

and the conditional p-value, for L = 1,

    p = 1 - F_{1, n-k-L}(LR),

and for L >= 2, with K = Gamma(L/2) / (sqrt(pi) Gamma((L-1)/2)),

    p = 1 - 2 K int_0^1 Fchi2_L( (QT + LR) / (1 + QT x^2 / LR) )
                         (1 - x^2)^((L-3)/2) dx.

The substitution x = sin(theta) turns that into

    p = 1 - 2 K int_0^{pi/2} Fchi2_L( (QT + LR)
                / (1 + QT sin^2(theta) / LR) ) cos^(L-2)(theta) dtheta,

which is what is evaluated here.  The substituted integrand is bounded
and smooth for every L >= 2, so the epsilon-regularised special case
that ``ivmodel`` needs at L = 4 (where ``(1 - x^2)^(1/2)`` has an
infinite derivative at the endpoint) is not needed.  For L = 2 the
substituted form reduces to exactly the ``sin(x)^2`` integral in the
package source.  A fixed 4096-interval composite Simpson rule is used,
so the result is deterministic and has no tolerance-driven exit.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

import math

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["causal_iv_kleibergen_moreira"]

_SIMPSON_N = 4096


def _partial_out(M, C):
    # residuals of each column of M after regression on C
    if C.shape[1] == 0:
        return M
    return M - C @ _mlstsq(C, M)


def _mlstsq(A, B):
    # least-squares coefficients, one column of B at a time
    return np.column_stack([np.linalg.lstsq(A, B[:, j], rcond=None)[0] for j in range(B.shape[1])])


def _cond_pvalue(m, qT, L, df2):
    """Conditional p-value P(LR > m | Q_T = qT), Andrews-Moreira-Stock (2007)."""
    if L == 1:
        return float(stats.f.sf(m, 1, df2))
    if m <= 0.0:
        return 1.0
    logK = math.lgamma(L / 2.0) - 0.5 * math.log(math.pi) - math.lgamma((L - 1) / 2.0)
    K = math.exp(logK)
    a, b = 0.0, math.pi / 2.0
    h = (b - a) / _SIMPSON_N
    total = 0.0
    for i in range(_SIMPSON_N + 1):
        th = a + i * h
        s = math.sin(th)
        arg = (qT + m) / (1.0 + qT * s * s / m)
        val = float(stats.chi2.cdf(arg, L)) * math.cos(th) ** (L - 2)
        if i == 0 or i == _SIMPSON_N:
            w = 1.0
        elif i % 2 == 1:
            w = 4.0
        else:
            w = 2.0
        total += w * val
    integral = total * h / 3.0
    return float(min(1.0, max(0.0, 1.0 - 2.0 * K * integral)))


def causal_iv_kleibergen_moreira(y, X, Z, beta0=0.0, X_exog=None, add_intercept=True):
    """Moreira conditional LR test that the structural coefficient is ``beta0``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    X : array-like, shape (n,)
        The single endogenous regressor.
    Z : array-like, shape (n, L)
        Instruments.
    beta0 : float, default 0.0
        Null value of the structural coefficient.
    X_exog : array-like, shape (n, q), optional
        Included exogenous covariates, partialled out of y, X and Z.
    add_intercept : bool, default True

    Returns
    -------
    RichResult
        keys: ``statistic`` (the CLR statistic), ``p_value``, ``QS``,
        ``QT``, ``QTS``, ``beta0``, ``n``, ``n_instruments``, ``df2``,
        ``method``.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = int(y.size)
    d = np.asarray(X, dtype=float).ravel()
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.shape[0] != n:
        Z = Z.T
    cols = []
    if add_intercept:
        cols.append(np.ones(n))
    if X_exog is not None:
        Xe = np.atleast_2d(np.asarray(X_exog, dtype=float))
        if Xe.shape[0] != n:
            Xe = Xe.T
        cols.extend(Xe[:, j] for j in range(Xe.shape[1]))
    C = np.column_stack(cols) if cols else np.zeros((n, 0))
    k = int(C.shape[1])
    L = int(Z.shape[1])
    df2 = n - k - L
    if L < 1 or df2 < 1:
        raise ValueError("need L >= 1 and n > k + L")

    YD = _partial_out(np.column_stack([y, d]), C)
    Za = _partial_out(Z, C)
    PZ = Za @ _mlstsq(Za, YD)
    RZ = YD - PZ

    sigma = (RZ.T @ RZ) / df2
    sigma_inv = np.linalg.inv(sigma)
    b0 = np.array([1.0, -float(beta0)])
    a0 = np.array([float(beta0), 1.0])
    denomS = float(b0 @ sigma @ b0)
    denomT = float(a0 @ sigma_inv @ a0)
    u = PZ @ b0
    v = PZ @ (sigma_inv @ a0)
    QS = float(u @ u) / denomS
    QT = float(v @ v) / denomT
    QTS = float(u @ v) / (math.sqrt(denomS) * math.sqrt(denomT))

    disc = (QS + QT) ** 2 - 4.0 * (QS * QT - QTS * QTS)
    lr = 0.5 * (QS - QT + math.sqrt(max(0.0, disc)))
    return RichResult(
        payload={
            "statistic": float(lr),
            "p_value": _cond_pvalue(lr, QT, L, df2),
            "QS": QS,
            "QT": QT,
            "QTS": QTS,
            "beta0": float(beta0),
            "n": n,
            "n_instruments": L,
            "df2": int(df2),
            "method": "Moreira conditional LR weak-IV-robust test (Kleibergen 2002, Moreira 2003)",
        }
    )


def cheatsheet():
    return "causivkm: conditional LR weak-instrument-robust test"
