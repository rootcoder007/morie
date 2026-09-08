# morie.fn -- slice k04 (rootcoder007/morie)
"""Anderson-Rubin weak-instrument-robust test of H0: beta = beta0.

Source FETCHED: the reference implementation ``AR.test`` in the CRAN
package ``ivmodel`` (file ``R/AR.r``), which implements Anderson, T. W.
and Rubin, H. (1949), "Estimation of the parameters of a single
equation in a complete system of stochastic equations", *Annals of
Mathematical Statistics* 20, 46-63.  The 1949 paper itself was not
reachable here; the package source is by the method authors of the
modern treatment and states the statistic explicitly::

    temp  = Y - beta0 * D                       (residual under H0)
    Fstat = ||P_Z temp||^2 / ||M_Z temp||^2 * (n - k - L) / L

with P_Z the projection on the instruments (after any exogenous
covariates have been partialled out of Y, D and Z), L the number of
instruments, k the number of exogenous covariates, and

    p = 1 - F_{L, n-k-L}(Fstat).

The test is exact under Gaussian errors and stays valid however weak
the instruments are, which is the point of using it.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["causal_iv_anderson_rubin"]


def _partial_out(M, C):
    # residuals of each column of M after regression on C
    if C.shape[1] == 0:
        return M
    return M - C @ _mlstsq(C, M)


def _mlstsq(A, B):
    # least-squares coefficients, one column of B at a time
    return np.column_stack([np.linalg.lstsq(A, B[:, j], rcond=None)[0] for j in range(B.shape[1])])


def causal_iv_anderson_rubin(y, X, Z, beta0=0.0, X_exog=None, add_intercept=True):
    """Anderson-Rubin test that the structural coefficient equals ``beta0``.

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
        keys: ``statistic``, ``p_value``, ``df1``, ``df2``, ``beta0``,
        ``n``, ``n_instruments``, ``method``.
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

    ya = _partial_out(y.reshape(n, 1), C).ravel()
    da = _partial_out(d.reshape(n, 1), C).ravel()
    Za = _partial_out(Z, C)

    temp = ya - float(beta0) * da
    coef, *_ = np.linalg.lstsq(Za, temp, rcond=None)
    fit = Za @ coef
    ssf = float(fit @ fit)
    sse = float(temp @ temp) - ssf
    if sse <= 0.0:
        raise ValueError("degenerate fit: residual sum of squares is not positive")
    stat = ssf / sse * df2 / L
    return RichResult(
        payload={
            "statistic": float(stat),
            "p_value": float(stats.f.sf(stat, L, df2)),
            "df1": L,
            "df2": int(df2),
            "beta0": float(beta0),
            "n": n,
            "n_instruments": L,
            "method": "Anderson-Rubin weak-IV-robust test (Anderson and Rubin 1949)",
        }
    )


def cheatsheet():
    return "causivar: Anderson-Rubin weak-instrument-robust test"
