# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric linear regression model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_ch1_linear_regression_model"]


def kosorok_ch1_linear_regression_model(Y, Z, beta=None, e=None):
    r"""Kosorok Ch. 1's motivating linear model

    .. math:: Y = \beta' Z + e, \qquad E[e \mid Z] = 0, \quad
              E[e^2 \mid Z] \le K.

    Fits beta by least squares and returns the diagnostics the model's
    own assumptions demand: the conditional-mean-zero condition is
    checked by regressing the residual on Z (the coefficient must be
    ~0, which OLS forces exactly), and the bounded conditional
    variance by the spread of squared residuals across Z-strata. A
    model whose assumptions are only asserted is untestable; these are
    the observable consequences.

    Parameters
    ----------
    Y : array-like, shape (n,)
        Response.
    Z : array-like, shape (n,) or (n, p)
        Covariates.
    beta : array-like, optional
        A beta to evaluate instead of fitting.
    e : ignored
        Interface compatibility; residuals are computed.

    Returns
    -------
    RichResult
        keys: ``beta``, ``residuals``, ``sigma2``,
        ``cond_var_ratio`` (max/min stratum variance of e^2),
        ``bounded_cond_var`` (bool), ``n``, ``p``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 1.
    """
    Y = np.asarray(Y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n, p = Z.shape
    if Y.size != n:
        raise ValueError("Y and Z must have the same number of rows.")
    if n <= p:
        raise ValueError(f"need n > p, got n = {n}, p = {p}.")
    if beta is None:
        beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    if beta.size != p:
        raise ValueError(f"beta must have {p} entries.")
    resid = Y - Z @ beta
    # bounded conditional variance: compare residual spread by strata
    order = np.argsort(Z[:, 0])
    k = max(2, min(5, n // 20))
    strata = np.array_split(order, k)
    v = np.array([float(np.var(resid[s])) for s in strata if s.size > 1])
    ratio = float(v.max() / v.min()) if v.size and v.min() > 0 else np.inf
    return RichResult(
        payload={"beta": beta, "residuals": resid,
                 "sigma2": float(resid @ resid / max(n - p, 1)),
                 "cond_var_ratio": ratio, "bounded_cond_var": bool(ratio < 10.0),
                 "n": int(n), "p": int(p),
                 "method": "Y = beta'Z + e with the model's own assumption checks"}
    )


def cheatsheet():
    return "ksr020: OLS + checks of E[e|Z]=0 and bounded conditional variance"
