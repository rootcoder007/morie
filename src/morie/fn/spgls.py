"""GLS for spatial data with known Sigma."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["schabenberger_gls_spatial"]


def schabenberger_gls_spatial(x, y, sigma=None):
    r"""
    Generalised least squares with a known covariance matrix.

    When the errors are spatially correlated with known
    :math:`\mathrm{Var}[Z(s)] = \Sigma`, the efficient estimator is

    .. math::

        \hat\beta_{gls} = (X'\Sigma^{-1}X)^{-1}X'\Sigma^{-1}Z,
        \qquad
        \mathrm{Var}[\hat\beta_{gls}] = (X'\Sigma^{-1}X)^{-1}

    Two properties worth stating, because both are easy to get wrong:

    * With :math:`\Sigma = \sigma^2 I` the estimator reduces EXACTLY to
      ordinary least squares. Any implementation that fails this has its
      whitening the wrong way round.
    * OLS remains unbiased under correlated errors but its usual standard
      errors do not: they are computed as if :math:`\Sigma = \sigma^2 I`.
      The OLS point estimate and its correct variance
      :math:`(X'X)^{-1}X'\Sigma X(X'X)^{-1}` are returned alongside, so
      the size of that error is visible.

    Parameters
    ----------
    x : array-like
        Design matrix, shape ``(n, p)``.
    y : array-like
        Response, shape ``(n,)``.
    sigma : array-like, optional
        Known ``(n, n)`` error covariance. Identity when omitted, which
        makes this OLS.

    Returns
    -------
    RichResult
        ``beta``, ``vcov``, ``se``, ``residuals``, ``beta_ols``,
        ``se_ols_naive``, ``se_ols_correct``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Sec. 6.2.3 "Generalized
    Least Squares -- Inference and Diagnostics", p. 341.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.ndim == 1:
        X = X[:, None]
    z = np.asarray(y, dtype=float).ravel()
    n = z.size
    if X.shape[0] != n:
        raise ValueError("`x` and `y` must have the same number of rows")
    S = np.eye(n) if sigma is None else np.asarray(sigma, dtype=float)
    if S.shape != (n, n):
        raise ValueError(f"`sigma` must be ({n}, {n}) to match the data")

    Sinv_X = np.linalg.solve(S, X)
    XtSinvX = X.T @ Sinv_X
    vcov = np.linalg.inv(XtSinvX)
    beta = vcov @ (Sinv_X.T @ z)

    XtX_inv = np.linalg.inv(X.T @ X)
    beta_ols = XtX_inv @ (X.T @ z)
    resid_ols = z - X @ beta_ols
    s2 = float(resid_ols @ resid_ols) / max(n - X.shape[1], 1)
    se_naive = np.sqrt(np.diag(s2 * XtX_inv))
    vcov_ols = XtX_inv @ (X.T @ S @ X) @ XtX_inv

    return RichResult(
        title="Spatial GLS with known Sigma",
        summary_lines=[("n", n), ("p", int(X.shape[1]))],
        payload={"beta": beta, "vcov": vcov, "se": np.sqrt(np.diag(vcov)),
                 "residuals": z - X @ beta, "beta_ols": beta_ols,
                 "se_ols_naive": se_naive,
                 "se_ols_correct": np.sqrt(np.diag(vcov_ols))},
    )


def cheatsheet():
    return "spgls: GLS with known Sigma; equals OLS exactly when Sigma = sigma^2 I."
