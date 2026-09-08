# morie.fn -- function file (rootcoder007/morie)
"""Partially linear regression."""

from . import _array_core as np

from ._horowitz import local_linear, silverman_bw
from ._richresult import RichResult

__all__ = ["hrz_partially_linear", "horowitz_robinson_plr"]


def hrz_partially_linear(X, Z, y, h=None, kernel_name="gaussian"):
    r"""Robinson's partially linear model (Horowitz Ch. 2):

    .. math:: Y = X'\beta + g(Z) + \epsilon, \qquad
              \hat\beta = \Big(\widehat{E}[\tilde X \tilde X']
              \Big)^{-1} \widehat{E}[\tilde X \tilde Y],

    with :math:`\tilde X = X - \widehat E[X|Z]` and likewise for Y.
    Partialling out the nonparametric part by kernel regression leaves
    beta **root-n** consistent even though g converges slowly -- the
    slower nuisance rate does not contaminate the parametric one,
    which is exactly Robinson's result.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Parametric regressors.
    Z : array-like, shape (n,)
        The covariate entering nonparametrically.
    y : array-like, shape (n,)
        Response.
    h : float, optional
        Bandwidth for the partialling-out regressions.
    kernel_name : str
        Kernel.

    Returns
    -------
    RichResult
        keys: ``beta``, ``se``, ``residuals``, ``g_fitted``,
        ``bandwidth``, ``root_n`` (True), ``n``, ``p``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (partially linear models; Robinson 1988).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    if X.shape[0] != y.size:
        X = X.T
    if X.shape[0] != y.size or Z.size != y.size:
        raise ValueError("X, Z and y must have the same number of rows.")
    n, p = X.shape
    h = silverman_bw(Z) if h is None else float(h)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")

    _, my, _, _ = local_linear(Z, y, grid=Z, h=h, name=kernel_name)
    Xt = np.empty_like(X)
    for j in range(p):
        _, mj, _, _ = local_linear(Z, X[:, j], grid=Z, h=h, name=kernel_name)
        Xt[:, j] = X[:, j] - mj
    yt = y - my
    ok = np.isfinite(yt) & np.all(np.isfinite(Xt), axis=1)
    if ok.sum() <= p:
        raise ValueError("too few usable observations after partialling out.")
    A = Xt[ok].T @ Xt[ok]
    beta = np.linalg.solve(A, Xt[ok].T @ yt[ok])
    resid = yt[ok] - Xt[ok] @ beta
    s2 = float(resid @ resid) / max(ok.sum() - p, 1)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A)))
    g_fit = y - X @ beta
    return RichResult(payload={"beta": beta, "se": se, "residuals": resid,
                               "g_fitted": g_fit, "bandwidth": h,
                               "root_n": True, "n": int(n), "p": int(p),
                               "method": "Robinson partialling-out; beta root-n despite slow g"})


def cheatsheet():
    return "hrzplr: partialling out leaves beta root-n; the slow g does not contaminate it"


#: Catalogue alias for :func:`hrz_partially_linear`.
horowitz_robinson_plr = hrz_partially_linear
