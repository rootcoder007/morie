# morie.fn -- function file (rootcoder007/morie)
"""Semiparametric least squares (Ichimura)."""

import numpy as np

from ._horowitz import kernel, silverman_bw
from ._richresult import RichResult

__all__ = ["hrz_semiparametric_ls", "horowitz_nls_sim"]


from scipy import optimize


def hrz_semiparametric_ls(X, y, h=None, kernel_name="gaussian", beta0=None):
    r"""Ichimura's semiparametric least squares (Horowitz Ch. 2):

    .. math:: \hat\beta = \arg\min_{b:\,|b_1|=1} \sum_i
              \big(Y_i - \hat G_{-i,b}(X_i'b)\big)^2,

    with G estimated by LEAVE-ONE-OUT kernel regression. Both details
    matter: the normalisation :math:`|b_1| = 1` is required because
    the index scale is not identified (G absorbs any rescaling), and
    leaving observation i out of its own G prevents the criterion from
    being driven to zero by interpolation.

    Despite G converging at a nonparametric rate, beta is root-n
    consistent -- the book's headline result for index models.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Covariates.
    y : array-like, shape (n,)
        Response.
    h : float, optional
        Bandwidth.
    kernel_name : str
        Kernel.
    beta0 : array-like, optional
        Starting value.

    Returns
    -------
    RichResult
        keys: ``beta`` (normalised so beta[0] = 1), ``sse``,
        ``converged``, ``n``, ``d``, ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (Ichimura's estimator).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.size:
        X = X.T
    if X.shape[0] != y.size:
        raise ValueError("X must have one row per entry of y.")
    n, d = X.shape
    if d < 2:
        raise ValueError("single-index estimation needs at least 2 covariates.")

    def sse(rest):
        b = np.r_[1.0, rest]  # |b_1| = 1 normalisation
        v = X @ b
        hh = silverman_bw(v) if h is None else float(h)
        K = kernel((v[:, None] - v[None, :]) / hh, kernel_name)
        np.fill_diagonal(K, 0.0)  # leave-one-out
        den = K.sum(axis=1)
        with np.errstate(invalid="ignore"):
            G = np.where(den > 0, (K @ y) / np.maximum(den, 1e-300), np.nan)
        r = y - G
        r = r[np.isfinite(r)]
        return float(np.sum(r**2)) if r.size else 1e18

    start = np.zeros(d - 1) if beta0 is None else \
        np.atleast_1d(np.asarray(beta0, dtype=float))[1:] / \
        np.atleast_1d(np.asarray(beta0, dtype=float))[0]
    res = optimize.minimize(sse, start, method="Nelder-Mead",
                            options={"maxiter": 2000, "fatol": 1e-8})
    beta = np.r_[1.0, res.x]
    return RichResult(payload={"beta": beta, "sse": float(res.fun),
                               "converged": bool(res.success), "n": int(n),
                               "d": int(d),
                               "method": "Ichimura SLS; |b1|=1 and leave-one-out are both required"})


def cheatsheet():
    return "hrznls: |b1|=1 fixes the unidentified scale; LOO stops interpolation"


#: Catalogue alias for :func:`hrz_semiparametric_ls`.
horowitz_nls_sim = hrz_semiparametric_ls
