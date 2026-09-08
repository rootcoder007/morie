# morie.fn -- function file (rootcoder007/morie)
"""Series (sieve) regression."""

from . import _array_core as np

from ._horowitz import sieve_basis
from ._richresult import RichResult

__all__ = ["hrz_series_regression", "horowitz_series_regression"]


def hrz_series_regression(x, y, K=5, kind="poly", grid=None):
    r"""Series (sieve) regression (Horowitz Ch. 2):

    .. math:: \hat m(x) = \sum_{k=1}^{K} \hat a_k p_k(x),
              \qquad \hat a = \arg\min \frac1n \sum_i
              \Big(Y_i - \sum_k a_k p_k(X_i)\Big)^2.

    K plays the role a bandwidth plays in kernel methods and must grow
    with n: fixed K is a parametric model that never becomes
    consistent, while K growing too fast overfits. The effective
    degrees of freedom are returned so the trade-off is visible.

    Parameters
    ----------
    x, y : array-like
        Regressor and response.
    K : int, default 5
        Sieve dimension.
    kind : {"poly", "fourier"}
        Basis.
    grid : array-like, optional
        Evaluation points.

    Returns
    -------
    RichResult
        keys: ``grid``, ``fitted``, ``coefficients``, ``K``,
        ``r_squared``, ``df_ratio`` (K/n), ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Ch. 2 (series estimators).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    K = int(K)
    if K < 1 or K > x.size:
        raise ValueError(f"K must lie in 1..{x.size}, got {K}.")
    P = sieve_basis(x, K=K, kind=kind)
    a, *_ = np.linalg.lstsq(P, y, rcond=None)
    fit_in = P @ a
    ss_res = float(np.sum((y - fit_in) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    g = x if grid is None else np.atleast_1d(np.asarray(grid, dtype=float))
    # rebuild the basis on the pooled range so the grid maps consistently
    Pg = sieve_basis(np.r_[x, g], K=K, kind=kind)[x.size:]
    return RichResult(payload={"grid": g, "fitted": Pg @ a, "coefficients": a,
                               "K": K, "r_squared": 1 - ss_res / ss_tot
                               if ss_tot > 0 else np.nan,
                               "df_ratio": K / x.size,
                               "method": "Series regression; K must grow with n, like 1/h"})


def cheatsheet():
    return "hrzsier: fixed K never becomes consistent -- K is the bandwidth here"


#: Catalogue alias for :func:`hrz_series_regression`.
horowitz_series_regression = hrz_series_regression
