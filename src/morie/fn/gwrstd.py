# morie.fn -- function file (rootcoder007/morie)
"""GWR local standard errors of coefficients."""

from .gwrcoef import _fit


def gwrstd(y, X, coords, bw=0.5, kernel="bisquare", adaptive=False):
    r"""Standard errors of the local GWR coefficients, ``sqrt(sigma2 diag(C_i C_i^T))`` with ``C_i = (X^T W_i X)^{-1} X^T W_i`` and ``sigma2 = RSS / (n - 2 tr S + tr S^T S)`` (Fotheringham, Brunsdon and Charlton 2002, sec. 4.5; ``GWmodel::gwr.basic``).

    References
    ----------
    Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2002).
    *Geographically Weighted Regression*. Wiley.

    Examples
    --------
    >>> P = [(float(i % 4), float(i // 4)) for i in range(16)]
    >>> X = [[(0.3 * i) % 1.7] for i in range(16)]
    >>> y = [1.0 + 2.0 * X[i][0] + 0.1 * P[i][0] + 0.05 * (i % 3) for i in range(16)]
    >>> round(gwrstd(y, X, P, 3.0, kernel="gaussian")[0][1], 10)
    0.0574752116
    """
    return _fit(y, X, coords, bw, kernel, adaptive)["se"]


gwrstd_fn = gwrstd


def cheatsheet() -> str:
    return "gwrstd(y, X, coords, bw) -> local GWR standard errors sqrt(sigma2 diag(C_i C_i^T))."
