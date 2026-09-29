# morie.fn -- function file (rootcoder007/morie)
"""GWR local residuals."""

from .gwrcoef import _fit


def gwrres(y, X, coords, bw=0.5, kernel="bisquare", adaptive=False):
    r"""GWR residuals ``e_i = y_i - x_i beta_i`` with ``beta_i`` the local estimate at location ``i`` (``GWmodel::gwr.basic``; see :func:`morie.fn.gwrcoef.gwrcoef`).

    References
    ----------
    Brunsdon, C., Fotheringham, A. S. and Charlton, M. E. (1996).
    Geographically weighted regression: a method for exploring spatial
    nonstationarity. *Geographical Analysis* 28, 281-298.

    Examples
    --------
    >>> P = [(float(i % 4), float(i // 4)) for i in range(16)]
    >>> X = [[(0.3 * i) % 1.7] for i in range(16)]
    >>> y = [1.0 + 2.0 * X[i][0] + 0.1 * P[i][0] + 0.05 * (i % 3) for i in range(16)]
    >>> round(gwrres(y, X, P, 3.0, kernel="gaussian")[5], 10)
    -0.0605255395
    """
    return _fit(y, X, coords, bw, kernel, adaptive)["residuals"]


gwrres_fn = gwrres


def cheatsheet() -> str:
    return "gwrres(y, X, coords, bw) -> GWR residuals y_i - x_i beta_i."
