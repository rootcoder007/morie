# morie.fn -- function file (rootcoder007/morie)
"""GWR leave-one-out cross-validation score."""

from ._qpcore import ssum
from .gwrcoef import _setup, _wls


def gwrcv(y, X, coords, bw=0.5, kernel="bisquare", adaptive=False):
    r"""Leave-one-out cross-validation score ``CV = sum_i (y_i - x_i beta_{(-i)})^2`` where ``beta_{(-i)}`` is the local estimate at ``i`` with ``w_ii = 0`` (Cleveland 1979; ``GWmodel::gwr.cv``), the bandwidth-selection criterion of Brunsdon, Fotheringham and Charlton (1996).

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
    >>> round(gwrcv(y, X, P, 3.0, kernel="gaussian"), 10)
    0.2163109487
    """
    yv, Xm, _, _, Wc = _setup(y, X, coords, bw, kernel, adaptive)
    cv = 0.0
    for i, w in enumerate(Wc):
        w = list(w)
        w[i] = 0.0
        b, _ = _wls(Xm, w, yv)
        cv += (yv[i] - ssum(Xm[i][a] * b[a] for a in range(len(b)))) ** 2
    return cv


gwrcv_fn = gwrcv


def cheatsheet() -> str:
    return "gwrcv(y, X, coords, bw) -> leave-one-out CV score (GWmodel::gwr.cv)."
