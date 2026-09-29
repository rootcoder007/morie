# morie.fn -- function file (rootcoder007/morie)
"""GWR delta test for coefficient stationarity."""

from .gwrcoef import _setup
from .gwrext import gwr_f_tests


def gwrdlt(y, X, coords, bw=0.5, kernel="bisquare", adaptive=False, method="leung"):
    r"""Leung-Mei-Zhang tests of GWR non-stationarity built on the traces delta_1 = tr R, delta_2 = tr R^2.

    With R = (I - S)^T (I - S) the GWR residual sum of squares is
    approximated by delta_1 sigma^2 with delta_1^2 / delta_2 degrees of
    freedom; F1/F2 compare GWR with OLS and F3 tests, coefficient
    by coefficient, whether the local estimates vary more than sampling
    error allows (Leung, Mei and Zhang 2000). Thin front-end to
    :func:`morie.fn.gwrext.gwr_f_tests` (method="gwmodel" reproduces
    GWmodel::gwr.basic(F123.test = TRUE)).

    References
    ----------
    Leung, Y., Mei, C.-L. and Zhang, W.-X. (2000). Statistical tests for
    spatial nonstationarity based on the geographically weighted regression
    model. *Environment and Planning A* 32, 9-32.

    Examples
    --------
    >>> P = [(float(i % 4), float(i // 4)) for i in range(16)]
    >>> X = [[(0.3 * i) % 1.7] for i in range(16)]
    >>> y = [1.0 + 2.0 * X[i][0] + 0.1 * P[i][0] + 0.05 * (i % 3) for i in range(16)]
    >>> r = gwrdlt(y, X, P, 3.0, kernel="gaussian")
    >>> round(r["F4"], 10)
    0.7815064369
    """
    yv, Xm, P, _, _ = _setup(y, X, coords, bw, kernel, adaptive)
    return gwr_f_tests(yv, Xm, P, bw, kernel=kernel, adaptive=adaptive, method=method)


gwrdlt_fn = gwrdlt


def cheatsheet() -> str:
    return "gwrdlt(y, X, coords, bw) -> Leung-Mei-Zhang F1-F4 tests of GWR non-stationarity."
