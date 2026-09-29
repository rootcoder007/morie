# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel random effects estimator."""

from .sppanel import spatial_panel_re_lag
from .sppfe import _stack


def sppre(y, X, W, time_id, unit_id):
    r"""Random-effects spatial lag panel model by maximum likelihood (Elhorst 2003).

    y_t = rho W y_t + X_t beta + mu + e_t with mu_i ~ N(0, sigma_mu^2);
    the quasi-demeaning weight phi^2 = sigma^2 / (T sigma_mu^2 + sigma^2)
    and (rho, beta, sigma^2) are updated alternately (Breusch 1987). Rows
    are matched to W by sorting on (period, unit id); constant columns of
    X are dropped (an intercept is fitted). Thin front-end to
    :func:`morie.fn.sppanel.spatial_panel_re_lag`.

    References
    ----------
    Elhorst, J. P. (2003). Specification and estimation of spatial panel data
    models. *International Regional Science Review* 26, 244-268.
    Breusch, T. S. (1987). Maximum likelihood estimation of random effects
    models. *Journal of Econometrics* 36, 383-389.

    Examples
    --------
    >>> import math
    >>> W = [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
    >>> tid = [t for t in range(5) for _ in range(4)]
    >>> uid = [u for _ in range(5) for u in range(4)]
    >>> X = [[math.sin(1.3 * k) + 0.1 * k] for k in range(20)]
    >>> y = [1.0 + 0.8 * X[k][0] + 0.3 * math.cos(2.1 * k) + 0.2 * (k % 4) for k in range(20)]
    >>> round(sppre(y, X, W, tid, uid)["phi"], 8)
    0.44478295
    """
    yv, Xm, N, _ = _stack(y, X, time_id, unit_id)
    return spatial_panel_re_lag(yv, Xm, W, N)


sppre_fn = sppre


def cheatsheet() -> str:
    return "sppre(y, X, W, time_id, unit_id) -> random-effects spatial lag panel by ML (Elhorst 2003)."
