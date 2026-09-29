# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel random effects (redirect to the real estimator)."""

from __future__ import annotations

from .sppanel import spatial_panel_re_lag


def spatial_panel_re(y, X, W, n_units, *, interval=(-0.99, 0.99), tol: float = 1e-10, maxit: int = 500):
    r"""Spatial panel random effects.

    Random-effects spatial lag panel model by maximum likelihood (Elhorst
    2003; :func:`morie.fn.sppanel.spatial_panel_re_lag`). This module
    formerly returned the variance of its input.

    Parameters
    ----------
    y : outcome stacked by period.
    X : regressors (an intercept column is usual).
    W : ``n_units x n_units`` spatial weights.
    n_units : number of cross-sectional units.
    interval, tol, maxit : as in ``spatial_panel_re_lag``.

    Returns
    -------
    RichResult
        ``rho``, ``coefficients``, ``phi``, ``sigma2``, ``sigma2_mu``, ``loglik``.

    References
    ----------
    Elhorst, J. P. (2003). Specification and estimation of spatial panel data
    models. *International Regional Science Review* 26, 244-268.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> y = [1.0, 2.0, 1.5, 1.2, 2.4, 1.1, 0.9, 2.2, 1.8]
    >>> X = [[0.5], [1.0], [0.2], [0.7], [1.3], [0.1], [0.4], [0.9], [0.6]]
    >>> spatial_panel_re(y, X, W, 3).phi == spatial_panel_re_lag(y, X, W, 3).phi
    True
    """
    return spatial_panel_re_lag(y, X, W, n_units, interval=interval, tol=tol, maxit=maxit)


spat = spatial_panel_re


def cheatsheet() -> str:
    return "spatial_panel_re(y, X, W, ...) -> Spatial panel random effects"


# compact alias per ledger/NAMING.md
spatialpanelre = spatial_panel_re
