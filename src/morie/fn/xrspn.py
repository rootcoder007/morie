# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel fixed effects (redirect to the real estimator)."""

from __future__ import annotations

from .sppanel import spatial_panel_ml


def spatial_panel_fe(
    y, X, W, n_units, *, model: str = "lag", effects: str = "individual", lee_yu: bool = False, interval=(-0.99, 0.99)
):
    r"""Spatial panel fixed effects.

    Maximum likelihood fixed-effects spatial panel model of Elhorst (2003)
    (:func:`morie.fn.sppanel.spatial_panel_ml`): ``model`` is ``"lag"``,
    ``"error"`` or ``"durbin"``; the data (stacked by period, ``n_units``
    units per period) are demeaned for ``effects``. This module formerly
    returned the variance of its input.

    Parameters
    ----------
    y : outcome stacked by period.
    X : regressors (no intercept).
    W : ``n_units x n_units`` spatial weights.
    n_units : number of cross-sectional units.
    model, effects, lee_yu, interval : as in ``spatial_panel_ml``.

    Returns
    -------
    RichResult
        ``rho``, ``coefficients``, ``sigma2``, ``loglik``, ``residuals``, ``n_obs``.

    References
    ----------
    Elhorst, J. P. (2003). Specification and estimation of spatial panel data
    models. *International Regional Science Review* 26, 244-268.

    Examples
    --------
    >>> W = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
    >>> y = [1.0, 2.0, 1.5, 1.2, 2.4, 1.1, 0.9, 2.2, 1.8]
    >>> X = [[0.5], [1.0], [0.2], [0.7], [1.3], [0.1], [0.4], [0.9], [0.6]]
    >>> spatial_panel_fe(y, X, W, 3).rho == spatial_panel_ml(y, X, W, 3).rho
    True
    """
    if effects not in ("individual", "time", "twoways"):
        raise ValueError("effects must be 'individual', 'time' or 'twoways'")
    return spatial_panel_ml(y, X, W, n_units, model=model, effects=effects, lee_yu=lee_yu, interval=interval)


spat = spatial_panel_fe


def cheatsheet() -> str:
    return "spatial_panel_fe(y, X, W, ...) -> Spatial panel fixed effects"


# compact alias per ledger/NAMING.md
spatialpanelfe = spatial_panel_fe
