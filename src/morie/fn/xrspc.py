# morie.fn -- function file (rootcoder007/morie)
"""Spatial Poisson regression (redirect to the real estimator)."""

from __future__ import annotations

from .spcount import sar_poisson


def spatial_poisson(y, X, W, *, rho_bounds=(-0.99, 0.99)):
    r"""Spatial Poisson regression.

    Spatial-lag Poisson regression ``E y = exp((I - rho W)^{-1} X beta)`` by
    profile maximum likelihood (:func:`morie.fn.spcount.sar_poisson`). This
    module formerly returned the variance of its input.

    Parameters
    ----------
    y : non-negative counts.
    X : design matrix with an intercept column.
    W : spatial weights matrix.
    rho_bounds : search interval for ``rho``.

    Returns
    -------
    RichResult
        ``coefficients``, ``rho``, ``fitted``, ``loglik``, ``k``.

    References
    ----------
    Lambert, D. M., Brown, J. P. and Florax, R. J. G. M. (2010). A two-step
    estimator for a spatial lag model of counts. *Regional Science and Urban
    Economics* 40, 241-252.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [0.5, 0, 0.5, 0, 0, 0], [0, 0.5, 0, 0.5, 0, 0], [0, 0, 0.5, 0, 0.5, 0], [0, 0, 0, 0.5, 0, 0.5], [0, 0, 0, 0, 1, 0]]
    >>> X = [[1, 0.1], [1, 0.3], [1, 0.5], [1, 0.9], [1, 0.2], [1, 0.4]]
    >>> r = spatial_poisson([1, 0, 3, 5, 1, 2], X, W)
    >>> r.rho == sar_poisson([1, 0, 3, 5, 1, 2], X, W).rho
    True
    """
    return sar_poisson(y, X, W, rho_bounds=rho_bounds)


spat = spatial_poisson


def cheatsheet() -> str:
    return "spatial_poisson(y, X, W, ...) -> Spatial Poisson regression"


# compact alias per ledger/NAMING.md
spatialpoisson = spatial_poisson
