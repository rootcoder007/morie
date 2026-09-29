# morie.fn -- function file (rootcoder007/morie)
"""Spatial zero-inflated Poisson (redirect to the real estimator)."""

from __future__ import annotations

from .spcount import sar_zip


def spatial_zip(y, X, W, *, Z=None, rho_bounds=(-0.99, 0.99)):
    r"""Spatial zero-inflated Poisson.

    Spatial-lag zero-inflated Poisson regression
    (:func:`morie.fn.spcount.sar_zip`): logit zero model ``Z gamma`` and count
    mean ``exp((I - rho W)^{-1} X beta)``, EM for fixed ``rho`` and profile
    likelihood over ``rho_bounds``. This module formerly returned the
    variance of its input.

    Parameters
    ----------
    y : non-negative counts.
    X : design matrix with an intercept column.
    W : spatial weights matrix.
    Z : zero-model design (default intercept only).
    rho_bounds : search interval for ``rho``.

    Returns
    -------
    RichResult
        ``count_coefficients``, ``zero_coefficients``, ``rho``, ``loglik``, ``k``.

    References
    ----------
    Lambert, D. (1992). Zero-inflated Poisson regression, with an application
    to defects in manufacturing. *Technometrics* 34, 1-14.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [0.5, 0, 0.5, 0, 0, 0], [0, 0.5, 0, 0.5, 0, 0], [0, 0, 0.5, 0, 0.5, 0], [0, 0, 0, 0.5, 0, 0.5], [0, 0, 0, 0, 1, 0]]
    >>> X = [[1, 0.1], [1, 0.3], [1, 0.5], [1, 0.9], [1, 0.2], [1, 0.4]]
    >>> r = spatial_zip([0, 0, 3, 5, 0, 2], X, W, rho_bounds=(0, 0))
    >>> r.loglik == sar_zip([0, 0, 3, 5, 0, 2], X, W, rho_bounds=(0, 0)).loglik
    True
    """
    return sar_zip(y, X, W, Z=Z, rho_bounds=rho_bounds)


spat = spatial_zip


def cheatsheet() -> str:
    return "spatial_zip(y, X, W, ...) -> Spatial zero-inflated Poisson"


# compact alias per ledger/NAMING.md
spatialzip = spatial_zip
