# morie.fn -- function file (rootcoder007/morie)
"""Spatial logit model (redirect to the real estimator)."""

from __future__ import annotations

from .spdiscrete import spatial_logit_gmm


def spatial_logit(y, X, W, *, Z=None):
    r"""Spatial logit model.

    The linearized GMM spatial logit of Klier and McMillen (2008)
    (:func:`morie.fn.spdiscrete.spatial_logit_gmm`), with instruments ``Z``
    (default ``[X, W X]``). This module formerly returned the variance of its input.

    Parameters
    ----------
    y : 0/1 outcomes.
    X : design matrix with an intercept column.
    W : spatial weights matrix (usually row-standardised).
    Z : instrument matrix (optional).

    Returns
    -------
    RichResult
        ``coefficients``, ``rho``, ``se``, ``logit``.

    References
    ----------
    Klier, T. and McMillen, D. P. (2008). Clustering of auto supplier plants in
    the United States: generalized method of moments spatial logit for large
    samples. *Journal of Business and Economic Statistics* 26, 460-471.

    Examples
    --------
    >>> W = [[1.0 if abs(i - j) == 1 else 0.0 for j in range(10)] for i in range(10)]
    >>> W = [[v / sum(r) for v in r] for r in W]
    >>> X = [[1.0, v] for v in (2.0, -1.0, 0.1, 1.5, 0.6, -0.4, 0.9, -1.3, 0.2, 1.1)]
    >>> y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
    >>> spatial_logit(y, X, W).rho == spatial_logit_gmm(y, X, W).rho
    True
    """
    return spatial_logit_gmm(y, X, W, Z=Z)


spat = spatial_logit


def cheatsheet() -> str:
    return "spatial_logit(y, X, W, ...) -> Spatial logit model"


# compact alias per ledger/NAMING.md
spatiallogit = spatial_logit
