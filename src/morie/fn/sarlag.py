"""Spatial AR lag model -- alias of ``sarla``."""

from .sarla import spatial_ar_lag

__all__ = ["spatial_ar_lag_model"]


def spatial_ar_lag_model(y, X, W):
    """
    Spatial autoregressive lag model::

        y = rho W y + X beta + eps,   eps ~ N(0, sigma2 I)

    An alias.  The estimator already exists as ``sarla``
    (``spatial_ar_lag``) -- the same concentrated maximum likelihood in
    ``rho``; ``ledger/wave2/DUPMAP.tsv`` records ``sarlag`` as a
    duplicate of ``sarla``.  Carrying the likelihood a second time would
    give two copies that agree with each other at 1e-9 forever and are
    never checked against anything else, so this only adapts the calling
    convention: ``sarla`` takes ``(x, y, w)``, this takes ``(y, X, W)``.

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, p)
        Design matrix; the intercept must be explicit.
    W : array-like, shape (n, n)

    Returns
    -------
    RichResult
        Whatever ``sarla`` returns, unchanged.

    References
    ----------
    Anselin, L. (1988). Spatial Econometrics: Methods and Models.
    Ord, J. K. (1975). Estimation methods for models of spatial
    interaction. Journal of the American Statistical Association 70(349),
    120-126.
    """
    return spatial_ar_lag(X, y, W)


def cheatsheet():
    return "sarlag: spatial AR lag model (alias of sarla)"


# compact alias per ledger/NAMING.md
spatialarlagmodel = spatial_ar_lag_model
