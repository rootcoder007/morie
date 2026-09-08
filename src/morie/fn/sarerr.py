"""Spatial AR error model (SEM) -- alias of ``sarre``."""

from .sarre import spatial_ar_error

__all__ = ["spatial_ar_error_model"]


def spatial_ar_error_model(y, X, W):
    """
    Spatial autoregressive error model (SEM)::

        y = X beta + u,   u = lam W u + eps,   eps ~ N(0, sigma2 I)

    An alias.  The estimator already exists as ``sarre``
    (``spatial_ar_error``) -- the same concentrated maximum likelihood in
    ``lam`` over the admissible eigenvalue interval.  Carrying it a
    second time would give two copies that agree with each other at 1e-9
    forever and are never checked against anything else, so this only
    adapts the calling convention: ``sarre`` takes ``(x, y, w)``, this
    takes ``(y, X, W)``.

    ``ledger/wave2/DUPMAP.tsv`` originally recorded ``sarerr`` as a
    duplicate of ``lmerr``.  That is wrong and the correction is appended
    there: ``lmerr`` is Anselin's *Lagrange multiplier diagnostic* for
    spatial error dependence, a test statistic, not the estimator.

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, p)
        Design matrix; the intercept must be explicit.
    W : array-like, shape (n, n)

    Returns
    -------
    RichResult
        Whatever ``sarre`` returns, unchanged.

    References
    ----------
    Anselin, L. (1988). Spatial Econometrics: Methods and Models.
    Schabenberger, O. and Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis, Sec. 6.2.2, pp. 335-341.
    """
    return spatial_ar_error(X, y, W)


def cheatsheet():
    return "sarerr: spatial AR error model (alias of sarre)"


# compact alias per ledger/NAMING.md
spatialarerrormodel = spatial_ar_error_model
