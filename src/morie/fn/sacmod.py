"""SAC model -- spatial lag and spatial error sharing one weights matrix."""

from .sarmix import spatial_ar_combined

from ._richresult import RichResult

__all__ = ["spatial_combined"]


def spatial_combined(y, X, W):
    """
    SAC: combined spatial autoregressive lag and autoregressive error,
    with the *same* weights matrix on both terms::

        y = rho W y + X beta + u
        u = lam W u + eps,     eps ~ N(0, sigma2 I)

    This is the SARAR model of ``sarmix`` at ``W1 = W2 = W``; LeSage and
    Pace (2009) call it the SAC specification and note it is the general
    nesting model from which SAR (``lam = 0``) and SEM (``rho = 0``)
    drop out.  Carrying the concentrated likelihood a second time would
    give two copies that agree with each other at 1e-9 forever and are
    never checked against anything else, so this only fixes the second
    weights argument -- the audit note is recorded in
    ``ledger/wave2/DUPMAP.tsv``.

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, p)
        Design matrix; the intercept must be explicit.
    W : array-like, shape (n, n)
        Spatial weights, used for both the lag and the disturbance.

    Returns
    -------
    RichResult
        Whatever ``sarmix`` returns, with ``method`` relabelled.

    References
    ----------
    LeSage, J. and Pace, R. K. (2009). Introduction to Spatial
    Econometrics. Chapman & Hall/CRC, Sec. 2.4 (the SAC model).
    Kelejian, H. H. and Prucha, I. R. (1998). The Journal of Real Estate
    Finance and Economics 17(1), 99-121. doi:10.1023/A:1007707430416.
    Anselin, L. (1988). Spatial Econometrics: Methods and Models.
    """
    res = spatial_ar_combined(y, X, W, W)
    pay = dict(res)
    pay["method"] = "SAC (spatial lag + spatial error, one W) by concentrated ML"
    return RichResult(payload=pay)


def cheatsheet():
    return "sacmod: SAC combined spatial lag + spatial error model"


# compact alias per ledger/NAMING.md
spatialcombined = spatial_combined
