# morie.fn -- function file (rootcoder007/morie)
"""SAC robust (HC) standard errors."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult
from .gmsar import gs2sls_sac


def sacrob(y, X, W, robust="HC0"):
    r"""SAC robust (HC) standard errors.

    Heteroskedasticity-robust inference for the SAC model ``y = rho W y + X
    beta + u``, ``u = lambda W u + e``: generalised spatial two-stage least
    squares (Kelejian and Prucha 1998) with the White sandwich covariance of
    ``(rho, beta)`` in the final 2SLS step (``robust = "HC0"``, or
    ``"HC1"`` with the ``n / (n - k)`` correction, i.e. ``sig2n_k = TRUE``;
    Kelejian and Prucha 2010), exactly ``spatialreg::gstsls(robust = TRUE)`` via
    :func:`morie.fn.gmsar.gs2sls_sac`. Maximum likelihood has no
    heteroskedasticity-robust form, hence the GM estimator.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p)
        Design matrix; a leading column of ones is the intercept.
    W : array-like, shape (n, n)
        Spatial weights.
    robust : str
        ``"HC0"`` or ``"HC1"``.

    Returns
    -------
    SpatialResult
        ``statistic`` is ``rho``; ``extra`` has ``coefficients`` (``rho``
        first), ``se``, ``cov`` and ``lambda``.

    References
    ----------
    Kelejian, H. H. and Prucha, I. R. (2010). Specification and estimation of spatial autoregressive
    models with autoregressive and heteroskedastic disturbances. *Journal of Econometrics*, 157(1), 53-67.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> X = [[1, .1], [1, .7], [1, .2], [1, .9], [1, .4], [1, .6]]
    >>> r = sacrob([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], X, W)
    >>> round(r.statistic, 8), round(r.extra["se"][0], 8)
    (0.20083429, 0.0125366)
    """
    if robust not in ("HC0", "HC1"):
        raise ValueError("robust must be 'HC0' or 'HC1'")
    v = gs2sls_sac(sd._vec(y), sd._mat(X), sd._mat(W), robust=robust, sig2n_k=robust == "HC1").value
    return SpatialResult(
        name="sacrob",
        statistic=v["rho"],
        extra={"coefficients": list(v["coefficients"]), "se": list(v["se"]), "cov": v["cov"], "lambda": v["lambda"]},
    )


sacrob_fn = sacrob


def cheatsheet() -> str:
    return "sacrob(y, X, W, robust) -> GS2SLS SAC with HC sandwich covariance"
