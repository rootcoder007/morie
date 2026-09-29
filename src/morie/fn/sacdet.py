# morie.fn -- function file (rootcoder007/morie)
"""SAC log-determinant product."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sacdet(W, rho, lam):
    r"""SAC log-determinant product.

    ``log|det(I - rho W)| + log|det(I - lambda W)|``, the two Jacobian terms
    of the SAC (SARAR) log-likelihood, each by LU decomposition (Anselin
    1988; :func:`morie.fn.spdurbin.log_jacobian`).

    Parameters
    ----------
    W : array-like, shape (n, n)
        Spatial weights (the same matrix in the lag and the error process).
    rho, lam : float
        Lag and error autoregressive parameters.

    Returns
    -------
    SpatialResult
        ``statistic`` is the sum; ``extra`` has the two terms.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Examples
    --------
    >>> round(sacdet([[0, 1], [1, 0]], 0.5, -0.5).statistic, 12)
    -0.575364144904
    """
    a, b = sd.logdet(W, rho), sd.logdet(W, lam)
    return SpatialResult(name="sacdet", statistic=a + b, extra={"logdet_rho": a, "logdet_lambda": b})


sacdet_fn = sacdet


def cheatsheet() -> str:
    return "sacdet(W, rho, lam) -> log|I - rho W| + log|I - lam W|"
