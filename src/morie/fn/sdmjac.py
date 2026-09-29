# morie.fn -- function file (rootcoder007/morie)
"""SDM Jacobian log-determinant."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sdmjac(W, rho):
    r"""SDM Jacobian log-determinant.

    ``log|det(I - rho W)|`` computed by LU decomposition with partial
    pivoting, so ``W`` need not be symmetric (the eigenvalue shortcut
    ``sum log(1 - rho ev)`` is only valid for real spectra); this is the
    Jacobian term of the spatial Durbin log-likelihood (Ord 1975) and equals
    :func:`morie.fn.spdurbin.log_jacobian`.

    Parameters
    ----------
    W : array-like, shape (n, n)
        Spatial weights.
    rho : float
        Autoregressive parameter.

    Returns
    -------
    SpatialResult
        ``statistic`` is the log-determinant.

    References
    ----------
    Ord, K. (1975). Estimation methods for models of spatial interaction. *Journal of the American
    Statistical Association*, 70(349), 120-126.

    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Examples
    --------
    >>> round(sdmjac([[0, 1], [1, 0]], 0.5).statistic, 12)
    -0.287682072452
    """
    return SpatialResult(name="sdmjac", statistic=sd.logdet(W, rho), extra={"rho": float(rho)})


sdmjac_fn = sdmjac


def cheatsheet() -> str:
    return "sdmjac(W, rho) -> log|I - rho W| (spatial Durbin Jacobian)"
