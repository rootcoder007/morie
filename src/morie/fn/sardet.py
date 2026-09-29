# morie.fn -- function file (rootcoder007/morie)
"""SAR log-determinant ln|I - rho*W|."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sardet(W, rho):
    r"""SAR log-determinant ln|I - rho*W|.

    ``log|det(I - rho W)|`` computed by LU decomposition with partial
    pivoting, so ``W`` need not be symmetric (the eigenvalue shortcut
    ``sum log(1 - rho ev)`` is only valid for real spectra); this is the
    Jacobian term of the spatial lag log-likelihood (Ord 1975) and equals
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
    >>> round(sardet([[0, 1], [1, 0]], 0.5).statistic, 12)
    -0.287682072452
    """
    return SpatialResult(name="sardet", statistic=sd.logdet(W, rho), extra={"rho": float(rho)})


sardet_fn = sardet


def cheatsheet() -> str:
    return "sardet(W, rho) -> log|I - rho W| (spatial lag Jacobian)"
