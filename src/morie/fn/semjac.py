# morie.fn -- function file (rootcoder007/morie)
"""SEM Jacobian ln|I - lambda*W|."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def semjac(W, lam):
    r"""SEM Jacobian ln|I - lambda*W|.

    ``log|det(I - lam W)|`` computed by LU decomposition with partial
    pivoting, so ``W`` need not be symmetric (the eigenvalue shortcut
    ``sum log(1 - lam ev)`` is only valid for real spectra); this is the
    Jacobian term of the spatial error log-likelihood (Ord 1975) and equals
    :func:`morie.fn.spdurbin.log_jacobian`.

    Parameters
    ----------
    W : array-like, shape (n, n)
        Spatial weights.
    lam : float
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
    >>> round(semjac([[0, 1], [1, 0]], 0.5).statistic, 12)
    -0.287682072452
    """
    return SpatialResult(name="semjac", statistic=sd.logdet(W, lam), extra={"lam": float(lam)})


semjac_fn = semjac


def cheatsheet() -> str:
    return "semjac(W, lam) -> log|I - lam W| (spatial error Jacobian)"
