# morie.fn -- function file (rootcoder007/morie)
"""CAR Jacobian term."""

from ._containers import SpatialResult
from .sarreg import _logdet


def carjac(W, rho):
    r"""Jacobian (log-determinant) term of the Gaussian CAR log-likelihood, ``0.5 log|I - rho W|``.

    For ``y ~ N(X beta, sigma2 (I - rho W)^{-1})`` the log-likelihood is
    ``-n/2 log(2 pi sigma2) + 1/2 log|I - rho W| - r'(I - rho W) r /
    (2 sigma2)`` (``spatialreg::spautolm(family = "CAR")``); the Jacobian term
    is half the SAR one because the CAR precision enters once. Computed by LU
    with partial pivoting (``determinant`` in the R arm).

    References
    ----------
    Besag, J. (1974). Spatial interaction and the statistical analysis of
    lattice systems. *J. R. Stat. Soc. B* 36, 192-236.
    Ord, J. K. (1975). Estimation methods for models of spatial interaction.
    *JASA* 70, 120-126.

    Examples
    --------
    >>> W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> round(carjac(W, 0.4).statistic, 12)
    -0.192831240406
    """
    Wm = [[float(v) for v in r] for r in (W.tolist() if hasattr(W, "tolist") else W)]
    val = 0.5 * _logdet(Wm, float(rho))
    return SpatialResult(name="carjac", statistic=val, extra={"rho": float(rho)})


carjac_fn = carjac


def cheatsheet() -> str:
    return "carjac(W, rho) -> 0.5 log|I - rho W|, the CAR log-likelihood Jacobian term."
