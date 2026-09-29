# morie.fn -- function file (rootcoder007/morie)
"""CAR GMM estimation."""

from ._containers import SpatialResult
from .sgcar import car_rho_ols


def cargmm(y, W, X=None):
    r"""Method-of-moments (conditional least squares) estimator of the CAR parameter.

    In the CAR model ``E[e_i | e_-i] = rho sum_j w_ij e_j``, so the
    innovation ``e_i - rho (We)_i`` is orthogonal to ``(We)_i``; the sample
    analogue of this single moment condition, applied to the OLS residuals
    ``e`` of ``y`` on ``X`` (intercept only by default), gives
    ``rho = e'We / e'W^2 e`` for symmetric ``W`` (Besag 1975; Haining 1990,
    p. 130), which is consistent for the one-parameter CAR model. Thin
    front-end to :func:`morie.fn.sgcar.car_rho_ols`.

    References
    ----------
    Besag, J. (1975). Statistical analysis of non-lattice data. *The
    Statistician* 24, 179-195.
    Haining, R. (1990). *Spatial Data Analysis in the Social and Environmental
    Sciences*. Cambridge University Press.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> round(cargmm([1.0, 2.0, 2.5, 4.5], W).statistic, 12)
    0.315789473684
    """
    rho = float(car_rho_ols(y, W, X))
    return SpatialResult(name="cargmm", statistic=rho, extra={"estimator": "conditional least squares moment"})


cargmm_fn = cargmm


def cheatsheet() -> str:
    return "cargmm(y, W, X=None) -> CAR rho from the moment E[e (We)] = 0: e'We / e'W^2 e."
