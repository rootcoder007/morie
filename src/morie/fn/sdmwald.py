# morie.fn -- function file (rootcoder007/morie)
"""SDM Wald test on rho."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def sdmwald(rho, se_rho):
    r"""SDM Wald test on rho.

    ``W = (rho / se)^2`` against chi-square(1), equivalently the
    two-sided z test ``z = rho / se`` (Anselin 1988, sec. 6.3);
    :func:`morie.fn.spdurbin.spatial_wald_test` with a 1 x 1 covariance.

    Parameters
    ----------
    rho : float
        Estimate.
    se_rho : float
        Its standard error.

    Returns
    -------
    SpatialResult
        ``statistic`` (Wald), ``p_value``; ``extra["z"]``.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer, Dordrecht.

    Examples
    --------
    >>> r = sdmwald(0.3, 0.1)
    >>> round(r.statistic, 12), round(r.p_value, 12)
    (9.0, 0.002699796063)
    """
    z = float(rho) / float(se_rho)
    return SpatialResult(name="sdmwald", statistic=z * z, p_value=sd._upper_p(z * z, 1), extra={"z": z, "df": 1})


sdmwald_fn = sdmwald


def cheatsheet() -> str:
    return "sdmwald(rho, se_rho) -> Wald (est/se)^2, chi-square(1)"
