# morie.fn -- function file (rootcoder007/morie)
"""SEM Wald test on spatial error parameter lambda."""

from __future__ import annotations

from . import _spdiag as sd
from ._containers import SpatialResult


def semwald(lam, se_lam):
    r"""SEM Wald test on spatial error parameter lambda.

    ``W = (lam / se)^2`` against chi-square(1), equivalently the
    two-sided z test ``z = lam / se`` (Anselin 1988, sec. 6.3);
    :func:`morie.fn.spdurbin.spatial_wald_test` with a 1 x 1 covariance.

    Parameters
    ----------
    lam : float
        Estimate.
    se_lam : float
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
    >>> r = semwald(0.3, 0.1)
    >>> round(r.statistic, 12), round(r.p_value, 12)
    (9.0, 0.002699796063)
    """
    z = float(lam) / float(se_lam)
    return SpatialResult(name="semwald", statistic=z * z, p_value=sd._upper_p(z * z, 1), extra={"z": z, "df": 1})


semwald_fn = semwald


def cheatsheet() -> str:
    return "semwald(lam, se_lam) -> Wald (est/se)^2, chi-square(1)"
