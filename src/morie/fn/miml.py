# morie.fn -- function file (rootcoder007/morie)
"""Moran's I test on residuals of a maximum-likelihood spatial model."""

from . import _robust_core as _rc
from ._containers import SpatialResult


def miml(resid, W, alternative="greater"):
    r"""Moran's I of residuals from a maximum-likelihood spatial model.

    After fitting a spatial lag or error model by ML the usual check for
    remaining autocorrelation is Moran's I of the (spatially filtered)
    residuals referred to its randomisation moments, ``spdep::moran.test(
    residuals(fit), listw)`` (Bivand, Pebesma and Gomez-Rubio 2013, sec.
    10.2.1). The moments treat the residuals as observed values and ignore the
    estimation effect, so the test is approximate; for OLS residuals use the
    exact :func:`morie.fn.miols.miols`.

    References
    ----------
    Bivand, R. S., Pebesma, E. and Gomez-Rubio, V. (2013). *Applied Spatial
    Data Analysis with R*, 2nd ed. Springer.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> round(miml([0.5, -1.0, 1.5, -1.0], W).statistic, 12)
    -1.037037037037
    """
    r = _rc.morans_i_test(resid, W, randomisation=True, alternative=alternative)
    return SpatialResult(
        name="miml",
        statistic=r["estimate"],
        p_value=r["p_value"],
        expected=r["expectation"],
        variance=r["variance"],
        extra={"z": r["statistic"], "alternative": alternative},
    )


miml_fn = miml


def cheatsheet() -> str:
    return "miml(resid, W) -> Moran's I of ML-model residuals, randomisation moments (spdep::moran.test)."
