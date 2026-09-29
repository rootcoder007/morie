# morie.fn -- function file (rootcoder007/morie)
"""Moran's I (Moran 1950) with its normality-assumption test."""

from . import _robust_core as _rc
from ._containers import SpatialResult


def miorig(y, W, alternative="greater"):
    r"""Moran's (1950) I with the moments under the normality assumption.

    ``I = (n / S0) z'Wz / z'z`` with ``z = y - mean(y)``; ``E[I] = -1/(n-1)``
    and the variance is Moran's normality form, as ``spdep::moran.test(...,
    randomisation = FALSE)``. Thin front-end to
    :func:`morie.fn._robust_core.morans_i_test`.

    References
    ----------
    Moran, P. A. P. (1950). Notes on continuous stochastic phenomena.
    *Biometrika* 37, 17-23.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> round(miorig([1.0, 2.0, 3.0, 4.0], W).statistic, 12)
    0.333333333333
    """
    r = _rc.morans_i_test(y, W, randomisation=False, alternative=alternative)
    return SpatialResult(
        name="miorig",
        statistic=r["estimate"],
        p_value=r["p_value"],
        expected=r["expectation"],
        variance=r["variance"],
        extra={"z": r["statistic"], "alternative": alternative, "assumption": "normality"},
    )


miorig_fn = miorig


def cheatsheet() -> str:
    return (
        "miorig(y, W) -> Moran's (1950) I with normality-assumption E, Var, p (spdep::moran.test randomisation=FALSE)."
    )
