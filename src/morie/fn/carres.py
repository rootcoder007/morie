# morie.fn -- function file (rootcoder007/morie)
"""CAR residual Moran test."""

from .miml import miml


def carres(resid, W, alternative="greater"):
    r"""Moran's I test of residual spatial autocorrelation after a CAR fit.

    Moran's I of the model residuals with randomisation moments (Cliff and
    Ord 1981), the check applied after ``spatialreg::spautolm`` (Bivand,
    Pebesma and Gomez-Rubio 2013, ch. 10). Thin front-end to
    :func:`morie.fn.miml.miml`.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion, London.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> round(carres([0.5, -1.0, 1.5, -1.0], W).statistic, 12)
    -1.037037037037
    """
    r = miml(resid, W, alternative=alternative)
    r.name = "carres"
    return r


carres_fn = carres


def cheatsheet() -> str:
    return "carres(resid, W) -> Moran's I of CAR-model residuals, randomisation moments."
