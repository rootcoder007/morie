# morie.fn -- function file (rootcoder007/morie)
"""Moran's I of spatially filtered residuals."""

from .miml import miml


def sfmi(resid_f, W, alternative="greater"):
    r"""Moran's I of the residuals of a spatially filtered regression, with randomisation moments.

    The check that an eigenvector filter has removed the residual spatial
    autocorrelation (Tiefelsdorf and Griffith 2007): Moran's I of the
    filtered-model residuals referred to its randomisation moments. Thin
    front-end to :func:`morie.fn.miml.miml`.

    References
    ----------
    Tiefelsdorf, M. and Griffith, D. A. (2007). Semiparametric filtering of
    spatial autocorrelation: the eigenvector approach. *Environment and
    Planning A* 39, 1193-1221.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> round(sfmi([0.5, -1.0, 1.5, -1.0], W).statistic, 12)
    -1.037037037037
    """
    r = miml(resid_f, W, alternative=alternative)
    r.name = "sfmi"
    return r


sfmi_fn = sfmi


def cheatsheet() -> str:
    return "sfmi(resid_f, W) -> Moran's I of spatially filtered residuals (randomisation moments)."
