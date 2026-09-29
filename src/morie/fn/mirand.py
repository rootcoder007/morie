# morie.fn -- function file (rootcoder007/morie)
"""Moran's I test under the randomisation assumption."""

from . import _robust_core as _rc
from ._containers import SpatialResult


def mirand(y, W, alternative="greater"):
    r"""Moran's I test under the randomisation assumption (Cliff and Ord 1981).

    Conditions on the observed values and treats only their arrangement as
    random: the variance uses the sample kurtosis ``b2`` (see :func:`mivar`),
    as ``spdep::moran.test`` with its default ``randomisation = TRUE``. Thin
    front-end to :func:`morie.fn._robust_core.morans_i_test`.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion, London.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> round(mirand([1.0, 2.0, 3.0, 4.0], W).expected, 12)
    -0.333333333333
    """
    r = _rc.morans_i_test(y, W, randomisation=True, alternative=alternative)
    return SpatialResult(
        name="mirand",
        statistic=r["estimate"],
        p_value=r["p_value"],
        expected=r["expectation"],
        variance=r["variance"],
        extra={"z": r["statistic"], "alternative": alternative, "assumption": "randomisation"},
    )


mirand_fn = mirand


def cheatsheet() -> str:
    return "mirand(y, W) -> Moran's I test under randomisation (spdep::moran.test default)."
