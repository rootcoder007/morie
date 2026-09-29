# morie.fn -- function file (rootcoder007/morie)
"""Normal-approximation test of Moran's I from the statistic and the weights constants."""

from ._containers import SpatialResult
from .miexp import miexp
from .mivar import mivar
from .mizval import mizval


def minorm(I, n, S0, S1, S2, b2=None, alternative="greater"):  # noqa: E741
    r"""Normal-approximation p-value of Moran's I.

    Combines ``E[I] = -1/(n-1)`` (:func:`miexp`), the normality (or, with the
    sample kurtosis ``b2``, the randomisation) variance (:func:`mivar`) and the
    standard normal tail of ``z = (I - E[I]) / sqrt(Var[I])`` (:func:`mizval`),
    exactly as ``spdep::moran.test``.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion, London.

    Examples
    --------
    >>> r = minorm(0.2, 4, 6.0, 12.0, 40.0)
    >>> round(r.statistic, 12)
    1.385640646055
    """
    e = miexp(n).statistic
    v = mivar(n, S0, S1, S2, b2=b2).statistic
    r = mizval(I, e, v, alternative=alternative)
    return SpatialResult(
        name="minorm",
        statistic=r.statistic,
        p_value=r.p_value,
        expected=e,
        variance=v,
        extra={"I": float(I), "alternative": alternative, "assumption": "normality" if b2 is None else "randomisation"},
    )


minorm_fn = minorm


def cheatsheet() -> str:
    return "minorm(I, n, S0, S1, S2, b2=None, alternative='greater') -> Moran's I normal-approximation z and p."
