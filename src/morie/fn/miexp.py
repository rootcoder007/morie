# morie.fn -- function file (rootcoder007/morie)
"""Expected value of Moran's I under the null of no spatial autocorrelation."""

from ._containers import SpatialResult


def miexp(n):
    r"""Expected value of Moran's I, ``E[I] = -1 / (n - 1)``.

    Under the null of no spatial autocorrelation the expectation of Moran's I
    is the same under the normality and the randomisation assumptions and
    depends only on the number of observations (Cliff and Ord 1981, eq. 1.35;
    ``spdep::moran.test`` "Expectation").

    Parameters
    ----------
    n : int
        Number of observations (at least 2).

    Returns
    -------
    SpatialResult
        ``statistic`` and ``expected`` are ``-1 / (n - 1)``.

    References
    ----------
    Cliff, A. D. and Ord, J. K. (1981). *Spatial Processes: Models and
    Applications*. Pion, London.

    Examples
    --------
    >>> round(miexp(10).statistic, 12)
    -0.111111111111
    """
    n = int(n)
    if n < 2:
        raise ValueError("n must be at least 2")
    e = -1.0 / (n - 1.0)
    return SpatialResult(name="miexp", statistic=e, expected=e, extra={"n": n})


miexp_fn = miexp


def cheatsheet() -> str:
    return "miexp(n) -> E[I] = -1/(n-1), the null expectation of Moran's I (Cliff and Ord 1981)."
