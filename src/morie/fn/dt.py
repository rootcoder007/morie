# morie.fn -- function file (rootcoder007/morie)
"""Student's t-distribution probability density function."""

from typing import Union

from . import _array_core as np
from . import _stats_core as stats


def dt(x: Union[float, np.ndarray], df: float, ncp: float | None = None, log: bool = False) -> Union[float, np.ndarray]:
    """
    Student's t-distribution probability density function.

    Mirrors R's ``dt(x, df, ncp, log)``.

    :param x: Quantile(s).
    :param df: Degrees of freedom (> 0).
    :param ncp: Non-centrality parameter. If None, central t is used.
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). dt {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    dist = stats.nct(df=df, nc=ncp) if ncp is not None else stats.t(df=df)
    return dist.logpdf(x) if log else dist.pdf(x)


def cheatsheet() -> str:
    return "dt({}) -> Student's t-distribution probability density function."
