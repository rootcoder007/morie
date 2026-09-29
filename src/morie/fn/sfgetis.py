# morie.fn -- function file (rootcoder007/morie)
"""Getis spatial filtering approach."""

from .sfilter import getis_filter


def sfgetis(y, W):
    r"""Getis (1995) spatial filter of a positive variable: y_i* = y_i (W_i/(n-1)) / G_i.

    G_i = sum_{j != i} w_ij y_j / sum_{j != i} y_j is the local Getis-Ord
    statistic and W_i = sum_{j != i} w_ij (binary distance weights); the
    filtered variable y* is free of the spatial dependence captured by
    G_i and y - y* is the spatial component. Thin front-end to
    :func:`morie.fn.sfilter.getis_filter` (returns its filtered and
    spatial lists).

    References
    ----------
    Getis, A. (1995). Spatial filtering in a regression framework: examples
    using data on urban crime, regional inequality, and government
    expenditures. In L. Anselin and R. Florax (eds), *New Directions in
    Spatial Econometrics*. Springer, 172-185.

    Examples
    --------
    >>> [round(v, 6) for v in sfgetis([2.0, 4.0, 6.0], [[0, 1, 0], [1, 0, 1], [0, 1, 0]])["filtered"]]
    [2.5, 4.0, 4.5]
    """
    return getis_filter(y, W)


sfgetis_fn = sfgetis


def cheatsheet() -> str:
    return "sfgetis(y, W) -> Getis (1995) filtered variable y* and spatial component y - y*."
