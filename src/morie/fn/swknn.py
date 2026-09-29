# morie.fn -- function file (rootcoder007/morie)
"""k-nearest-neighbour spatial weights matrix."""

from .spwgt import spatial_weights


def swknn(coords, k=4, style="B"):
    r"""The ``k`` nearest neighbours of each point, ties broken by index (``spdep::knn2nb(knearneigh(coords, k))``); not symmetric in general.

    Thin front-end to :func:`morie.fn.spwgt.spatial_weights` (method
    ``"knn"``); ``style`` is an ``spdep::nb2listw`` coding (``B`` binary,
    ``W`` row-standardised, ...). Returns its ``SpatialResult``:
    ``statistic`` is the mean number of neighbours and ``extra["W"]`` the
    weights matrix.

    References
    ----------
    Bivand, R. S., Pebesma, E. and Gomez-Rubio, V. (2013). *Applied Spatial
    Data Analysis with R*, 2nd ed., ch. 9. Springer.

    Examples
    --------
    >>> coords = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.4], [2, 0.5]]
    >>> r = swknn(coords, k=2)
    >>> [sorted(v) for v in r.extra["neighbours"]][5]
    [1, 3]
    """
    r = spatial_weights(coords, "knn", k=int(k), style=style)
    r.name = "swknn"
    return r


swknn_fn = swknn


def cheatsheet() -> str:
    return "swknn(coords, k=4) -> k-nearest-neighbour weights (spdep::knearneigh)."
