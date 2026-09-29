# morie.fn -- function file (rootcoder007/morie)
"""Distance-band spatial weights matrix."""

from .spwgt import spatial_weights


def swdist(coords, d=1.0, style="B", d_min=0.0):
    r"""Distance-band neighbours ``d_min < d_ij <= d`` (``spdep::dnearneigh``).

    Thin front-end to :func:`morie.fn.spwgt.spatial_weights` (method
    ``"distance"``); ``style`` is an ``spdep::nb2listw`` coding (``B`` binary,
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
    >>> r = swdist(coords, d=1.2)
    >>> [sorted(v) for v in r.extra["neighbours"]][5]
    [1, 3]
    """
    r = spatial_weights(coords, "distance", threshold=float(d), style=style, d_min=float(d_min))
    r.name = "swdist"
    return r


swdist_fn = swdist


def cheatsheet() -> str:
    return "swdist(coords, d=1.0) -> distance-band weights d_min < d_ij <= d (spdep::dnearneigh)."
