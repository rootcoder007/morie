# morie.fn -- function file (rootcoder007/morie)
"""Gabriel graph spatial weights."""

from .spwgt import spatial_weights


def swgab(coords, style="B"):
    r"""Gabriel graph neighbours: ``i ~ j`` unless some third point lies strictly inside the circle with diameter ``ij``, ``d_ik^2 + d_jk^2 < d_ij^2`` (Gabriel and Sokal 1969; ``spdep::gabrielneigh``).

    Thin front-end to :func:`morie.fn.spwgt.spatial_weights` (method
    ``"gabriel"``); ``style`` is an ``spdep::nb2listw`` coding (``B`` binary,
    ``W`` row-standardised, ...). Returns its ``SpatialResult``:
    ``statistic`` is the mean number of neighbours and ``extra["W"]`` the
    weights matrix.

    References
    ----------
    Gabriel, K. R. and Sokal, R. R. (1969). A new statistical approach to
    geographic variation analysis. *Systematic Zoology* 18, 259-278.

    Examples
    --------
    >>> coords = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.4], [2, 0.5]]
    >>> r = swgab(coords)
    >>> [sorted(v) for v in r.extra["neighbours"]][5]
    [1, 3]
    """
    r = spatial_weights(coords, "gabriel", style=style)
    r.name = "swgab"
    return r


swgab_fn = swgab


def cheatsheet() -> str:
    return "swgab(coords) -> Gabriel graph weights (spdep::gabrielneigh)."
