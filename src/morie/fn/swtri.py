# morie.fn -- function file (rootcoder007/morie)
"""Triangulate weights from Delaunay tessellation."""

from .spwgt import spatial_weights


def swtri(coords, style="B"):
    r"""Delaunay triangulation neighbours (Bowyer-Watson), as ``spdep::tri2nb``.

    Thin front-end to :func:`morie.fn.spwgt.spatial_weights` (method
    ``"delaunay"``); ``style`` is an ``spdep::nb2listw`` coding (``B`` binary,
    ``W`` row-standardised, ...). Returns its ``SpatialResult``:
    ``statistic`` is the mean number of neighbours and ``extra["W"]`` the
    weights matrix.

    References
    ----------
    Bowyer, A. (1981). Computing Dirichlet tessellations. *Computer Journal*
    24, 162-166.

    Examples
    --------
    >>> coords = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.4], [2, 0.5]]
    >>> r = swtri(coords)
    >>> [sorted(v) for v in r.extra["neighbours"]][5]
    [1, 3]
    """
    r = spatial_weights(coords, "delaunay", style=style)
    r.name = "swtri"
    return r


swtri_fn = swtri


def cheatsheet() -> str:
    return "swtri(coords) -> Delaunay triangulation weights (spdep::tri2nb)."
