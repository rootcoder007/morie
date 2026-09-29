# morie.fn -- function file (rootcoder007/morie)
"""Adaptive kernel weights (variable bandwidth)."""

from .spwgt import spatial_weights


def swadapt(coords, k=5, kernel="gaussian", style="B"):
    r"""Adaptive kernel weights: the bandwidth of point i is the distance to its k-th nearest neighbour and w_ij = K(d_ij / h_i) for i != j.

    Thin front-end to :func:`morie.fn.spwgt.spatial_weights` (method
    ``"kernel"``); ``style`` is an ``spdep::nb2listw`` coding (``B`` binary,
    ``W`` row-standardised, ...). Returns its ``SpatialResult``:
    ``statistic`` is the mean number of neighbours and ``extra["W"]`` the
    weights matrix.

    References
    ----------
    Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2002).
    *Geographically Weighted Regression*. Wiley.

    Examples
    --------
    >>> coords = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.4], [2, 0.5]]
    >>> r = swadapt(coords, k=3, kernel="quartic")
    >>> [sorted(v) for v in r.extra["neighbours"]][5]
    [1, 3]
    """
    r = spatial_weights(coords, "kernel", k=int(k), bandwidth=None, kernel=kernel, style=style)
    r.name = "swadapt"
    return r


swadapt_fn = swadapt


def cheatsheet() -> str:
    return "swadapt(coords, k=5) -> adaptive kernel weights, bandwidth = k-th neighbour distance."
