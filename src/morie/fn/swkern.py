# morie.fn -- function file (rootcoder007/morie)
"""Kernel spatial weights (Gaussian/bisquare)."""

from .spwgt import spatial_weights


def swkern(coords, bw=1.0, kernel="gaussian", style="B"):
    r"""Fixed-bandwidth kernel weights ``w_ij = K(d_ij / bw)`` for ``i != j`` with the uniform, triangular, Epanechnikov, quartic (bisquare) or Gaussian kernel.

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
    >>> r = swkern(coords, bw=1.5, kernel="quartic")
    >>> [sorted(v) for v in r.extra["neighbours"]][5]
    [1, 3]
    """
    r = spatial_weights(coords, "kernel", bandwidth=float(bw), kernel=kernel, style=style)
    r.name = "swkern"
    return r


swkern_fn = swkern


def cheatsheet() -> str:
    return "swkern(coords, bw=1.0, kernel='gaussian') -> fixed-bandwidth kernel weights."
