# morie.fn -- function file (rootcoder007/morie)
"""Moran scatterplot quadrant classification."""

from __future__ import annotations

from . import _lattice as lat
from ._containers import SpatialResult


def lacscat(y, W):
    r"""Moran scatterplot quadrant classification.

    The Moran scatterplot plots the spatial lag ``Wz`` against ``z = y -
    ybar`` (Anselin 1996). Each unit is classified as 1 high-high (``z >
    0``, ``Wz > 0``), 2 low-high, 3 low-low or 4 high-low (``z > 0``, ``Wz
    <= 0``); the slope of the least-squares line through the origin is
    ``z'Wz / z'z``, which is Moran's I when ``S0 = n`` (row-standardised
    ``W``).

    Parameters
    ----------
    y : array-like, shape (n,)
        Variable observed on the n lattice units.
    W : array-like, shape (n, n)
        Spatial weights (zero diagonal).

    Returns
    -------
    SpatialResult
        ``statistic`` is the slope; ``local_values`` the quadrant codes;
        ``extra`` has ``counts`` (HH, LH, LL, HL), ``z`` and ``lag``.

    References
    ----------
    Anselin, L. (1996). The Moran scatterplot as an ESDA tool to assess local instability in spatial
    association. In Fischer, Scholten and Unwin (eds), *Spatial Analytical Perspectives on GIS*,
    111-125. Taylor and Francis.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = lacscat([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], W)
    >>> round(r.statistic, 12), r.local_values
    (-0.612893982808, [2, 4, 2, 4, 2, 4])
    """
    q, z, lz = lat.quadrants(lat.vec(y), lat.mat(W))
    slope = lat.ssum(a * b for a, b in zip(z, lz)) / lat.ssum(a * a for a in z)
    return SpatialResult(
        name="lacscat",
        statistic=slope,
        local_values=q,
        extra={"counts": [q.count(k) for k in (1, 2, 3, 4)], "z": z, "lag": lz},
    )


lacscat_fn = lacscat


def cheatsheet() -> str:
    return "lacscat(y, W) -> Moran scatterplot quadrants and slope"
