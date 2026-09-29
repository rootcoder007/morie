# morie.fn -- function file (rootcoder007/morie)
"""Spherical (great-circle) distance weights."""

import math

from ._containers import SpatialResult
from ._qpcore import ssum


def _gc(lat1, lon1, lat2, lon2, radius):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(a)))


def swsph(lat, lon, d=500.0, radius=6371.0, style="B"):
    r"""Great-circle distance-band weights for longitude/latitude points.

    ``i ~ j`` when the haversine great-circle distance on a sphere of radius
    ``radius`` km is at most ``d`` km (Sinnott 1984); ``style="W"``
    row-standardises. ``extra["D"]`` holds the distance matrix.

    References
    ----------
    Sinnott, R. W. (1984). Virtues of the haversine. *Sky and Telescope* 68,
    159.

    Examples
    --------
    >>> r = swsph([43.65, 45.50, 49.28], [-79.38, -73.57, -123.12], d=600.0)
    >>> r.extra["neighbours"]
    [[1], [0], []]
    """
    la = [float(v) for v in lat]
    lo = [float(v) for v in lon]
    n = len(la)
    D = [[_gc(la[i], lo[i], la[j], lo[j], radius) for j in range(n)] for i in range(n)]
    W = [[1.0 if i != j and D[i][j] <= d else 0.0 for j in range(n)] for i in range(n)]
    if style == "W":
        W = [[v / ssum(r) if ssum(r) > 0 else 0.0 for v in r] for r in W]
    elif style != "B":
        raise ValueError("style must be 'B' or 'W'")
    nb = [[j for j in range(n) if W[i][j] != 0] for i in range(n)]
    return SpatialResult(
        name="swsph", statistic=ssum(len(v) for v in nb) / n, extra={"W": W, "D": D, "neighbours": nb, "style": style}
    )


swsph_fn = swsph


def cheatsheet() -> str:
    return "swsph(lat, lon, d=500) -> great-circle (haversine) distance-band weights, d in km."
