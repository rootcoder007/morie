# morie.fn -- function file (rootcoder007/morie)
"""Spatial correlation coefficient."""

from __future__ import annotations

import math

from . import _lattice as lat
from ._containers import SpatialResult


def lacscor(y, W):
    r"""Spatial correlation coefficient.

    Pearson's correlation between ``y`` and its spatial lag ``Wy``, the
    correlation displayed by the Moran scatterplot (Anselin 1996); unlike
    Moran's I (the regression slope) it is bounded by 1 in absolute value.
    Also returned: Lee's (2001) ``L_yy``, the spatial smoothing scalar
    ``n / sum_i W_i^2 * sum_i (Wz)_i^2 / sum z^2``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Variable observed on the n lattice units.
    W : array-like, shape (n, n)
        Spatial weights (zero diagonal).

    Returns
    -------
    SpatialResult
        ``statistic`` is ``cor(y, Wy)``; ``extra["lee_L"]``.

    References
    ----------
    Anselin, L. (1996). The Moran scatterplot as an ESDA tool to assess local instability in spatial
    association. In Fischer, Scholten and Unwin (eds), *Spatial Analytical Perspectives on GIS*,
    111-125. Taylor and Francis.

    Lee, S.-I. (2001). Developing a bivariate spatial association measure: an integration of
    Pearson's r and Moran's I. *Journal of Geographical Systems*, 3(4), 369-385.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> round(lacscor([1.0, 2.4, 1.3, 3.1, 1.9, 2.2], W).statistic, 12)
    -0.738389785723
    """
    yv, Wm = lat.vec(y), lat.mat(W)
    wy = lat.mv(Wm, yv)
    a, b = lat.centre(yv), lat.centre(wy)
    r = lat.ssum(u * v for u, v in zip(a, b)) / math.sqrt(lat.ssum(u * u for u in a) * lat.ssum(v * v for v in b))
    L, _ = lat.lee_l(yv, yv, Wm)
    return SpatialResult(name="lacscor", statistic=r, extra={"lee_L": L})


lacscor_fn = lacscor


def cheatsheet() -> str:
    return "lacscor(y, W) -> cor(y, Wy) and Lee's L_yy"
