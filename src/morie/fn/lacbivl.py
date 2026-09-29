# morie.fn -- function file (rootcoder007/morie)
"""Bivariate LISA (Lee 2001)."""

from __future__ import annotations

from . import _lattice as lat
from ._containers import SpatialResult


def lacbivl(x, y, W):
    r"""Bivariate LISA (Lee 2001).

    Lee's bivariate spatial association ``L = n / sum_i W_i^2 * sum_i
    (W z_x)_i (W z_y)_i / (|z_x| |z_y|)`` with ``z = v - vbar`` and ``W_i =
    sum_j w_ij``, and its local components ``L_i = n (W z_x)_i (W z_y)_i /
    (|z_x| |z_y|)`` (Lee 2001) -- exactly ``spdep::lee``. ``L`` combines
    Pearson's r of the smoothed variables with their spatial smoothing
    scalars.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        The two variables.
    W : array-like, shape (n, n)
        Spatial weights.

    Returns
    -------
    SpatialResult
        ``statistic`` L; ``local_values`` the ``L_i``.

    References
    ----------
    Lee, S.-I. (2001). Developing a bivariate spatial association measure: an integration of
    Pearson's r and Moran's I. *Journal of Geographical Systems*, 3(4), 369-385.

    Examples
    --------
    >>> W = [[0, 1, 0, 0, 0, 0], [.5, 0, .5, 0, 0, 0], [0, .5, 0, .5, 0, 0], [0, 0, .5, 0, .5, 0], [0, 0, 0, .5, 0, .5], [0, 0, 0, 0, 1, 0]]
    >>> r = lacbivl([0.5, 0.9, 0.2, 1.4, 0.8, 1.1], [1.0, 2.4, 1.3, 3.1, 1.9, 2.2], W)
    >>> round(r.statistic, 12)
    0.671140866213
    """
    L, loc = lat.lee_l(lat.vec(x), lat.vec(y), lat.mat(W))
    return SpatialResult(name="lacbivl", statistic=L, local_values=loc)


lacbivl_fn = lacbivl


def cheatsheet() -> str:
    return "lacbivl(x, y, W) -> Lee's bivariate L (spdep::lee)"
