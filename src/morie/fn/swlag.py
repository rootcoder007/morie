# morie.fn -- function file (rootcoder007/morie)
"""Spatial lag Wy."""

from .swops import lag_operator


def swlag(W, y):
    r"""Spatial lag ``Wy``, ``(Wy)_i = sum_j w_ij y_j`` (``spdep::lag.listw``).

    Thin front-end to :func:`morie.fn.swops.lag_operator`; returns the list of
    lagged values.

    References
    ----------
    Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.

    Examples
    --------
    >>> swlag([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]], [1.0, 2.0, 4.0])
    [2.0, 2.5, 2.0]
    """
    return lag_operator(W, y, 1)


swlag_fn = swlag


def cheatsheet() -> str:
    return "swlag(W, y) -> spatial lag Wy (spdep::lag.listw)."
