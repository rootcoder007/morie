# morie.fn -- function file (rootcoder007/morie)
"""Inverse-distance spatial weights."""

from .spwgt import spatial_weights


def swinv(coords, power=1.0, d=None, style="B"):
    r"""Inverse-distance weights ``w_ij = d_ij^(-power)`` for ``0 < d_ij <= d`` (all pairs when ``d`` is None).

    Thin front-end to :func:`morie.fn.spwgt.spatial_weights` (method
    ``"inverse"``); ``style`` is an ``spdep::nb2listw`` coding (``B`` binary,
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
    >>> r = swinv(coords, power=2.0, d=1.2)
    >>> [sorted(v) for v in r.extra["neighbours"]][5]
    [1, 3]
    """
    r = spatial_weights(
        coords, "inverse", threshold=float("inf") if d is None else float(d), style=style, alpha=float(power)
    )
    r.name = "swinv"
    return r


swinv_fn = swinv


def cheatsheet() -> str:
    return "swinv(coords, power=1.0) -> inverse-distance weights d^-power."
