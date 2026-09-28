"""Rook-contiguity spatial weights (grid)."""

from ._containers import SpatialResult


def swrook(nrow=4, ncol=4):
    """Rook-contiguity spatial weights of a regular ``nrow`` x ``ncol`` grid.

    Binary weights with cells numbered row by row, as ``spdep::cell2nb(type =
    "rook")`` (see :func:`morie.fn.swbuild.grid_contiguity`). ``statistic`` is
    the number of directed links ``S0``; ``extra["W"]`` holds the matrix and
    ``extra["cardinality"]`` the neighbour counts.

    Examples
    --------
    >>> r = swrook(2, 2)
    >>> r.statistic
    8.0
    """
    from .swbuild import grid_contiguity

    W = grid_contiguity(int(nrow), int(ncol), type="rook")
    card = [sum(row) for row in W]
    return SpatialResult(name="swrook", statistic=float(sum(card)), p_value=None, extra={"W": W, "cardinality": card})


swrook_fn = swrook


def cheatsheet() -> str:
    return "swrook({}) -> Rook-contiguity spatial weights (grid)."
