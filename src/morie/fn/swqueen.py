"""Queen-contiguity spatial weights (grid)."""

from ._containers import SpatialResult


def swqueen(nrow=4, ncol=4):
    """Queen-contiguity spatial weights of a regular ``nrow`` x ``ncol`` grid.

    Binary weights with cells numbered row by row, as ``spdep::cell2nb(type =
    "queen")`` (see :func:`morie.fn.swbuild.grid_contiguity`). ``statistic`` is
    the number of directed links ``S0``; ``extra["W"]`` holds the matrix and
    ``extra["cardinality"]`` the neighbour counts.

    Examples
    --------
    >>> r = swqueen(2, 2)
    >>> r.statistic
    12.0
    """
    from .swbuild import grid_contiguity

    W = grid_contiguity(int(nrow), int(ncol), type="queen")
    card = [sum(row) for row in W]
    return SpatialResult(name="swqueen", statistic=float(sum(card)), p_value=None, extra={"W": W, "cardinality": card})


swqueen_fn = swqueen


def cheatsheet() -> str:
    return "swqueen({}) -> Queen-contiguity spatial weights (grid)."
