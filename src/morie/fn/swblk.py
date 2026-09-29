# morie.fn -- function file (rootcoder007/morie)
"""Block spatial weights (group-based)."""

from ._containers import SpatialResult
from ._qpcore import ssum
from .swops import block_weights


def swblk(groups, style="B"):
    r"""Block (group-based) spatial weights: ``w_ij = 1`` for distinct units sharing a group.

    As ``spdep::nb2blocknb`` without prior neighbours (thin front-end to
    :func:`morie.fn.swops.block_weights`); ``style="W"`` row-standardises
    (singleton groups keep a zero row). Used for regional or regime
    interaction schemes (Case 1991).

    References
    ----------
    Case, A. C. (1991). Spatial patterns in household demand. *Econometrica*
    59, 953-965.

    Examples
    --------
    >>> swblk(["a", "b", "a", "a"], style="W").extra["W"][0]
    [0.0, 0.0, 0.5, 0.5]
    """
    W = block_weights(groups)
    if style == "W":
        W = [[v / ssum(r) if ssum(r) > 0 else 0.0 for v in r] for r in W]
    elif style != "B":
        raise ValueError("style must be 'B' or 'W'")
    nb = [[j for j, v in enumerate(r) if v != 0] for r in W]
    return SpatialResult(
        name="swblk", statistic=ssum(len(v) for v in nb) / len(W), extra={"W": W, "neighbours": nb, "style": style}
    )


swblk_fn = swblk


def cheatsheet() -> str:
    return "swblk(groups, style='B') -> block weights linking units in the same group (spdep::nb2blocknb)."
