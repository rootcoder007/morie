"""Join count test (BB, BW, WW)."""

from ._containers import SpatialResult
from .jjmsta import join_count


def lacjoin(y_binary, W):
    """Join count test for a binary map (Cliff and Ord 1981).

    The previous body returned ``y'Wy / y'y``, which is not a join count.
    This wraps :func:`morie.fn.jjmsta.join_count` (``spdep::joincount.test``
    moments under non-free sampling).

    Parameters
    ----------
    y_binary : array-like
        0/1 colour of each unit.
    W : array-like
        Spatial weights (diagonal ignored).

    Returns
    -------
    SpatialResult
        ``statistic`` is the black-black join count ``BB``, ``p_value`` its
        upper-tail normal p-value, ``expected`` ``E[BB]``, ``variance``
        ``Var[BB]``; ``extra`` holds every quantity of ``join_count``.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    >>> r = lacjoin([1, 1, 0, 0], W)
    >>> r.statistic, r.extra["WW"], r.extra["BW"]
    (1.0, 1.0, 1.0)
    """
    r = join_count(y_binary, W)
    return SpatialResult(
        name="lacjoin",
        statistic=float(r["BB"]),
        p_value=float(r["p_BB"]),
        expected=float(r["E_BB"]),
        variance=float(r["V_BB"]),
        extra=dict(r),
    )


lacjoin_fn = lacjoin


def cheatsheet() -> str:
    return "lacjoin(y, W) -> Join count test (BB, BW, WW; = spdep::joincount.test)."
