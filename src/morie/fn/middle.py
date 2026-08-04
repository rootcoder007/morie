"""Middle-out hierarchy approach."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["middle_out"]


def middle_out(middle, S):
    """
    Middle-out reconciliation of a hierarchical forecast.

    Formula: aggregate upward from the middle level, disaggregate below

    Verified against Hyndman & Athanasopoulos, FPP 3rd ed., Section 11.3
    "Middle-out approach" -- source consulted at otexts.com/fpp3:
    "For the series above the middle level, coherent forecasts are
    generated using the bottom-up approach by aggregating the
    middle-level forecasts upwards. For the series below the middle
    level, coherent forecasts are generated using a top-down approach
    by disaggregating the middle-level forecasts downwards."

    Both directions are the single product ``S x`` once ``S`` is written
    with 0/1 rows for the levels above the middle and proportion rows
    for the levels below. Rows are classified by whether their entries
    are all 0 or 1 (aggregation) or not (disaggregation), and the two
    groups are reported separately so the reading stays checkable.

    Parameters
    ----------
    middle : array-like
        Forecasts at the chosen middle level, length ``m``.
    S : nested sequence
        Structure matrix, ``k x m``: 0/1 rows aggregate upward,
        fractional rows disaggregate downward.

    Returns
    -------
    RichResult
        Keys: estimate (all ``k`` coherent forecasts), aggregated,
        disaggregated, middle, n, method.

    References
    ----------
    Hyndman, R.J. & Athanasopoulos, G. Forecasting: Principles and
    Practice, 3rd ed. OTexts. Section 11.3.
    """
    mv = [float(t) for t in np.atleast_1d(np.asarray(middle, dtype=float))]
    Sm = _big2.mat(S)
    m = len(mv)
    if len(Sm[0]) != m:
        raise ValueError("S must have one column per middle-level series")
    out = []
    agg = []
    dis = []
    for i in range(len(Sm)):
        val = sum(Sm[i][j] * mv[j] for j in range(m))
        out.append(val)
        if all(Sm[i][j] in (0.0, 1.0) for j in range(m)):
            agg.append(val)
        else:
            dis.append(val)
    return RichResult(
        payload={
            "estimate": out,
            "aggregated": agg,
            "disaggregated": dis,
            "middle": mv,
            "n": len(Sm),
            "method": "Middle-out reconciliation -- Hyndman & Athanasopoulos, FPP3 Sec. 11.3",
        }
    )


def cheatsheet():
    return "middle: Middle-out hierarchy approach"


# compact alias per ledger/NAMING.md
middleout = middle_out
