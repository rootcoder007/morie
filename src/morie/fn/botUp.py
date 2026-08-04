"""Bottom-up hierarchy aggregation."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["bottom_up_aggregation"]


def bottom_up_aggregation(bottoms, S):
    """
    Bottom-up reconciliation of a hierarchical forecast.

    Formula: ytilde = S yhat_bottom

    Verified against Hyndman & Athanasopoulos, FPP 3rd ed., Section 11.3
    "The bottom-up approach" -- source consulted at otexts.com/fpp3:
    "first generating forecasts for each series at the bottom level, and
    then summing these to produce forecasts for all the series".

    Parameters
    ----------
    bottoms : array-like
        Bottom-level forecasts, length ``m``.
    S : nested sequence
        Summing matrix, ``k x m``, whose rows give each series in the
        structure as a sum of bottom-level series.

    Returns
    -------
    RichResult
        Keys: estimate (all ``k`` coherent forecasts), total, bottom,
        n, method.

    References
    ----------
    Hyndman, R.J. & Athanasopoulos, G. Forecasting: Principles and
    Practice, 3rd ed. OTexts. Section 11.3.
    """
    bv = [float(t) for t in np.atleast_1d(np.asarray(bottoms, dtype=float))]
    Sm = _big2.mat(S)
    m = len(bv)
    if len(Sm[0]) != m:
        raise ValueError("S must have one column per bottom-level series")
    out = [sum(Sm[i][j] * bv[j] for j in range(m)) for i in range(len(Sm))]
    return RichResult(
        payload={
            "estimate": out,
            "total": sum(bv),
            "bottom": bv,
            "n": len(Sm),
            "method": "Bottom-up reconciliation S yhat -- Hyndman & Athanasopoulos, FPP3 Sec. 11.3",
        }
    )


def cheatsheet():
    return "botUp: Bottom-up hierarchy aggregation"
