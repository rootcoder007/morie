"""Top-down disaggregation."""

from . import _array_core as np
from . import _big2 as _big2

from ._richresult import RichResult

__all__ = ["top_down"]


def top_down(top, props):
    """
    Top-down disaggregation of a hierarchical forecast.

    Formula: ytilde_j = p_j yhat_total

    Verified against Hyndman & Athanasopoulos, FPP 3rd ed., Section 11.3
    "Top-down approaches" -- source consulted at otexts.com/fpp3, where
    ``p_1, ..., p_m`` are "a set of disaggregation proportions which
    determine how the forecasts of the Total series are to be
    distributed".

    The proportions are closed to sum 1 so that the disaggregated
    forecasts add back to the total; the raw sum is reported so a
    caller who supplied proportions that did not sum to 1 can see it.

    Parameters
    ----------
    top : float
        Forecast of the Total series.
    props : array-like
        Non-negative disaggregation proportions.

    Returns
    -------
    RichResult
        Keys: estimate (bottom-level forecasts), props, prop_sum,
        total, n, method.

    References
    ----------
    Hyndman, R.J. & Athanasopoulos, G. Forecasting: Principles and
    Practice, 3rd ed. OTexts. Section 11.3.
    """
    t = float(top)
    pv = [float(v) for v in np.atleast_1d(np.asarray(props, dtype=float))]
    if min(pv) < 0.0:
        raise ValueError("proportions must be non-negative")
    raw = sum(pv)
    if not (raw > 0.0):
        raise ValueError("proportions must have positive total")
    p = [v / raw for v in pv]
    return RichResult(
        payload={
            "estimate": [t * v for v in p],
            "props": p,
            "prop_sum": raw,
            "total": t,
            "n": len(p),
            "method": "Top-down disaggregation p_j yhat -- Hyndman & Athanasopoulos, FPP3 Sec. 11.3",
        }
    )


def cheatsheet():
    return "topDn: Top-down disaggregation"


# compact alias per ledger/NAMING.md
topdown = top_down
