# morie.fn -- function file (rootcoder007/morie)
"""Realised variance from intraday returns."""

import numpy as np

from ._richresult import RichResult

__all__ = ["vol_realised_variance"]


def vol_realised_variance(r_intraday, day_index=None):
    r"""Daily realised variance.

    .. math:: RV_d = \sum_i r_{d,i}^2,

    the sum of squared intraday returns per day -- the canonical
    nonparametric estimate of that day's integrated variance under
    no-noise, no-jump sampling.

    Parameters
    ----------
    r_intraday : array-like, shape (m,)
        Intraday returns, all days concatenated.
    day_index : array-like, optional
        Day label per return; None treats the input as one day.

    Returns
    -------
    RichResult
        keys: ``rv`` (per day, in day-label order of first
        appearance), ``days``, ``n_returns``, ``method``.

    References
    ----------
    Barndorff-Nielsen, O. E. & Shephard, N. (2004). Power and bipower
    variation with stochastic volatility and jumps. *Journal of
    Financial Econometrics*, 2(1), 1-48. (RV as the QV estimator the
    bipower measures are compared against)
    """
    r = np.asarray(r_intraday, dtype=float).ravel()
    if r.size < 2:
        raise ValueError("need at least 2 intraday returns.")
    if day_index is None:
        return RichResult(
            payload={
                "rv": float((r**2).sum()),
                "days": None,
                "n_returns": int(r.size),
                "method": "Realised variance (single day)",
            }
        )
    d = np.asarray(day_index).ravel()
    if d.size != r.size:
        raise ValueError("day_index must have one entry per return.")
    days = list(dict.fromkeys(d.tolist()))
    rv = np.array([float((r[d == day] ** 2).sum()) for day in days])
    return RichResult(
        payload={
            "rv": rv,
            "days": days,
            "n_returns": int(r.size),
            "method": "Realised variance per day",
        }
    )


def cheatsheet():
    return "volrv: RV_d = sum of squared intraday returns"
