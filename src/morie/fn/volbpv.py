# morie.fn -- function file (rootcoder007/morie)
"""Bipower variation: jump-robust realised volatility."""

import numpy as np

from ._richresult import RichResult

__all__ = ["vol_bipower_variation"]


def vol_bipower_variation(r_intraday, day_index=None):
    r"""Barndorff-Nielsen-Shephard realised bipower variation.

    .. math:: BPV = \mu_1^{-2} \sum_{i=2}^{m} |r_i|\,|r_{i-1}|,
              \qquad \mu_1 = \sqrt{2/\pi},

    so the scaling constant is :math:`\pi/2`. Because a single jump
    enters each product only next to an (order dt) diffusive
    neighbour, BPV converges to the *integrated variance alone* while
    RV converges to integrated variance plus jump variation -- which
    is what makes RV - BPV a jump detector.

    Parameters
    ----------
    r_intraday : array-like, shape (m,)
        Intraday returns.
    day_index : array-like, optional
        Day label per return; products never straddle a day boundary.

    Returns
    -------
    RichResult
        keys: ``bpv`` (scalar or per day), ``days``, ``n_returns``,
        ``method``.

    References
    ----------
    Barndorff-Nielsen, O. E. & Shephard, N. (2004). Power and bipower
    variation with stochastic volatility and jumps. *Journal of
    Financial Econometrics*, 2(1), 1-48.
    """
    r = np.asarray(r_intraday, dtype=float).ravel()
    if r.size < 3:
        raise ValueError("need at least 3 intraday returns.")
    scale = np.pi / 2.0

    def one(v):
        return float(scale * np.sum(np.abs(v[1:]) * np.abs(v[:-1])))

    if day_index is None:
        return RichResult(
            payload={
                "bpv": one(r),
                "days": None,
                "n_returns": int(r.size),
                "method": "Realised bipower variation (single day)",
            }
        )
    d = np.asarray(day_index).ravel()
    if d.size != r.size:
        raise ValueError("day_index must have one entry per return.")
    days = list(dict.fromkeys(d.tolist()))
    bpv = np.array([one(r[d == day]) for day in days])
    return RichResult(
        payload={
            "bpv": bpv,
            "days": days,
            "n_returns": int(r.size),
            "method": "Realised bipower variation per day",
        }
    )


def cheatsheet():
    return "volbpv: (pi/2) sum |r_i||r_{i-1}| -- robust to jumps (BNS 2004)"
